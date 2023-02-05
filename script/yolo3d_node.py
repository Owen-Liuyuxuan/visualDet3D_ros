#!/usr/bin/env python3
import numpy as np
import cv2
import torch
import torch.nn.functional as F

import rospy
from sensor_msgs.msg import Image, CameraInfo
from ros_util import object_to_marker, clear_single_box, clear_all_bbox
from visualization_msgs.msg import MarkerArray


def collate_fn(batch):
    rgb_images = np.array([item["image"] for item in batch])#[batch, H, W, 3]
    rgb_images = rgb_images.transpose([0, 3, 1, 2])

    calib = [item["calib"] for item in batch]
    return torch.from_numpy(rgb_images).float(), torch.tensor(calib).float()

class Yolo3DNode:
    def __init__(self):
        rospy.init_node("ros_node")
        rospy.loginfo("Starting Yolo3DNode.")

        self._read_params()
        self._init_model()
        self._init_static_memory()
        self._init_topics()

    def _read_params(self):
        rospy.loginfo("Reading params.")
        visual3d_path = rospy.get_param("~VISUAL3D_PATH", "/home/yxliu/IROS_try/visualDet3D_open")
        import sys
        sys.path.append(visual3d_path)
        from visualDet3D.utils.utils import cfg_from_file
        from visualDet3D.networks.utils.registry import DETECTOR_DICT, DATASET_DICT, PIPELINE_DICT
        from visualDet3D.networks.lib.fast_utils.hill_climbing import post_opt
        from visualDet3D.networks.utils import BBox3dProjector, BackProjection
        from visualDet3D.utils.utils import convertAlpha2Rot, convertRot2Alpha, draw_3D_box, compound_annotation
        import visualDet3D.data.kitti.dataset
        from visualDet3D.data.pipeline import build_augmentator


        cfg_file = rospy.get_param("~CFG_FILE", "/home/yxliu/IROS_try/visualDet3D_open/config/kitti_yolo3d_copied.py")
        self.cfg = cfg_from_file(cfg_file)
        self.cfg.detector.backbone.pretrained=False

        self.weight_path = rospy.get_param("~WEIGHT_PATH", "/home/yxliu/IROS_try/visualDet3D_open/workdirs/Mono3D/checkpoint/GroundAwareYolo3D_latest.pth")

        self.inference_w   = int(rospy.get_param("~INFERENCE_W",  1280))
        self.inference_h   = int(rospy.get_param("~INFERENCE_H",  288))
        self.crop_top      = int(rospy.get_param("~CROP_TOP", 100))
        self.inference_scale = float(rospy.get_param("~INFERENCE_SCALE", 1.0))
        self.cfg.data.test_augmentation[1].keywords.crop_top_index = self.crop_top 
        self.cfg.data.test_augmentation[2].keywords.size = (self.inference_h, self.inference_w)

        self.projector = BBox3dProjector().cuda()
        self.backprojector = BackProjection().cuda()

    def _init_model(self):
        rospy.loginfo("Loading model.")
        from visualDet3D.networks.utils.registry import DETECTOR_DICT, PIPELINE_DICT
        from visualDet3D.data.pipeline import build_augmentator

        detector = DETECTOR_DICT[self.cfg.detector.name](self.cfg.detector)
        self.detector = detector.cuda()
        state_dict = torch.load(
            self.weight_path, map_location='cuda:{}'.format(self.cfg.trainer.gpu)
        )
        self.detector.load_state_dict(state_dict, strict=False)
        self.detector.eval()
        
        # self.ort_session = ort.InferenceSession(self.onnx_path)

        self.transform = build_augmentator(self.cfg.data.test_augmentation)
        self.test_func = PIPELINE_DICT[self.cfg.trainer.test_func]
        rospy.loginfo("Done loading model.")

    def _init_static_memory(self):
        self.frame_id = None
        self.P = None
        self.num_objects = 0

    def _init_topics(self):
        self.bbox_publish        = rospy.Publisher("/bboxes", MarkerArray, queue_size=1, latch=True)
        rospy.Subscriber("/image_raw", Image, self.camera_callback, buff_size=2**26, queue_size=1)
        rospy.Subscriber("/camera_info", CameraInfo, self.camera_info_callback)
        clear_all_bbox(self.bbox_publish)

    def _predict(self, image):
        transformed_image, transformed_P2 = self.transform(image.copy(), p2=self.P.copy())
        data = {'calib': transformed_P2,
                       'image': transformed_image,
                       'original_shape':image.shape,
                       'original_P':self.P.copy()}
        data = collate_fn([data])
        
        with torch.no_grad():
            
            scores, bbox, obj_names = self.test_func(data, self.detector, None, cfg=self.cfg)
            bbox_2d = bbox[:, 0:4]
            bbox_3d_state = bbox[:, 4:] #[cx,cy,z,w,h,l,alpha]
            bbox_3d_state[:, 2] *= self.inference_scale
            bbox_3d_state_3d = self.backprojector(bbox_3d_state, transformed_P2) #[x, y, z, w,h ,l, alpha]
            abs_bbox, bbox_3d_corner_homo, thetas = self.projector(bbox_3d_state_3d, bbox_3d_state_3d.new(transformed_P2))

            original_P = self.P
            scale_x = original_P[0, 0] / transformed_P2[0, 0]
            scale_y = original_P[1, 1] / transformed_P2[1, 1]
            
            shift_left = original_P[0, 2] / scale_x - transformed_P2[0, 2]
            shift_top  = original_P[1, 2] / scale_y - transformed_P2[1, 2]
            bbox_2d[:, 0:4:2] += shift_left
            bbox_2d[:, 1:4:2] += shift_top

            bbox_2d[:, 0:4:2] *= scale_x
            bbox_2d[:, 1:4:2] *= scale_y

            bbox_2d = bbox_2d.cpu().numpy()
            bbox_3d_state_3d = bbox_3d_state_3d.cpu().numpy()
            thetas = thetas.cpu().numpy()

            objects = []
            N = len(bbox)
            for i in range(N):
                obj = {}
                obj['whl'] = bbox_3d_state_3d[i, 3:6]
                obj['theta'] = thetas[i]
                obj['score'] = scores[i]
                obj['type_name'] = obj_names[i]
                obj['xyz'] = bbox_3d_state_3d[i, 0:3]
                objects.append(obj)

        return objects

    def camera_callback(self, msg):
        height = msg.height
        width  = msg.width
        if self.P is not None:
            image = np.frombuffer(msg.data, dtype=np.uint8).reshape([height, width, 3]) #[BGR]
            objects = self._predict(image[:, :, ::-1].copy()) # BGR -> RGB
            # clear_all_bbox(self.bbox_publish)
            self.bbox_publish.publish([object_to_marker(obj, marker_id=i, duration=10, frame_id=self.frame_id) for i, obj in enumerate(objects)])

        N0 = len(objects)
        if N0 < self.num_objects:
            for i in range(N0, self.num_objects):
                clear_single_box(self.bbox_publish, marker_id=i)
        self.num_objects = N0

    def camera_info_callback(self, msg):
        self.P = np.zeros((3, 4))
        self.P[0:3, 0:3] = np.array(msg.K).reshape((3, 3))
        self.frame_id = msg.header.frame_id

if __name__ == "__main__":
    ros_node = Yolo3DNode()
    rospy.spin()