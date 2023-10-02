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
    collated_data = {}
    for key in batch[0]:
        if isinstance(batch[0][key], torch.Tensor):
            collated_data[key] = torch.stack([item[key] for item in batch], dim=0)
        elif isinstance(batch[0][key], np.ndarray):
            collated_data[key] = torch.stack([torch.from_numpy(item[key]) for item in batch], dim=0)

    return collated_data

class Mono3DNode:
    def __init__(self):
        rospy.init_node("Mono3DNode")
        rospy.loginfo("Starting Mono3DNode.")

        self._read_params()
        self._init_model()
        self._init_static_memory()
        self._init_topics()

    def _read_params(self):
        rospy.loginfo("Reading params.")
        visual3d_path = rospy.get_param("~VISUAL3D_PATH", "/home/yxliu/vision_factory")
        import sys
        sys.path.append(visual3d_path)
        from vision_base.utils.utils import cfg_from_file

        cfg_file = rospy.get_param("~CFG_FILE", "/home/yxliu/multi_cam/monoflex.py")
        self.cfg = cfg_from_file(cfg_file)

        self.weight_path = rospy.get_param("~WEIGHT_PATH", "/home/yxliu/multi_cam/monoflex.pth")

        self.inference_w   = int(rospy.get_param("~INFERENCE_W",  1280))
        self.inference_h   = int(rospy.get_param("~INFERENCE_H",  288))
        self.cfg.val_dataset.augmentation.cfg_list[1].size = (self.inference_h, self.inference_w)

    def _init_model(self):
        rospy.loginfo("Loading model.")
        from vision_base.utils.builder import build

        self.meta_arch = build(**self.cfg.meta_arch)
        self.meta_arch = self.meta_arch.cuda()
        state_dict = torch.load(
            self.weight_path, map_location='cuda:{}'.format(self.cfg.trainer.gpu)
        )
        self.meta_arch.load_state_dict(state_dict['model_state_dict'], strict=False)
        self.meta_arch.eval()
        
        # self.ort_session = ort.InferenceSession(self.onnx_path)

        self.transform = build(**self.cfg.val_dataset.augmentation)
        self.test_pipeline = build(**self.cfg.trainer.evaluate_hook.test_run_hook_cfg)
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
        data = dict()
        data['P'] = self.P.copy()
        data['original_P'] = self.P.copy()
        data['image'] = image
        data = self.transform(data)
        data = collate_fn([data])
        
        with torch.no_grad():
            output_dict = self.test_pipeline(data, self.meta_arch)
            scores = output_dict['scores'].cpu().numpy()
            bboxes = output_dict['bboxes'].cpu().numpy()
            #[0:4] x1 y1 x2 y2 #[4:]  x, y, z, w, h, l, alpha, theta
            #      0  1  2  3         4  5  6  7  8  9   10     11
            cls_names = output_dict['cls_names'] # nuscene classes
            objects = []
            N = len(bboxes)
            for i in range(N):
                obj = {}
                obj['whl'] = bboxes[i, 7:10]
                obj['theta'] = bboxes[i, 11]
                obj['score'] = scores[i]
                obj['type_name'] = cls_names[i]
                obj['xyz'] = bboxes[i, 4:7]
                objects.append(obj)

        return objects

    def camera_callback(self, msg):
        height = msg.height
        width  = msg.width
        if self.P is not None:
            image = np.frombuffer(msg.data, dtype=np.uint8).reshape([height, width, 3]) #[BGR]
            objects = self._predict(image[:, :, ::-1].copy()) # BGR -> RGB
            # clear_all_bbox(self.bbox_publish)
            self.bbox_publish.publish([object_to_marker(obj, marker_id=i, duration=0.5, frame_id=self.frame_id, use_nusc_color_map=True) for i, obj in enumerate(objects)])

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
    ros_node = Mono3DNode()
    rospy.spin()