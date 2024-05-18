import base64
import os
from typing import Dict, Any, List

import mmcv
import numpy as np
import torch
from ts.torch_handler.base_handler import BaseHandler

from mmdeploy.apis.utils import build_task_processor
from mmdeploy.utils import get_input_shape, load_config


class MMdetHandler(BaseHandler):
    threshold = 0.4

    def initialize(self, context):
        properties = context.system_properties
        self.manifest = context.manifest
        model_dir = properties.get('model_dir')
        serialized_file = self.manifest['model']['serializedFile']
        model_cfg = os.path.join(model_dir, 'config.py')
        deploy_cfg = os.path.join(model_dir, 'deploy.py')
        backend_model = [os.path.join(model_dir, serialized_file)]

        self.device = 'cuda:' + str(properties.get('gpu_id'))

        # read deploy_cfg and model_cfg
        deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)
        self.input_shape = get_input_shape(deploy_cfg)
        self.classes = model_cfg["classes"]

        # build task and backend model
        self.task_processor = build_task_processor(model_cfg, deploy_cfg, self.device)
        self.model = self.task_processor.build_backend_model(backend_model)
        self.initialized = True

    def preprocess(self, data) -> List[Dict[str, Any]]:
        images = []
        for row in data:
            image = row.get('data') or row.get('body')
            if isinstance(image, str):
                image = base64.b64decode(image)
            image = mmcv.imfrombytes(image)
            model_inputs, _ = self.task_processor.create_input(image, self.input_shape)
            images.append(model_inputs)
        return images

    def inference(self, data, *args, **kwargs):
        result_list = []
        with torch.no_grad():
            for i in data:
                result = self.model.test_step(i)
                result_list.extend(result)
        return result_list

    def postprocess(self, data):
        # Format output following the example ObjectDetectionHandler format
        output = []
        for data_sample in data:
            pred_instances = data_sample.pred_instances
            bboxes = pred_instances.bboxes.cpu().numpy().astype(np.float32).tolist()
            labels = pred_instances.labels.cpu().numpy().astype(np.int32).tolist()
            scores = pred_instances.scores.cpu().numpy().astype(np.float32).tolist()
            preds = []
            for idx in range(len(labels)):
                cls_score, bbox, cls_label = scores[idx], bboxes[idx], labels[
                    idx]
                if cls_score >= self.threshold:
                    class_name = self.classes[cls_label]
                    result = dict(
                        class_label=cls_label,
                        class_name=class_name,
                        bbox=bbox,
                        score=cls_score)
                    preds.append(result)
            output.append(preds)
        return output
