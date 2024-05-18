set -ex

python /openmmlab/mmdetection/tools/deployment/mmdet2torchserve.py \
  data/yolov8.py \
  ../yolov8_tensorRT/end2end.engine \
  --output-folder /openmmlab/torchserve/model_store/ \
  --model-name GravelDetectionYOLOv8s \
  --extra-files data/deploy.py \
  --handler data/handler.py \
  -f