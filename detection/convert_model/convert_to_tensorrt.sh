set -ex

mkdir -p ../yolov8_tensorRT
python /openmmlab/mmdeploy/tools/deploy.py \
  data/deploy.py \
  data/yolov8.py \
  models/yolo8_epoch_196.pth \
  data/images/1af2679c-0-5_32.jpeg \
  --device cuda:0 \
  --work-dir ../yolov8_tensorRT \
  --dump-info