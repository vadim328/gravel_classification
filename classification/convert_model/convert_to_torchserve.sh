set -ex

python /mmpretrain/tools/torchserve/mmpretrain2torchserve.py \
       data/GravelClassificationDeit3s.py \
       models/GravelClassificationDeit3s_100e.pth \
       --output-folder /torchserve/model_store \
       --model-name GravelClassificationDeit3s