python -m vit.vit \
  --data-path data/imagenette2-320/ \
  --neptune \
  --gpus 1 \
  --pretrained \
  --batch-size 64 \
  --arch vit_base_patch16_224


python -m vit.vit \
  --data-path data/imagenette2-320/ \
  --neptune \
  --gpus 1 \
  --pretrained \
  --batch-size 64 \
  --auto_lr_find True \
  --wd 0.0001 \
  --arch vit_base_patch16_224