# FlashKAT: Understanding and Addressing Performance Bottlenecks in the Kolmogorov-Arnold Transformer
## 🛠️ Installation and Dataset
Install the required dependencies with the following command.
```shell
# install torch and other things
pip install -r requirements.txt
```

📦 Data preparation: ImageNet with the following folder structure, you can extract ImageNet by this [script](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4)

```
│imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```

## 🎓Model Training

All training scripts are under `scripts/`
```shell
bash scripts/train_kat_tiny_8x128.sh
```
All our training was done on a single GPU.  Below is a distributed training script for FlashKAT.  If you want to change the hyper-parameters, can edit
```shell
#!/bin/bash
DATA_PATH=/local_home/dataset/imagenet/

bash ./dist_train.sh 8 $DATA_PATH \
    --model flash_kat_tiny_swish_patch16_224 \ # Rationals are initialized to be swish functions 
    -b 128 \
    --opt adamw \
    --lr 1e-3 \
    --weight-decay 0.05 \
    --epochs 300 \
    --mixup 0.8 \
    --cutmix 1.0 \
    --sched cosine \
    --smoothing 0.1 \
    --drop-path 0.1 \
    --aa rand-m9-mstd0.5 \
    --remode pixel --reprob 0.25 \
    --amp \
    --crop-pct 0.875 \
    --mean 0.485 0.456 0.406 \
    --std 0.229 0.224 0.225 \
    --model-ema \
    --model-ema-decay 0.9999 \
    --output output/kat_tiny_swish_patch16_224 \
    --model-kwargs m=6 n=4 num_groups=1 den_groups=8 \
    --log-wandb
```

The training script for KAT.  If you want to change the hyper-parameters, can edit
```shell
#!/bin/bash
DATA_PATH=/local_home/dataset/imagenet/

bash ./dist_train.sh 8 $DATA_PATH \
    --model kat_tiny_swish_patch16_224 \ # Rationals are initialized to be swish functions 
    -b 128 \
    --opt adamw \
    --lr 1e-3 \
    --weight-decay 0.05 \
    --epochs 300 \
    --mixup 0.8 \
    --cutmix 1.0 \
    --sched cosine \
    --smoothing 0.1 \
    --drop-path 0.1 \
    --aa rand-m9-mstd0.5 \
    --remode pixel --reprob 0.25 \
    --amp \
    --crop-pct 0.875 \
    --mean 0.485 0.456 0.406 \
    --std 0.229 0.224 0.225 \
    --model-ema \
    --model-ema-decay 0.9999 \
    --output output/kat_tiny_swish_patch16_224 \
    --log-wandb
```

## 🧪 Evaluation
To evaluate our `kat_tiny_patch16_224` models, run:

```shell
DATA_PATH=/local_home/dataset/imagenet/
CHECKPOINT_PATH=kat_tiny_patch16_224_1f3ad3b2e69821f3d412f2924cf159a0e266f142d739cb68f68f796f5a0fe289.pth
python validate.py $DATA_PATH --model kat_tiny_patch16_224 \
    --checkpoint $CHECKPOINT_PATH -b 512

###################
Validating in float32. AMP not enabled.
Loaded state_dict from checkpoint 'kat_tiny_patch16_224_1f3ad3b2e69821f3d412f2924cf159a0e266f142d739cb68f68f796f5a0fe289.pth'
Model kat_tiny_patch16_224 created, param count: 5718328
Data processing configuration for current model + dataset:
        input_size: (3, 224, 224)
        interpolation: bicubic
        mean: (0.485, 0.456, 0.406)
        std: (0.229, 0.224, 0.225)
        crop_pct: 0.875
        crop_mode: center
Test: [   0/98]  Time: 3.453s (3.453s,  148.28/s)  Loss:  0.6989 (0.6989)  Acc@1:  84.375 ( 84.375)  Acc@5:  96.875 ( 96.875)
.......
Test: [  90/98]  Time: 0.212s (0.592s,  864.23/s)  Loss:  1.1640 (1.1143)  Acc@1:  71.875 ( 74.270)  Acc@5:  93.750 ( 92.220)
 * Acc@1 74.558 (25.442) Acc@5 92.390 (7.610)
--result
{
    "model": "kat_tiny_patch16_224",
    "top1": 74.558,
    "top1_err": 25.442,
    "top5": 92.39,
    "top5_err": 7.61,
    "param_count": 5.72,
    "img_size": 224,
    "crop_pct": 0.875,
    "interpolation": "bicubic"
}
```


## 🙏 Acknowledgments
We extend our gratitude to the authors of [KAT](https://github.com/Adamdad/kat) for their implementation of the Kolmogorov-Arnold Transformer.

## 📚 Bibtex
If you use this repository, please cite the our work:
```bibtex
@article{raffel2025flashkat,
  title={FlashKAT: Understanding and Addressing Performance Bottlenecks in the Kolmogorov-Arnold Transformer},
  author={Raffel, Matthew and Chen, Lizhong},
  journal={arXiv preprint arXiv:2505.13813},
  year={2025}
}
```

Also, please cite the original Kolmogorov-Arnold Transformer:
```bibtex
@inproceedings{
  yang2025kolmogorovarnold,
  title={Kolmogorov-Arnold Transformer},
  author={Xingyi Yang, Xinchao Wang},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=BCeock53nt}
}
```
