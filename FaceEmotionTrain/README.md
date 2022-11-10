# Training baseline model

```bash
python train.py --device cuda
```

If you want to change training configuration, add ocnfiguration file(.yaml)

```bash
config
├── YOUR_CONFIG_NAME.yaml
``` 
and train model with,
```bash
python train.py --device cuda --config YOUR_CONFIG_NAME
```
# Ablation study(experiments) example
```bash
python train.py --device cuda --config YOUR_CONFIG_NAME --save checkpoint/YOUR_CONFIG_NAME --exp EXPERIMENT_NUMBER
```

# Dataset distribution
Use [Affect-Net dataset](https://paperswithcode.com/dataset/affectnet)  
You need to read README file in AFFNet directory. [Here](https://github.com/YAICON-HOTFACE/FER/tree/main/FaceEmotionTrain/AFFNet)
| Training dataset | Validation dataset |
|------------|-------------|
| <p align="center"><img src="train_dist.png"  width="450"></p> | <p align="center"><img src="val_dist.png"  width="450"></p> |

