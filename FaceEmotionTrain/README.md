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
