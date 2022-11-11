# Affectnet dataset configuration

```bash
AFFNet
├── train_set
│   ├── images
│   ├── annotations
│   └── train_csv.csv(optional)
├── val_set
│   ├── images
│   ├── annotations
│   ├── val_csv.csv(optional)
``` 

You have to download AffectNet dataset and make directory structure like above, also you need to change configuration file(.yaml) in config folder.
csv file is generated in order to bring dataset from directory more efficiently. I'll just give you files, however you should modify csv file in your training environment.
Create csv file with create_csv.py with following code.

## Training set
```python
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

if __name__ == "__main__":
    rootdir = os.path.abspath("./images")
    labeldir = os.path.abspath("./annotations")
    information = {"images":[], "labels":[], "lnds":[]}

    for file in tqdm(os.listdir(rootdir)):
        filename, fileext = os.path.splitext(file)
        if fileext == ".jpg":
            information["images"] += [os.path.join(rootdir, file)]
            information["labels"] += [np.load(os.path.join(labeldir, filename+"_exp.npy"))]
            information["lnds"] += [np.load(os.path.join(labeldir, filename+"_lnd.npy"))]

    df = pd.DataFrame.from_dict(information)
    df.to_csv("train_dataset.csv")


```

## Validation set
```python
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

if __name__ == "__main__":
    rootdir = os.path.abspath("./images")
    labeldir = os.path.abspath("./annotations")
    information = {"images":[], "labels":[], "lnds":[]}

    for file in tqdm(os.listdir(rootdir)):
        filename, fileext = os.path.splitext(file)
        if fileext == ".jpg":
            information["images"] += [os.path.join(rootdir, file)]
            information["labels"] += [np.load(os.path.join(labeldir, filename+"_exp.npy"))]
            information["lnds"] += [np.load(os.path.join(labeldir, filename+"_lnd.npy"))]

    df = pd.DataFrame.from_dict(information)
    df.to_csv("val_dataset.csv")


```
