# FILTER
The code runs on Python 3.10.6 with PyTorch 2.0.1 and torchvision 0.15.2.
## Setup

```bash
conda create -n FILTER python=3.10
conda activate FILTER
pip install -r requirements.txt
```

## Backspin
For each Python file, there is a corresponding YAML file serving as its configuration file. To change the configuration file information, simply execute the following command to run.

```bash
python xxx.py  # for example, python VILLAIN_Mask_CIFAR10.py
```