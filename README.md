# TDM_RPPG: Efficient Remote Photoplethysmography with Temporal Derivative Modules and Time-Shift Invariant Loss

The unofficial implementation by pytorch.


# 0. create conda env
```
$ conda create -n tdm_rppg python=3.7
$ conda activate tdm_rppg
$ pip install -r requirements.txt
```
# 1. generate data
```
$ python utils/png2pth.py
```

# 2. How to config and train it?

## modify your own config.json
modify data path as yours

## train
```
$ python train.py
```

