# A general remote photoplethysmography (rPPG) training framework

# 0. Create conda env
```
$ conda create -n torchrppg python=3.7
$ conda activate torchrppg
$ pip install -r requirements.txt
```
# Prepare data
## download prepcessed UBFC dataset png data.
链接: https://pan.baidu.com/s/1AK9nBNRPR78iXoM-qq13Lw 提取码: vb8e
## generate data from png
```
$ python utils/ubfc_png2pth.py
```

# 2. How to config and train it?

## modify your own config.json
modify data path as yours

## train
```
$ python train.py
```

# 3. How to evalate?
```
$ python utils/ubfc_evaluation.py
```

# 4. How to run with webcam?
```
$ cd deploy
$ python demo/webcam_demo.py
```
