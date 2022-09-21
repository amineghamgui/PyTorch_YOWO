# YOWO
Big thanks to [YOWO](https://github.com/wei-tim/YOWO) for their open source. I reimplemented ```YOWO``` and reproduced the performance. On the ```AVA``` dataset, my reproduced YOWO is better than the official YOWO. I hope that such a real-time action detector with simple structure and superior performance can attract your interest in the task of spatio-temporal action detection.

# Requirements
- We recommend you to use Anaconda to create a conda environment:
```Shell
conda create -n yowo python=3.6
```

- Then, activate the environment:
```Shell
conda activate yowo
```

- Requirements:
```Shell
pip install -r requirements.txt 
```

# Dataset

## UCF101-24 & JHMDB21
You can download **UCF24** and **JHMDB21** from the following links:

### Google Drive
For UCF24:

Link: https://drive.google.com/file/d/1Dwh90pRi7uGkH5qLRjQIFiEmMJrAog5J/view?usp=sharing

For JHMDB21: 

Link: https://drive.google.com/file/d/15nAIGrWPD4eH3y5OTWHiUbjwsr-9VFKT/view?usp=sharing

### BaiduYunDisk
For UCF24:

Link: https://pan.baidu.com/s/11GZvbV0oAzBhNDVKXsVGKg

Password: hmu6 

For JHMDB21: 

Link: https://pan.baidu.com/s/1HSDqKFWhx_vF_9x6-Hb8jA 

Password: tcjd 

## AVA
You can use instructions from [here](https://github.com/yjh0410/AVA_Dataset) to prepare **AVA** dataset.

# Experiment
## UCF24
|    Model    |    Frame mAP    |   FPS   |    Cls Accu    |    Recall    |    Weight    |
|-------------|-----------------|---------|----------------|--------------|--------------|
|    YOWO     |      80.4       |    -    |      94.5      |      93.5    |       -      |
| YOWO (Ours) |      82.5       |    36   |      93.8      |      95.6    | [github](https://github.com/yjh0410/PyTorch_YOWO/releases/download/yowo-weight/yowo_80.4.pth)   |


## Train on UCF24

```Shell
python train.py --cuda -d ucf24 -v yowo --num_workers 4 --eval_epoch 1 --eval
```

or you can just run the script:

```Shell
sh train_ucf.sh
```

##  Test on UCF24

```Shell
python test.py --cuda -d ucf24 -v yowo --weight path/to/weight --show
```

## Evaluate on UCF24
* on UCF24

```Shell
python eval.py \
        --cuda \
        -d ucf24 \
        -v yowo \
        --gt_folder ./evaluator/groundtruths_ucf_jhmdb/groundtruths_ucf/ \
        --weight path/to/weight \
        --cal_mAP \
        --redo
```

Our YOWO on UCF24:
```Shell
AP: 80.48% (1)
AP: 96.92% (10)
AP: 79.54% (11)
AP: 59.73% (12)
AP: 75.80% (13)
AP: 91.20% (14)
AP: 87.41% (15)
AP: 70.85% (16)
AP: 71.06% (17)
AP: 90.65% (18)
AP: 94.51% (19)
AP: 63.00% (2)
AP: 90.06% (20)
AP: 77.15% (21)
AP: 80.50% (22)
AP: 75.89% (23)
AP: 89.57% (24)
AP: 84.65% (3)
AP: 76.58% (4)
AP: 67.36% (5)
AP: 95.81% (6)
AP: 93.74% (7)
AP: 93.07% (8)
AP: 95.12% (9)
mAP: 82.53%
```

## AVA v2.2
|    Model    |    Clip    |    mAP    |   FPS   |    weight    |
|-------------|------------|-----------|---------|--------------|
|    YOWO     |     16     |   17.9    |    33   |       -      |
|    YOWO     |     32     |   19.1    |         |       -      |
| YOWO (Ours) |     16     |   20.6    |    33   |  [github](https://github.com/yjh0410/PyTorch_YOWO/releases/download/yowo-weight/yowo_ava_v2.2_20.6.pth)  |
