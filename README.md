This is an official implementation of CVPR23 paper 'Unbiased Multiple Instance Learning for Weakly Supervised Video Anomaly Detection' (https://arxiv.org/abs/2303.12369v1). 

    
# Environment Setup
To set up the environment, you can easily run the following command:
```
conda create -n UMIL python=3.7
conda activate UMIL
pip install -r requirements.txt
```

Install Apex as follows
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

# Data Preparation

Download the videos and labels for UCF-crime or TAD dataset and extract frames from videos.
The dataset directory should be origanized as follows: 
```
UCF/
├─ frames/    
    ├─ Abuse/
       ├─ Abuse001_x264.mp4/
           ├─ img_00000000.jpg
    ├─ Arrest/ 
    ...
    
TAD/
├─ frames/    
    ├─ abnormal/
        ├─ 01_Accident_001.mp4/
        ...
    ├─ normal/ 
        ...
```
> [**TAD extracted frames**](https://smu-my.sharepoint.com/:f:/r/personal/huilyu_smu_edu_sg/Documents/UMIL/TAD?csf=1&web=1&e=HxzRqC)

# Pre-trained model weights
Please find the model weights in the following:
> [**k400 pre-trained weights**](https://smu-my.sharepoint.com/:u:/g/personal/huilyu_smu_edu_sg/ESDZwxBmIAdLqJBuDwhU-YIB1kn7MNEQ0CEGAkkUSwfPkA?e=7dpMd5)

# Train
The config files lie in `configs`. For example, to train X-CLIP-B/32 with 5 frames on UCF on 2 GPUs, you can run
```
CUDA_VISIBLE_DEVICES=0,1 bash tools/dist_train_recognizer.sh 2
```

**Note:**
- The test during training is a fast test strategy, it does not represent the real AUC.
- Please specify the data path in config file(`configs/*.yaml`). Also, you can set them by attaching an argument `--opts DATA.ROOT /PATH/TO/videos DATA.TRAIN_FILE /PATH/TO/train.txt DATA.VAL_FILE /PATH/TO/val.txt`. Note that if you use the tar file(`videos.tar`), just set the `DATA.ROOT` to `/PATH/TO/videos.tar`. For standard folder, set that to `/PATH/TO/videos` naturally.
- The pretrained model will be automatically downloaded. Of course, you can specify it by using `--pretrained /PATH/TO/PRETRAINED`.

# Test
For example, to test the X-CLIP-B/32 with 5 frames on UCF, you can run
```
CUDA_VISIBLE_DEVICES=1 bash tools/dist_test_recognizer.sh 1
```

If you find this work helpful, please cite:
```
@inproceedings{Lv2023unbiased,
title={Unbiased Multiple Instance Learning for Weakly Supervised Video Anomaly Detection},
author={Hui Lv and Zhongqi Yue and Qianru Sun and Bin Luo and Zhen Cui and Hanwang Zhang},
booktitle={CVPR},
year={2023}
}
```
