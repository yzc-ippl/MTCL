Multi-Level Transitional Contrast Learning for Personalized Image Aesthetics Assessment (TMM2023)
===
__*Zhichao Yang, Leida Li, Yuzhe Yang, Yaqian Li, and Weisi Lin*__  
### PyTorch implementation for the [paper](https://ieeexplore.ieee.org/abstract/document/10168279)  

### Model weight：[model](https://pan.baidu.com/s/1wsb249NwjgaoPCBlHNRM1Q?pwd=0981)

## Inference Guide：

### 1. Overview
This guide will help you get started with the MTCL inference code.

### 2. Model Architecture

MTCL consists of three main components:
```
**GIAA Model**: General Image Aesthetic Assessment backbone (ResNet-50 based)
**Contrast Model**: Contrastive learning encoder for personalized features
**PIAA Model**: Fusion of GIAA and Contrast features with personalized regression head
```

### 3. Directory Structure
```
project_root/
├── code/
│   ├── GIAA/
│   │   └── train_GIAA_model.py       # GIAA model definition
│   ├── MTCL/
│   │   └── train_Contrast_model.py   # Contrast model definition
│   └── PIAA/
│       └── ├── FlickerAes_PIAA/
            │       └── image/        # Flickr-AES images
            │       └── label/
            │           ├── test_worker.csv                   # Test Worker information
            │           └── image_labeled_by_each_worker.csv  # Image ratings by workers

            └── test_PIAA.py          # This inference script
```

### 4. Download Required Files
Pre-trained PIAA Model: Place at 
```
./model/ResNet50/ResNet50-FlickrAes-PIAA.pt
./model/ResNext101/ResNext101-FlickrAes-PIAA.pt
```
Flickr-AES Dataset:  
```
Images: ./FlickerAes_PIAA/image/
Labels: ./FlickerAes_PIAA/label/
```

### 5. Running Inference
```
python test_PIAA.py
```

If you find our work is useful, pleaes cite the paper: 
```
@ARTICLE{10168279,  
  author={Yang, Zhichao and Li, Leida and Yang, Yuzhe and Li, Yaqian and Lin, Weisi},  
  journal={IEEE Transactions on Multimedia},   
  title={Multi-Level Transitional Contrast Learning for Personalized Image Aesthetics Assessment},   
  year={2024},  
  volume={26},  
  number={},  
  pages={1944-1956},  
  keywords={Task analysis;Feature extraction;Databases;Standards;Electronic mail;Videos;Training;Personalized image aesthetics assessment;contrast learning;aesthetic preferences},  
}
```
