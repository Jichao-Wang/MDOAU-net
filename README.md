# MDOAU-net

# Code files
**MDOAU-net** is defined as **my_model4** in my_model.py.  
The multi-scale feature-fusion block, dilated convolution block, offset convolution block, and attention block is also defined in my_model.py.  
Code of baseline models is in unet.py, attention_unet.py, Unet_plus_plus.py, segnet.py, Deeplab_v3_plus.py and Res_net.py.  
Lee.m is an implementation of the Lee filter.  
wjc_dore.py defines the training, validating, and testing logic.  
dataprocess.py creates a data loader in Pytorch.

# How to run
Running main.py can train and test a model.
You can just train a model or test a model by denoting some codes.

# Pre-trained models
They are available at  https://drive.google.com/drive/folders/1dwjUmNgzm5O2jd99jqg0skJDWx946yiZ?usp=sharing.

# Citing MDOAU-net
```
@article{wang2022mdoau,
  title={MDOAU-net: A Lightweight and Robust Deep Learning Model for SAR Image Segmentation in Aquaculture Raft Monitoring},
  author={Wang, Jichao and Fan, Jianchao and Wang, Jun},
  journal={IEEE Geoscience and Remote Sensing Letters},
  year={2022},
  publisher={IEEE}
}
```
