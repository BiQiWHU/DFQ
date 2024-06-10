# DFQ: Learning Generalized Medical Image Segmentation from Decoupled Feature Queries
This is the official implementation of our work entitled ```DFQ: Learning Generalized Medical Image Segmentation from Decoupled Feature Queries```, which has been accepted by ```AAAI2024```.

An example of training and inference is given below.

## Environment Configuration

```
    pip install torchvision==0.8.2
    pip install timm==0.3.2
    pip install mmcv-full==1.2.7
    pip install opencv-python==4.5.1.48
```

## Training on Source Domain
An example of training on ```CityScapes``` source domain is given below.

```
python train_net.py --num-gpus 2 --config-file configs/cityscapes/semantic-segmentation/swin/maskformer2_swin_base_IN21k_384_bs16_90k.yaml
```

## Inference on Unseen Target Domains

The below lines are the example code to infer on ```GTA``` and ```SYN``` unseen target domains.
```
python train_net.py --config-file configs/cityscapes/semantic-segmentation/swin/maskformer2_swin_base_IN21k_384_bs16_90k.yaml --eval-only MODEL.WEIGHTS E:/DGtask/DGViT/Mask2Former-main/output_gta/model_final.pth
```
```
python train_net.py --config-file configs/cityscapes/semantic-segmentation/swin/maskformer2_swin_base_IN21k_384_bs16_90k.yaml --eval-only MODEL.WEIGHTS E:/DGtask/DGViT/Mask2Former-main/output_syn/model_final.pth
```

# Citation

If you find our work useful, please cite as

```BibTeX
@inproceedings{bi2024learning,
  title={Learning Generalized Medical Image Segmentation from Decoupled Feature Queries},
  author={Qi, Bi and Jingjun, Yi and Hao, Zheng and Wei, Ji and Yawen, Huang and Yuexiang, Li and Yefeng, Zheng},
  journal={AAAI},
  year={2024}
}
```

# Acknowledgement

The development of ```Decoupled Feature Queries``` (DFQ) largely relies on two prior projects:

(1) The code of dataloader is based on ```RAM``` published in ```ECCV2022```, with the code link [https://github.com/zzzqzhou/RAM-DSIR].

(2) The code of ```feature as query``` is highly based on ```FeedFormer``` published in ```AAAI2023```, with the code link [https://github.com/jhshim1995/FeedFormer].
