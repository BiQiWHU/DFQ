# DFQ: Learning Generalized Medical Image Segmentation from Decoupled Feature Queries
This is the official implementation of our work entitled ```DFQ: Learning Generalized Medical Image Segmentation from Decoupled Feature Queries```, which has been accepted by ```AAAI2024```.

![avatar](/DFQframework.png)

An example of training and inference is given below.

## Environment Configuration
The basic enviroment dependencies include:
```
    pip install torchvision==0.8.2
    pip install timm==0.3.2
    pip install mmcv-full==1.2.7
    pip install opencv-python==4.5.1.48
```
For other minor packages, please refer to the ```requirements.txt``` file in this project.

## Training on Source Domain
An example of training on ```DD Fundus benchmark``` with ```domain-0``` as ```unseen target domain``` is given below.

```
python -W ignore train_feed.py --data_root D:/Med/dataset --dataset fundus --domain_idxs 1,2,3 --test_domain_idx 0 --is_out_domain --consistency --consistency_type kd --encoder b3 --save_path outdir/fundus/target0_pretrain_0.99_b3_feed_iw
```

## Inference on Unseen Target Domains

An example of inference on a pre-trained model is given below.
```
python -W ignore test_fundus_slice_feed.py --model_file outdir/fundus/target0_pretrain_0.99_b3_feed_iw/model_xx.xx.pth --dataset fundus --data_dir D:/Med/dataset --datasetTest 0 --encoder b3 --test_prediction_save_path results/fundus/target0_pretrain_0.99_b3_feed_iw_xx.xx --save_result
```
By using this CMD, not only the numerical results but also the visual prediction can be outputted.
Here ```model_xx.xx.pth``` refers to the name of a pre-trained model, where ```x``` refers to a number value.

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

(1) The code of dataloader is based on ```RAM-DSIR``` published in ```ECCV2022```, with the code link [https://github.com/zzzqzhou/RAM-DSIR].

(2) The code of ```feature as query``` is highly based on ```FeedFormer``` published in ```AAAI2023```, with the code link [https://github.com/jhshim1995/FeedFormer].
