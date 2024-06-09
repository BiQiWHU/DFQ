#!/usr/bin/env python
import argparse
import os
import os.path as osp
import torch.nn as nn
import torch
from utils.metrics import *
from dataset import utils
from utils.utils import save_per_img_prostate, _connectivity_region_analysis
from test_utils import *
# from networks.segformer import Encoder, Decoder
from networks.feedformer_mmseg import Encoder_b2, Decoder_b2, Encoder_b3, Decoder_b3, Encoder_b4, Decoder_b4, Encoder_b5, Decoder_b5
from networks.feedformer_mmseg import FeedFormerHead
# from networks.segformer_mmseg import RecDecoder_b2, RecDecoder_b3, RecDecoder_b4, RecDecoder_b5
from tqdm import tqdm
import numpy as np
from medpy.metric import binary
from torch.nn import DataParallel
import SimpleITK as sitk
import warnings
from thop import profile
warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser(description='Test on Prostate dataset (3D volume)')
    # basic settings
    parser.add_argument('--model_file', type=str, default=None, required=True, help='Model path')
    parser.add_argument('--dataset', type=str, default='prostate', help='training dataset')
    parser.add_argument('--data_dir', default='../dataset', help='data root path')
    parser.add_argument('--datasetTest', type=int, default=3, help='test folder id contain images ROIs to test')
    parser.add_argument('--in_channels', type=int, default=3, help='number of input channels')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size of testing')
    parser.add_argument('--num_classes', type=int, default=2, help='number of classes')
    parser.add_argument('--test_prediction_save_path', type=str, default=None, required=True, help='Path root for test image and mask')
    parser.add_argument('--save_result', action='store_true', help='Save Results')
    parser.add_argument('--freeze_bn', action='store_true', help='Freeze Batch Normalization')
    parser.add_argument('--norm', type=str, default='bn', help='normalization type')
    parser.add_argument('--activation', type=str, default='relu', help='feature activation function')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
    parser.add_argument('--encoder', type=str, default='b3')

    args = parser.parse_args()
    return args

def main(args):
    domain_name = domain_list[args.datasetTest]
    data_dir = os.path.join(args.data_dir, args.dataset)

    file_list = [item for item in os.listdir(os.path.join(data_dir, domain_name)) if 'segmentation' not in item]
    
    if not os.path.exists(args.test_prediction_save_path):
        os.makedirs(args.test_prediction_save_path)
    
    model_file = args.model_file
    output_path = os.path.join(args.test_prediction_save_path, 'test' + str(args.datasetTest))
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    if args.encoder == 'b3':
        encoder = Encoder_b3()
        # seg_decoder = Decoder_b3(num_classes=args.num_classes)
        seg_decoder = FeedFormerHead(in_channels=[64, 128, 320, 512], feature_strides=[4, 8, 16, 32], num_classes=args.num_classes)  
    elif args.encoder == 'b4':
        encoder = Encoder_b4()
        seg_decoder = Decoder_b4(num_classes=args.num_classes)
    elif args.encoder == 'b5':
        encoder = Encoder_b5()
        seg_decoder = Decoder_b5(num_classes=args.num_classes)
    
    state_dicts = torch.load(model_file)
    
    encoder.load_state_dict(state_dicts['encoder_state_dict'])
    seg_decoder.load_state_dict(state_dicts['seg_decoder_state_dict'])
    
    encoder = DataParallel(encoder).cuda()
    seg_decoder = DataParallel(seg_decoder).cuda()
    # encoder = encoder.cuda()
    # seg_decoder = seg_decoder.cuda()


    if not args.freeze_bn:
        encoder.eval()
        for m in encoder.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.train()
        seg_decoder.eval()
        for m in seg_decoder.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.train()
    else:
        encoder.eval()
        seg_decoder.eval()

    tbar = tqdm(file_list, ncols=150)

    val_dice = 0.0
    total_hd = 0.0
    total_asd = 0.0
    total_num = 0
    with torch.no_grad():
        for iindex, file_name in enumerate(tbar):
            itk_image = sitk.ReadImage(os.path.join(data_dir, domain_name, file_name))
            itk_mask = sitk.ReadImage(os.path.join(data_dir, domain_name, file_name.replace('.nii.gz', '_segmentation.nii.gz')))

            image = sitk.GetArrayFromImage(itk_image)
            mask = sitk.GetArrayFromImage(itk_mask)

            max_value = np.max(image)
            min_value = np.min(image)
            image = 2 * (image - min_value) / (max_value - min_value) - 1

            mask[mask==2] = 1
            pred_y = np.zeros(mask.shape)

            #### channel 3 ####
            frame_list = [kk for kk in range(1, image.shape[0] - 1)]

            for ii in range(int(np.floor(image.shape[0] // args.batch_size))):
                vol = np.zeros([args.batch_size, 3, image.shape[1], image.shape[2]])

                for idx, jj in enumerate(frame_list[ii * args.batch_size : (ii + 1) * args.batch_size]):
                    vol[idx, ...] = image[jj - 1 : jj + 2, ...].copy()
                vol = torch.from_numpy(vol).float().cuda()

                # pred_student = torch.max(torch.softmax(seg_decoder(encoder(vol)), dim=1), dim=1)[1].detach().data.cpu().numpy()
                ### iw
                # flops1, params1 = profile(encoder, inputs=(vol,), verbose=False)
                x, _ = encoder(vol)
                # flops2, params2 = profile(seg_decoder, inputs=(x,), verbose=False)
                # print('Encoder para {}, gflop {}'.format(params1, flops1))
                # print('Decoder para {}, gflop {}'.format(params2, flops2))
                # exit()
                pred = torch.softmax(seg_decoder(x), dim=1)
                # v, x_1, x_2, x_3, x_4 = seg_decoder(x)
                # pred = torch.softmax(v, dim=1)
                # pred = torch.softmax(seg_decoder(encoder(vol)), dim=1)
                pred = torch.nn.functional.interpolate(pred, size=(mask.shape[1], mask.shape[2]), mode="bilinear")
                pred_student = torch.max(pred, dim=1)[1].detach().data.cpu().numpy()

                for idx, jj in enumerate(frame_list[ii * args.batch_size : (ii + 1) * args.batch_size]):
                    ###### Ignore slices without prostate region ######
                    if np.sum(mask[jj, ...]) == 0:
                        continue
                    pred_y[jj, ...] = pred_student[idx, ...].copy()

            processed_pred_y = _connectivity_region_analysis(pred_y)
            # if iindex == 0:
            #     preds = pred.detach().cpu().numpy()
            #     x_s1 = x_1.detach().cpu().numpy()
            #     x_s2 = x_2.detach().cpu().numpy()
            #     x_s3 = x_3.detach().cpu().numpy()
            #     x_s4 = x_4.detach().cpu().numpy()
            # else:
            #     preds = np.concatenate((preds, pred.detach().cpu().numpy()), 0)
            #     x_s1 = np.concatenate((x_s1, x_1.detach().cpu().numpy()), axis=0)
            #     x_s2 = np.concatenate((x_s2, x_2.detach().cpu().numpy()), axis=0)
            #     x_s3 = np.concatenate((x_s3, x_3.detach().cpu().numpy()), axis=0)
            #     x_s4 = np.concatenate((x_s4, x_4.detach().cpu().numpy()), axis=0)
            if iindex == 0:
                x_s1 = x[0].cpu().numpy()
                x_s2 = x[1].cpu().numpy()
                x_s3 = x[2].cpu().numpy()
                x_s4 = x[3].cpu().numpy()
                preds = pred.cpu().numpy()
            else:
                x_s1 = np.concatenate((x_s1, x[0].detach().cpu().numpy()), axis=0)
                x_s2 = np.concatenate((x_s2, x[1].detach().cpu().numpy()), axis=0)
                x_s3 = np.concatenate((x_s3, x[2].detach().cpu().numpy()), axis=0)
                x_s4 = np.concatenate((x_s4, x[3].detach().cpu().numpy()), axis=0)
                preds = np.concatenate((preds, pred.detach().cpu().numpy()), axis=0)


            dice_coeff = binary.dc(np.asarray(processed_pred_y, dtype=np.bool_),
                                np.asarray(mask, dtype=np.bool_))
            hd = binary.hd95(np.asarray(processed_pred_y, dtype=np.bool_),
                                np.asarray(mask, dtype=np.bool_))
            asd = binary.asd(np.asarray(processed_pred_y, dtype=np.bool_),
                                np.asarray(mask, dtype=np.bool_))
            count = 0

            if args.save_result:
                for i in range(image.shape[0]):
                    count += 1
                    for img, lt, lp in zip([image[i]], [mask[i]], [processed_pred_y[i]]):
                        ###### Ignore slices without prostate region ######
                        if np.sum(lt) == 0:
                            continue
                        img, lt = utils.untransform_prostate(img, lt)
                        img = np.repeat(np.expand_dims(img, axis=0), repeats=3, axis=0)
                        save_per_img_prostate(img.transpose(1, 2, 0),
                                            output_path,
                                            file_name.split('.')[0] + '_' + str(count),
                                            lp, lt, mask_path=None, ext="bmp")

            val_dice += dice_coeff
            total_hd += hd
            total_asd += asd
            total_num += 1
        
        val_dice /= total_num
        total_hd /= total_num
        total_asd /= total_num
        
        print('''\n==>val_dice : %.2f''' % (100 * val_dice))
        print('''\n==>average_hd : %.2f''' % (total_hd))
        print('''\n==>average_asd : %.2f''' % (total_asd))
        with open(osp.join(output_path, '../test' + str(args.datasetTest) + '_log.csv'), 'a') as f:
                log = [['batch-size: '] + [args.batch_size] + [args.model_file] + \
                    ['dice coefficence: '] + [val_dice] + \
                    ['average_hd: '] + [total_hd] + \
                    ['average_asd: '] + [total_asd]]
                log = map(str, log)
                f.write(','.join(log) + '\n')
        # np.save('pred_prostate_{}_mid.npy'.format(str(args.datasetTest)), preds)
        # print(x_s1.shape)
        # np.save('D:/Med/tsne/feat_prostate{}_stage1_mid_en.npy'.format(str(args.datasetTest)), x_s1)
        # print(x_s2.shape)
        # np.save('D:/Med/tsne/feat_prostate{}_stage2_mid_en.npy'.format(str(args.datasetTest)), x_s2)
        # print(x_s3.shape)
        # np.save('D:/Med/tsne/feat_prostate{}_stage3_mid_en.npy'.format(str(args.datasetTest)), x_s3)
        # print(x_s4.shape)
        # np.save('D:/Med/tsne/feat_prostate{}_stage4_mid_en.npy'.format(str(args.datasetTest)), x_s4)
        # print(preds.shape)
        # np.save('D:/Med/tsne/feat_prostate{}_preds_mid_en.npy'.format(str(args.datasetTest)), preds)


if __name__ == '__main__':
    args = parse_args()
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    domain_list = ['ISBI', 'ISBI_1.5', 'I2CVB', 'UCL', 'BIDMC', 'HK']
    main(args)