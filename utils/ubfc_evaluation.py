import math
from collections import OrderedDict

import torch
import os
from models.model import PhysNetUpsample, TDMNet, N3DED128, N3DED64, N3DED32, N3DED16, N3DED8, ViT_ST_ST_Compact3_TDC_gra_sharp, PhysNet_padding_ED_peak
from ppg_process_common_function import evaluation, mae, sd, rmse, pearson

if __name__ == '__main__':
    fps = 30
    model_path = '/media/pxierra/4ddb33c4-42d9-4544-b7b4-796994f061ce/xiongzhuang/1-PycharmProjects/torchrppg/saved/models/RPPG_PhysNetUpsample_UBFC_PhysFormerLoss/0105_143856/model_best.pth'
    val_pth_dir = "/media/pxierra/e70ff8ce-d5d4-4f52-aa2b-921ff250e5fc/UBFC_PATH_orignal_wave_30train/val"
    # evalution
    print("Start eval ... ")
    # load model
    # model = ViT_ST_ST_Compact3_TDC_gra_sharp(patches=4, image_h=128, image_w=128, frame=240, dim=96, ff_dim=144,
    #                                          num_heads=4, num_layers=12, dropout_rate=0.1, theta=0.7)
    model = PhysNetUpsample()
    model = model.to('cuda:0')
    checkpoint = torch.load(model_path)
    # model.load_state_dict(checkpoint['state_dict'])
    new_state_dict = OrderedDict()
    for k in checkpoint['state_dict']:
        name = k.replace('module.', '')
        new_state_dict[name] = checkpoint['state_dict'].setdefault(k)
    model.load_state_dict(new_state_dict)
    model.eval()
    # load data
    data_list = os.listdir(val_pth_dir)
    video_level_hr_predict_list = []
    video_level_hr_gt_list = []
    wave_pearson = []
    data_list.sort()
    hr_predict_dict = {'subject38': [], 'subject39': [], 'subject40': [], 'subject41': [], 'subject42': [], 'subject43': [], 'subject44': [],
                       'subject45': [], 'subject46': [], 'subject47': [], 'subject48': [], 'subject49': []}
    hr_gt_dict = {'subject38': [], 'subject39': [], 'subject40': [], 'subject41': [], 'subject42': [], 'subject43': [], 'subject44': [],
                       'subject45': [], 'subject46': [], 'subject47': [], 'subject48': [], 'subject49': []}
    for data_path in data_list:
        path = os.path.join(val_pth_dir, data_path)
        hr_predict, hr_gt, wave_predict, wave_gt = evaluation(model, path, length=240, visualize=False, method='dft')
        wave_pearson.append(pearson(wave_predict, wave_gt))
        print("data_path: ", data_path, "hr predict: ", hr_predict, "hr gt: ", hr_gt)
        hr_predict_dict[data_path.split('_')[0]].append(hr_predict)
        hr_gt_dict[data_path.split('_')[0]].append(hr_gt)
    for k in hr_predict_dict.keys():
        print(f"{k}: hr predict: {sum(hr_predict_dict[k])/len(hr_predict_dict[k])}, hr gt: {sum(hr_gt_dict[k])/len(hr_gt_dict[k])}")
        video_level_hr_predict_list.append(sum(hr_predict_dict[k])/len(hr_predict_dict[k]))
        video_level_hr_gt_list.append(sum(hr_gt_dict[k])/len(hr_gt_dict[k]))
    sd = sd(video_level_hr_predict_list)
    rmse_result = rmse(video_level_hr_predict_list, video_level_hr_gt_list)
    mae_result = mae(video_level_hr_predict_list, video_level_hr_gt_list)
    pearson_result = sum(wave_pearson) / len(wave_pearson)
    print("sd: ", sd, "rmse: ", rmse_result, "mae: ", mae_result, "pearson: ", pearson_result)
    print("Finsh eval! ")
