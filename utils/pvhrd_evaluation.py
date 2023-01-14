import math
import torch
import os
from models.model import PhysNetUpsample, TDMNet, N3DED128, N3DED64, N3DED32, N3DED16, N3DED8, ViT_ST_ST_Compact3_TDC_gra_sharp, PhysNet_padding_ED_peak
from ppg_process_common_function import evaluation, mae, sd, rmse, pearson
from util import load_model


if __name__ == '__main__':
    fps = 30
    # evalution
    model_path = '/media/pxierra/4ddb33c4-42d9-4544-b7b4-796994f061ce/xiongzhuang/1-PycharmProjects/rppg_tdm_talos/saved/models/RPPG_TDMNet_UBFC_MSELoss/0929_155543/model_best.pth'
    val_pth_dir = "/media/pxierra/e70ff8ce-d5d4-4f52-aa2b-921ff250e5fc/P-VHRD-PTH/val"
    # load model
    model = PhysNetUpsample()
    model = load_model(model, model_path)
    # load data
    data_list = os.listdir(val_pth_dir)
    hr_predict_dict = {'v1': [], 'v2': [], 'v3': [], 'v4': [], 'v5': [], 'v6': [], 'v7': []}
    hr_gt_dict = {'v1': [], 'v2': [], 'v3': [], 'v4': [], 'v5': [], 'v6': [], 'v7': []}
    wave_pearson = {'v1': [], 'v2': [], 'v3': [], 'v4': [], 'v5': [], 'v6': [], 'v7': []}
    for data_path in data_list:
        path = os.path.join(val_pth_dir, data_path)
        scence = data_path.split('_')[0][-2:]
        hr_predict, hr_gt, wave_predict, wave_gt = evaluation(model, path, length=240, visualize=False)
        pearson_value = pearson(wave_predict, wave_gt)
        print("scence: ", scence, "hr predict: ", hr_predict, "hr gt: ", hr_gt)
        hr_predict_dict[f'{scence}'].append(hr_predict)
        hr_gt_dict[f'{scence}'].append(hr_gt)
        wave_pearson[f'{scence}'].append(pearson_value)
    for vx in ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7']:
        sd_result = sd(hr_predict_dict[f'{vx}'])
        rmse_result = rmse(hr_predict_dict[f'{vx}'], hr_gt_dict[f'{vx}'])
        mae_result = mae(hr_predict_dict[f'{vx}'], hr_gt_dict[f'{vx}'])
        pearson_result = sum(wave_pearson[f'{vx}']) / len(wave_pearson[f'{vx}'])
        print(f"{vx} ", "sd: ", sd_result, "rmse: ", rmse_result, "mae: ", mae_result, "pearson: ", pearson_result)
