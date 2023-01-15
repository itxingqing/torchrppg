import math
import torch
import os
from models.model import PhysNetUpsample, TDMNet, N3DED128, N3DED64, N3DED32, N3DED16, N3DED8, ViT_ST_ST_Compact3_TDC_gra_sharp, PhysNet_padding_ED_peak
from ppg_process_common_function import evaluation, mae, sd, rmse, pearson
from util import load_model

if __name__ == '__main__':
    fps = 30
    diff_flag = False
    model_path = '/media/pxierra/4ddb33c4-42d9-4544-b7b4-796994f061ce/xiongzhuang/1-PycharmProjects/torchrppg/saved/models/RPPG_PhysTransformer_VIPL_PhysFormerLoss/1220_143403/model_best.pth'
    val_pth_dir = "/media/pxierra/e70ff8ce-d5d4-4f52-aa2b-921ff250e5fc/VIPL_PTH/val"
    # evalution
    print("Start eval ... ")
    # load model
    model = PhysNetUpsample()
    if model._get_name() == 'EfficientPhys_Conv':
        diff_flag = True
    model = load_model(model, model_path)
    model.eval()
    # load data
    data_list = os.listdir(val_pth_dir)
    hr_predict_list = []
    hr_gt_list = []
    wave_pearson = []
    data_list.sort()
    for data_path in data_list:
        path = os.path.join(val_pth_dir, data_path)
        hr_predict, hr_gt, wave_predict, wave_gt = evaluation(model, path, length=240, diff_flag=diff_flag, visualize=True)
        wave_pearson.append(pearson(wave_predict, wave_gt))
        print("data_path: ", data_path, "hr predict: ", hr_predict, "hr gt: ", hr_gt)
        hr_predict_list.append(hr_predict)
        hr_gt_list.append(hr_gt)
    sd = sd(hr_predict_list)
    rmse_result = rmse(hr_predict_list, hr_gt_list)
    mae_result = mae(hr_predict_list, hr_gt_list)
    pearson_result = sum(wave_pearson) / len(wave_pearson)
    print("sd: ", sd, "rmse: ", rmse_result, "mae: ", mae_result, "pearson: ", pearson_result)
    print("Finsh eval! ")
