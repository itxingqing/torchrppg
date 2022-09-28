import math
import torch
import os
from models.model import Model
from ppg_process_common_function import evaluation, mae, mse, rmse


if __name__ == '__main__':
    fps = 30
    val_pth_dir = "/media/pxierra/e70ff8ce-d5d4-4f52-aa2b-921ff250e5fc/P-VHRD-PTH/val"
    # evalution
    model_path = '/media/pxierra/4ddb33c4-42d9-4544-b7b4-796994f061ce/xiongzhuang/1-PycharmProjects/rppg_tdm_talos/saved/models/RPPG_TDM_MSELoss/0919_190645/model_best.pth'
    # load model
    model = Model()
    model = model.to('cuda:0')
    model = torch.nn.DataParallel(model, device_ids=[0, 1])
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    # load data
    data_list = os.listdir(val_pth_dir)
    hr_predict_dict = {'v1': [], 'v2': [], 'v3': [], 'v4': [], 'v5': [], 'v6': [], 'v7': []}
    hr_gt_dict = {'v1': [], 'v2': [], 'v3': [], 'v4': [], 'v5': [], 'v6': [], 'v7': []}
    for data_path in data_list:
        path = os.path.join(val_pth_dir, data_path)
        scence = data_path.split('_')[0][-2:]
        hr_predict, hr_gt = evaluation(model, path, fps=fps, visualize=False)
        print("scence: ", scence, "hr predict: ", hr_predict, "hr gt: ", hr_gt)
        hr_predict_dict[f'{scence}'].append(hr_predict)
        hr_gt_dict[f'{scence}'].append(hr_gt)
    for vx in ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7']:
        mse_result = mse(hr_predict_dict[f'{vx}'], hr_gt_dict[f'{vx}'])
        rmse_result = rmse(hr_predict_dict[f'{vx}'], hr_gt_dict[f'{vx}'])
        mae_result = mae(hr_predict_dict[f'{vx}'], hr_gt_dict[f'{vx}'])
        print(f"{vx} ", "mse: ", mse_result, "rmse: ", rmse_result, "mae: ", mae_result)