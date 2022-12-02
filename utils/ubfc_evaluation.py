import math
import torch
import os
from models.model import PhysNetUpsample, TDMNet, N3DED128, N3DED64, N3DED32, N3DED16, N3DED8, ViT_ST_ST_Compact3_TDC_gra_sharp, PhysNet_padding_ED_peak
from ppg_process_common_function import evaluation, mae, sd, rmse, pearson

if __name__ == '__main__':
    fps = 30
    model_path = '/media/pxierra/4ddb33c4-42d9-4544-b7b4-796994f061ce/xiongzhuang/1-PycharmProjects/rppg_tdm_talos/saved/models/RPPG_Physformer_UBFC_MSELoss/1201_175701/model_best.pth'
    val_pth_dir = "/media/pxierra/4ddb33c4-42d9-4544-b7b4-796994f061ce/data/pluse/UBFC/TDM_rppg_input/DATASET_2_PTH/val"
    # evalution
    print("Start eval ... ")
    # load model
    model = TDMNet()
    model = model.to('cuda:0')
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    # load data
    data_list = os.listdir(val_pth_dir)
    hr_predict_list = []
    hr_gt_list = []
    data_list.sort()
    for data_path in data_list:
        path = os.path.join(val_pth_dir, data_path)
        if data_path.split('_')[0] != 'subject11':
            hr_predict, hr_gt = evaluation(model, path, fps=fps, visualize=False)
            print("data_path: ", data_path, "hr predict: ", hr_predict, "hr gt: ", hr_gt)
            hr_predict_list.append(hr_predict)
            hr_gt_list.append(hr_gt)
    sd = sd(hr_predict_list)
    rmse_result = rmse(hr_predict_list, hr_gt_list)
    mae_result = mae(hr_predict_list, hr_gt_list)
    pearson_result = pearson(hr_predict_list, hr_gt_list)
    print("sd: ", sd, "rmse: ", rmse_result, "mae: ", mae_result, "pearson: ", pearson_result)
    print("Finsh eval! ")
