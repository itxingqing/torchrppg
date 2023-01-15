import os
from models.model import PhysNetUpsample, TDMNet, N3DED128, N3DED64, N3DED32, N3DED16, N3DED8, ViT_ST_ST_Compact3_TDC_gra_sharp, PhysNet_padding_ED_peak, EfficientPhys_Conv
from ppg_process_common_function import evaluation, mae, sd, rmse, pearson
from util import load_model

if __name__ == '__main__':
    fps = 30
    diff_flag = False
    model_path = '/tmp/pycharm_project_973/saved/models/RPPG_EfficientPhys_Conv_UBFC_Neg_Pearson/0115_162714/model_best.pth'
    val_pth_dir = "/media/xiongzhuang/UBFC_PTH_36x36/val"
    # evalution
    print("Start eval ... ")
    # load model
    # model = ViT_ST_ST_Compact3_TDC_gra_sharp(patches=4, image_h=128, image_w=128, frame=240, dim=96, ff_dim=144,
    #                                          num_heads=4, num_layers=12, dropout_rate=0.1, theta=0.7)
    model = PhysNetUpsample()
    if model._get_name() == 'EfficientPhys_Conv':
        diff_flag = True
    model = load_model(model, model_path)
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
        hr_predict, hr_gt, wave_predict, wave_gt = evaluation(model, path, length=240, visualize=False, diff_flag=diff_flag, method='dft')
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
