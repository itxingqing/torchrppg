import torch
from models.model import Model
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data_path = '/media/pxierra/e70ff8ce-d5d4-4f52-aa2b-921ff250e5fc/P-VHRD-PTH/train/p2v5_0.pth'
    model_path = '/media/pxierra/4ddb33c4-42d9-4544-b7b4-796994f061ce/xiongzhuang/1-PycharmProjects/rppg_tdm_talos/saved/models/RPPG_TDM_TALOS/0824_125315_TALOS_UBFC/checkpoint-epoch59.pth'
    # load model
    model = Model()
    model = model.to('cuda:0')
    model = torch.nn.DataParallel(model, device_ids=[0, 1])
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    # load data
    data = torch.load(data_path)
    input = data['face']
    input = torch.unsqueeze(input, dim=0)
    gt, subject = data['wave']
    gt = torch.unsqueeze(gt, dim=0)
    # inference
    ouput = model(input)
    print(ouput.size())
    fig = plt.figure(1)
    plt.plot(ouput[0, ].cpu().detach().numpy(), '-')
    plt.plot(gt[0, ].cpu().detach().numpy(), '--')
    plt.show()
    # plt.pause(2)
    # plt.close(fig)

