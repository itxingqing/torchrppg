import os
import torch
import onnx
from onnxsim import simplify

from utils.util import load_model
from models.model import  PhysNetUpsample, EfficientPhys_Conv


if __name__ == '__main__':
    torch_model_path = 'model/EfficientPhys_Conv.pth'
    onnx_model_path = 'model/EfficientPhys_Conv.onnx'
    sim_onnx_model_path = 'model/EfficientPhys_Conv.sim.onnx'
    # use forward_deploy instead of forward
    model = EfficientPhys_Conv()
    model = load_model(model, torch_model_path)
    model = model.cpu()

    input_data = (torch.randn(((1, 3, 240, 36, 36))))
    input_names = ['input.1']
    output_names = ['rPPG']
    # export onnx
    # torch.onnx.export(
    #     model,
    #     input_data,
    #     onnx_model_path,
    #     keep_initializers_as_inputs=False,
    #     verbose=False,
    #     input_names=input_names,
    #     output_names=output_names,
    #     dynamic_axes=None,
    #     opset_version=11)
    torch.onnx.export(model, input_data, onnx_model_path, verbose=True, input_names=input_names,
                      output_names=output_names)
    # simplify onnx
    model = onnx.load(onnx_model_path)
    model, check = simplify(model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model, sim_onnx_model_path)

    cmd = './MNNConvert -f ONNX --modelFile model/EfficientPhys_Conv.sim.onnx --MNNModel model/EfficientPhys_Conv.mnn --bizCode biz'
    os.system(cmd)


