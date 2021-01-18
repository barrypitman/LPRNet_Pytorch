import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.onnx

from data.load_data import CHARS
from model.LPRNet import LPRNet
# from LPRNet_Test import decode


def export_to_onnx():


    lprnet = LPRNet(lpr_max_len=10, phase='train', class_num=len(CHARS), dropout_rate=0.5)


    lprnet.load_state_dict(
        torch.load('/content/gdrive/Shareddrives/ANPR_datasets/ocr/lprnet/pytorch/94x24/LPRNet__iteration_20000.pth',
                   map_location=lambda storage, loc: storage))

    dummy_input = torch.randn(1, 3, 24, 94)
    torch.onnx.export(lprnet, (dummy_input, ),
                      'lprnet.onnx',
                      verbose=True,
                      input_names=['input'],
                      output_names=['output'])

    # Load the ONNX model
    model = onnx.load('lprnet.onnx')

    # Check that the IR is well formed
    onnx.checker.check_model(model)

    # Print a human readable representation of the graph
    print('-------------------')
    print(onnx.helper.printable_graph(model.graph))

    # test
    image = cv2.imread('/content/gdrive/Shareddrives/ANPR_datasets/ocr/lprnet/validation/CA137-993_32564.png')
    im = cv2.resize(image, (94, 24), interpolation=cv2.INTER_CUBIC)
    im = (np.transpose(np.float32(im), (2, 0, 1)) - 127.5) * 0.0078125
    im = np.expand_dims(im, axis=0)

    sess_config = ort.SessionOptions()
    sess_config.log_severity_level = 0
    ort_session = ort.InferenceSession('lprnet.onnx', sess_options=sess_config)
    outputs = ort_session.run(None, {'input': im})

    preds = outputs[0]
    labels, pred_labels = decode(preds, CHARS)
    print(labels)
    print('done')


def convert_to_rknn():
    from rknn.api import RKNN
    # Create RKNN object
    rknn = RKNN(verbose=True)

    # pre-process config
    print('--> config model')
    rknn.config(channel_mean_value='127.5 127.5 127.5 128',
                reorder_channel='0 1 2')
    print('done')

    # Load onnx model
    print('--> Loading model')
    ret = rknn.load_onnx(model='lprnet.onnx')
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=False, pre_compile=True, dataset='./data/dataset.txt')
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export RKNN model')
    ret = rknn.export_rknn('./lprnet.rknn')
    if ret != 0:
        print('Export model failed!')
        exit(ret)
    print('done')



def test_rknn():
    from rknn.api import RKNN
    rknn = RKNN()

    # Load rknn model
    print('--> Load RKNN model')
    ret = rknn.load_rknn('lprnet.rknn')
    if ret != 0:
        print('Export model failed!')
        exit(ret)

    # init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime(target='rk1808')
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')

    # Inference
    print('--> Running model')
    image = cv2.imread('data/eval/000256.png')
    outputs = rknn.inference(inputs=[image])
    preds = outputs[0]
    labels, pred_labels = decode(preds, CHARS)
    print(labels)
    print('done')

    rknn.release()

def to_torchscript():

    lprnet = LPRNet(lpr_max_len=10, phase='test', class_num=len(CHARS), dropout_rate=0.5)

    lprnet.load_state_dict(
        torch.load('/content/gdrive/Shareddrives/ANPR_datasets/ocr/lprnet/pytorch/94x24/LPRNet__iteration_20000.pth',
                   map_location=lambda storage, loc: storage))

    dummy_input = torch.randn(1, 3, 24, 94)



    lprnet.eval()
    # see https://pytorch.org/mobile/home/
    quantized_model=torch.quantization.convert(lprnet)
    script_model = torch.jit.trace(quantized_model,dummy_input)
    opt_model = torch.mobile_optimizer.optimize_for_mobile(script_model)
    torch.jit.save(opt_model, 'lrnet_opt.pt')
    script_model.save("lrnet.pt")

if __name__ == '__main__':
    # export_to_onnx()
    #convert_to_rknn()
    #test_rknn()
    to_torchscript()