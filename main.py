import time
import torch
import wjc_core
import argparse
from tensorboardX import SummaryWriter
from attention_unet import AttU_Net
from segnet import SegNet
from unet import Unet
from Unet_plus_plus import NestedUNet
from Res_net import ResNet50, ResNet101
from MDOAU_net import my_model4  # MDOAU_net

if __name__ == '__main__':
    model, name = my_model4(1, 1), 'data_MDOAU_net_250epoch'
    parse = argparse.ArgumentParser()
    parse.add_argument("--model_name", type=str, default=name)
    parse.add_argument("--batch_size", type=int, default=1)
    parse.add_argument("--epoch", type=int, default=250)
    parse.add_argument("--data_file", type=str, default="data")
    parse.add_argument("--ckpt", type=str, help="the path of model weight file", default="./" + name + "/weights")
    args = parse.parse_args()

    # Prepare a space for saving trained model and predicted results.
    wjc_core.init_work_space(args)

    # Train a model.
    start_time = time.time()
    writer = SummaryWriter('./' + args.model_name + '/runs')
    wjc_core.train(args, writer, model)
    writer.close()
    end_time = time.time()
    print("Training cost ", end_time - start_time, " seconds")

    # Test a model.
    start_time = time.time()
    # test the model trained
    wjc_core.test(args)
    # or test a certain model
    # wjc_core.test(args, save_gray=True, manual=True, weight_path='./weights/MDOAU_net.pth')
    end_time = time.time()
    print("Testing cost ", end_time - start_time, " seconds")

    # Print the validation accuracy of the MODAU-net model. *You can change the pth file.
    print(wjc_core.validation(args, torch.load('./weights/MDOAU_net.pth', map_location='cpu')))

    # Visualize feature maps with an input image and a certain trained model.
    wjc_core.model_forward_visualization(image_path="./data/train/image/8.png",
                                         weight_path="./weights/Attention_Unet.pth")

    # Print parameter number of each models.
    wjc_core.model_print(model)
