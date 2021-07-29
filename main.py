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
from my_model import my_model4 # my_model4 is the MDOAU-net


if __name__ == '__main__':
    model, name = Unet(1, 1), 'Unet_500epoch'
    parse = argparse.ArgumentParser()
    parse.add_argument("--model_name", type=str, default=name)
    parse.add_argument("--batch_size", type=int, default=8)
    parse.add_argument("--epoch", type=int, default=200)
    parse.add_argument("--data_file", type=str, default="data")
    parse.add_argument("--ckpt", type=str, help="the path of model weight file", default="./" + name + "/weights")
    args = parse.parse_args()
    wjc_core.init_work_space(args)

	start_time = time.time()
    writer = SummaryWriter('./' + args.model_name + '/runs')

    wjc_core.train(args, writer, model)
    writer.close()
    end_time = time.time()
    print("Training cost ", end_time - start_time, " seconds")

    wjc_core.test(args)
    start_time = time.time()
	print("Testing cost ", start_time - end_time, " seconds")
    writer.close()
