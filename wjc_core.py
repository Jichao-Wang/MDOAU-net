import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import PIL.Image as Image
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.transforms import transforms
from dataset import Train_Dataset, Validation_Dataset, Test_Dataset
import skimage.io as io
import shutil

threshold = 0.5  # 二分类阈值
# 是否使用cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# mask只需要转换为tensor
y_transforms = transforms.ToTensor()


def makedir(new_path):
    folder = os.path.exists(new_path)
    if not folder:
        os.makedirs(new_path)
    else:
        shutil.rmtree(new_path)
        os.makedirs(new_path)


def init_work_space(args):
    makedir('./' + args.model_name + '/results')
    makedir(args.ckpt)
    makedir('./' + args.model_name + '/runs')


def train_model(args, writer, model, criterion, optimizer, dataload, regular=''):
    save_epoch, best_val_acc = 0, -0.1
    for epoch in range(args.epoch):
        print('Epoch {}/{}'.format(epoch, args.epoch - 1))
        print('-' * 10)
        dt_size = len(dataload.dataset)
        epoch_loss = 0
        epoch_correct_pixels, epoch_total_pixels = [], []
        step = 0
        for x, y in dataload:
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = model(inputs).to(device)
            del inputs
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # calculate accuracy
            predicted = outputs.detach().numpy()
            predicted[predicted >= threshold] = 1
            predicted[predicted < threshold] = 0
            correct = (predicted == labels.detach().numpy()).sum()
            del predicted
            pixel_num = 1.0
            for i in range(len(labels.size())):
                pixel_num *= labels.size()[i]

            epoch_correct_pixels.append(correct)
            epoch_total_pixels.append(pixel_num)
            epoch_loss += float(loss.item())
            del labels
            del loss
        val_accuracy = validation(args, model, method='train')
        epoch_loss = epoch_loss / step
        epoch_train_accuracy = np.mean(epoch_correct_pixels) / np.mean(epoch_total_pixels)
        print("epoch %d loss:%0.3f train accuracy:%0.3f val accuracy:%0.3f" % (
            epoch, epoch_loss, epoch_train_accuracy, val_accuracy))
        writer.add_scalar('loss', epoch_loss / step, global_step=epoch)
        writer.add_scalar('train accuracy', epoch_train_accuracy, global_step=epoch)
        writer.add_scalar('validated accuracy', val_accuracy, global_step=epoch)
        writer.add_scalars('accuracy/group',
                           {'train_accuracy': epoch_train_accuracy, 'validated accuracy': val_accuracy},
                           global_step=epoch)
        if best_val_acc < val_accuracy:
            save_epoch = epoch
            torch.save(model, args.ckpt + '/' + args.model_name + '.pth')
            best_val_acc = val_accuracy
    print("Model:", args.model_name)
    print("Dataset:", args.data_file)
    print("Best epoch is" + str(save_epoch))
    print("Best val acc is " + str(best_val_acc))
    return model


# 训练模型
def train(args, writer, model, regular=''):
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), )
    liver_dataset = Train_Dataset(args.data_file, transform=x_transforms, target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    train_model(args, writer, model, criterion, optimizer, dataloaders, regular)


# 用于测试模型在有image有label的数据中的表现
def validation(args, model, print_each=False, method='train'):
    liver_dataset = Validation_Dataset(args.data_file, transform=x_transforms, target_transform=y_transforms)  #
    dataloaders = DataLoader(liver_dataset, batch_size=1)
    if method == 'train':
        dataloaders = DataLoader(liver_dataset, batch_size=8)
    model.eval()
    epoch_correct_pixels, epoch_total_pixels = [], []
    with torch.no_grad():
        for x, y, x_path in dataloaders:
            inputs = x.to(device)
            labels = y.to(device)
            predicted = model(inputs).detach().numpy()
            predicted[predicted >= threshold] = 1
            predicted[predicted < threshold] = 0
            correct = (predicted == labels.detach().numpy()).sum()
            del predicted
            pixel_num = 1.0
            for i in range(len(labels.size())):
                pixel_num *= labels.size()[i]
            epoch_correct_pixels.append(correct)
            epoch_total_pixels.append(pixel_num)
            if print_each:
                print(x_path, 'acc', correct / pixel_num)
    return np.mean(epoch_correct_pixels) / np.mean(epoch_total_pixels)


# 用于测试只有image但没有label的数据
def test(args, save_gray=False, manual=False, weight_path=''):
    model = None
    if not manual:
        model = torch.load(args.ckpt + '/' + args.model_name + '.pth', map_location='cpu')
    if manual:
        model = torch.load(weight_path, map_location='cpu')  # use certain model weight.

    liver_dataset = Test_Dataset(args.data_file, transform=x_transforms, target_transform=y_transforms)

    dataloaders = DataLoader(liver_dataset, batch_size=1)

    model.eval()
    with torch.no_grad():
        for x, pic_name_i in dataloaders:
            pic_name_i = pic_name_i[0]
            io.imsave(args.model_name + "/results/" + pic_name_i.split('.')[0] + "_x.png", torch.squeeze(x).numpy())
            predict = model(x)
            predict = torch.squeeze(predict).detach().numpy()
            if save_gray:
                io.imsave(args.model_name + "/results/" + pic_name_i.split('.')[0] + "_gray_pre.png", predict)

            predict[predict >= threshold] = 1
            predict[predict < threshold] = 0
            io.imsave(args.model_name + "/results/" + pic_name_i.split('.')[0] + "_label_pre.png", predict)


class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

    def clear(self):
        self.outputs = []


def model_forward_visualization(image_path, weight_path, model_name=''):
    """输入一张测试图像和训练好的模型权重，可视化每一步卷积的结果"""
    model = torch.load(weight_path, map_location='cpu')  # load trained model

    save_output = SaveOutput()  # register hooks for each layer
    hook_handles, k1, k2 = [], 0, 0
    for layer in model.modules():
        k1 += 1
        if isinstance(layer, torch.nn.modules.conv.Conv2d):
            k2 += 1
            handle = layer.register_forward_hook(save_output)
            hook_handles.append(handle)

    x = x_transforms(Image.open(image_path).convert('L').resize(size=(512, 512))).unsqueeze(0)
    print(x, x.dtype)
    y = model(x)

    def module_output_to_numpy(tensor):
        return tensor.detach().to('cpu').numpy()

    for layer_idx in range(len(save_output.outputs)):
        images = module_output_to_numpy(save_output.outputs[layer_idx])
        # 这里的0代表读取output里第一个卷积层的输出

        print(type(images))
        print(images.shape)
        mid_1 = images.shape[1]
        mid_idx = 0
        while mid_idx < mid_1:
            # mid_idx is the index of feature
            with plt.style.context("seaborn-white"):
                plt.figure(frameon=False)
            for idx in range(64):
                # idx is the index of subplot
                if mid_idx == mid_1:
                    break
                plt.subplot(8, 8, idx + 1)
                plt.imshow(images[0, mid_idx])
                mid_idx += 1
            plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
            plt.savefig(
                './model_visualization/' + model_name + '/layer_' + str(layer_idx) + '_mid_' + str(mid_idx) + '.png')
            plt.cla()
            plt.close('all')


def model_print(model):
    print(sum(p.numel() for p in model.parameters()))
