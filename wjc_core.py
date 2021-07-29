import torch
import numpy as np
import os
import cv2
from torchstat import stat
import matplotlib.pyplot as plt
import PIL.Image as Image
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.transforms import transforms
from dataset import Train_Dataset, Validation_Dataset, Test_Dataset, All_Test_Dataset
import skimage.io as io
import shutil
from my_loss import L1_norm, L2_norm, DiceLoss, BCEDiceLoss

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
            # print(outputs)
            # print(labels)
            loss = criterion(outputs, labels)
            if regular == 'L1' and epoch >= 100:
                loss = loss + 0.01 * L1_norm(model)
            elif regular == 'L2' and epoch >= 100:
                loss = loss + 0.01 * L2_norm(model)
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

            # print("%d/%d,train loss:%0.3f" % (step, (dt_size - 1) // dataload.batch_size + 1, loss.item()))

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
def train(args, writer, model, loss='BCELoss', regular=''):
    model.to(device)
    criterion = nn.BCELoss()  # nn.BCEWithLogitsLoss()
    if loss == 'DiceLoss':
        criterion = nn.BCELoss()
    elif loss == 'BCEDiceLoss':
        criterion = BCEDiceLoss()
    optimizer = optim.Adam(model.parameters(), )
    liver_dataset = Train_Dataset(args.data_file, transform=x_transforms, target_transform=y_transforms)
    # now, list of torch.Size([1, 512, 512])  [channel, img_x, img_y]
    dataloaders = DataLoader(liver_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    # now, list of torch.Size([2, 1, 512, 512])  [batch_size, channel, img_x, img_y]
    train_model(args, writer, model, criterion, optimizer, dataloaders, regular)


def validation(args, model, print_each=False, method='train'):
    liver_dataset = Validation_Dataset(args.data_file, transform=x_transforms, target_transform=y_transforms)  #
    dataloaders = DataLoader(liver_dataset, batch_size=1)
    if method == 'train':
        dataloaders = DataLoader(liver_dataset, batch_size=8)
    model.eval()
    epoch_correct_pixels, epoch_total_pixels, OTSU_threshold = [], [], []
    with torch.no_grad():
        for x, y, x_path in dataloaders:
            inputs = x.to(device)
            labels = y.to(device)
            predicted = model(inputs).detach().numpy()
            predicted[predicted >= threshold] = 1
            predicted[predicted < threshold] = 0
            correct = (predicted == labels.detach().numpy()).sum()

            # predicted = model(inputs)
            # io.imsave("mid_delete.png", torch.squeeze(predicted).detach().numpy())
            # predicted = cv2.imread("mid_delete.png", cv2.IMREAD_GRAYSCALE)
            # mid_OTSU_threshold, predicted = cv2.threshold(predicted, 0, 255, cv2.THRESH_OTSU)
            # predicted = torch.from_numpy(predicted) / 255.0
            # OTSU_threshold.append(mid_OTSU_threshold)
            # if print_each:
            #     print('OTSU_threshold', mid_OTSU_threshold)
            # correct = (predicted.numpy() == labels.detach().numpy()).sum()

            del predicted
            pixel_num = 1.0
            for i in range(len(labels.size())):
                pixel_num *= labels.size()[i]
            epoch_correct_pixels.append(correct)
            epoch_total_pixels.append(pixel_num)
            if print_each:
                print(x_path, 'acc', correct / pixel_num)
    return np.mean(epoch_correct_pixels) / np.mean(epoch_total_pixels)


def compare(args, model, manual=False, print_each=False, weight_path='', save_each=False, list_return=False):
    # 输出在指定数据集(args指定)中(包括train & val),使用一个模型进行预测的全部分割图.并对其和label的区别用颜色标记.
    # 绿色表示正确分割,红色表示label中是raft但prediction不是(failure prediction),黄色表示label不是raft但prediction是(failure alert).
    liver_dataset = Validation_Dataset(args.data_file, transform=x_transforms, target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size=1)
    ans = []
    if manual:
        model = torch.load(weight_path, map_location='cpu')
    model.eval()
    epoch_correct_pixels, epoch_total_pixels, OTSU_threshold = [], [], []
    for x, y, x_path in dataloaders:
        inputs = x.to(device)
        labels = y.to(device)
        predicted = model(inputs).detach().numpy()
        if save_each:
            io.imsave('compare/' + x_path[0].split('\\')[-1].split('.')[0] + "_gray_pre.png",
                      torch.squeeze(torch.tensor(predicted)).numpy())
        predicted[predicted >= threshold] = 1
        predicted[predicted < threshold] = 0
        correct = (predicted == labels.detach().numpy()).sum()
        if save_each:
            io.imsave('compare/' + x_path[0].split('\\')[-1].split('.')[0] + "_pre.png",
                      torch.squeeze(torch.tensor(predicted)).numpy())
            io.imsave('compare/' + x_path[0].split('\\')[-1].split('.')[0] + "_inputs.png", torch.squeeze(x).numpy())
            io.imsave('compare/' + x_path[0].split('\\')[-1].split('.')[0] + "_label.png", torch.squeeze(y).numpy())

        # predicted = model(inputs)
        # io.imsave("mid_delete.png", torch.squeeze(predicted).detach().numpy())
        # predicted = cv2.imread("mid_delete.png", cv2.IMREAD_GRAYSCALE)
        # mid_OTSU_threshold, predicted = cv2.threshold(predicted, 0, 255, cv2.THRESH_OTSU)
        # predicted = torch.from_numpy(predicted) / 255.0
        # OTSU_threshold.append(mid_OTSU_threshold)
        # if print_each:
        #     print('OTSU_threshold', mid_OTSU_threshold)
        # correct = (predicted.numpy() == labels.detach().numpy()).sum()

        del predicted
        pixel_num = 1.0
        for i in range(len(labels.size())):
            pixel_num *= labels.size()[i]
        epoch_correct_pixels.append(correct)
        epoch_total_pixels.append(pixel_num)
        if print_each:
            with open("compare.txt", "a") as f:
                f.write(str(x_path) + 'acc' + str(correct / pixel_num) + '\n')  # 这句话自带文件关闭功能，不需要再写f.close()
                print(x_path, 'acc', correct / pixel_num)
                ans.append(correct / pixel_num)
    if list_return:
        return ans
    else:
        return np.mean(epoch_correct_pixels) / np.mean(epoch_total_pixels)


# 显示模型的输出结果
def test(args, save_gray=False, manual=False, weight_path='', test_all_weights=False):
    model = None
    if not manual:
        model = torch.load(args.ckpt + '/' + args.model_name + '.pth', map_location='cpu')
    if manual:
        model = torch.load(weight_path, map_location='cpu')  # use certain model weight.

    liver_dataset = Test_Dataset(args.data_file, transform=x_transforms, target_transform=y_transforms)
    if test_all_weights:
        liver_dataset = All_Test_Dataset(args.data_file, transform=x_transforms, target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size=1)

    # print("val_accuracy", validation(args, model, True))  # train 0.88340  val 0.856924

    model.eval()
    with torch.no_grad():
        OTSU_threshold = []
        for x, pic_name_i in dataloaders:
            # print(x)
            pic_name_i = pic_name_i[0]
            io.imsave(args.model_name + "/results/" + pic_name_i.split('.')[0] + "_x.png", torch.squeeze(x).numpy())
            predict = model(x)
            predict = torch.squeeze(predict).detach().numpy()

            # predict = model(x)
            # img_y = cv2.imread(args.model_name + "/results/" + pic_name_i.split('.')[0] + "_gray_pre.png",
            #                    cv2.IMREAD_GRAYSCALE)
            # mid_OTSU_threshold, img_y = cv2.threshold(predict, 0, 255, cv2.THRESH_OTSU)
            # print(mid_OTSU_threshold)
            # OTSU_threshold.append(mid_OTSU_threshold)
            # img_y[img_y >= mid_OTSU_threshold] = 1
            # img_y[img_y < mid_OTSU_threshold] = 0
            # io.imsave(args.model_name + "/results/" + pic_name_i.split('.')[0] + "_label_pre.png", img_y)
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
    # print(model)
    # print('one layer name')
    # print(model.Up2.up)
    # print(model.Up2.up[0])
    # print('layer number', k1, k2)
    # print(len(save_output.outputs))

    x = x_transforms(Image.open(image_path).convert('L').resize(size=(512, 512))).unsqueeze(
        0)  # prepare the input image torch.Size([1, 1, 512, 512])
    print(x, x.dtype)
    y = model(x)

    # print(len(save_output.outputs))

    def module_output_to_numpy(tensor):
        return tensor.detach().to('cpu').numpy()

    # for x in save_output.outputs:
    #     print(x.shape)
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
            # plt.show()
            plt.cla()
            plt.close('all')


def model_parameter_visualization(weight_path):
    """使用一个全为1的torch，输入训练好的模型权重，可视化每一步卷积的结果"""
    model = torch.load(weight_path, map_location='cpu')  # load trained model

    save_output = SaveOutput()  # register hooks for each layer
    hook_handles, k1, k2 = [], 0, 0
    for layer in model.modules():
        k1 += 1
        if isinstance(layer, torch.nn.modules.conv.Conv2d):
            k2 += 1
            handle = layer.register_forward_hook(save_output)
            hook_handles.append(handle)

    x = torch.full([1, 1, 512, 512], 1, dtype=torch.float32)  # prepare the input image
    print(x)
    y = model(x)

    # print(len(save_output.outputs))

    def module_output_to_numpy(tensor):
        return tensor.detach().to('cpu').numpy()

    # for x in save_output.outputs:
    #     print(x.shape)
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
            plt.savefig('./model_visualization/layer_' + str(layer_idx) + '_mid_' + str(mid_idx) + '.png')
            # plt.show()
            plt.cla()
            plt.close('all')


# model_forward_visualization(image_path="./data/train/image/8.png",
#                             weight_path="./Attention_Unet_500epoch/weights/Attention_Unet_500epoch_weights_90.pth")
# model_parameter_visualization(weight_path="./Attention_Unet_500epoch/weights/Attention_Unet_500epoch_weights_90.pth")


def model_print(model, input_shape=(1, 512, 512)):
    # 1. print parameter number
    # 2. draw model structure

    # from torchstat import stat
    # stat(model, input_shape)

    print(sum(p.numel() for p in model.parameters()))


def all_test(data_file='./data', model_path='./weights/', save_gray=True):
    # 使用指定的数据集，测试存储在weights文件夹中的每一个模型在全部图像上的分割情况。test图像可以没有对应的label
    model_files = os.listdir(model_path)
    for i in range(len(model_files)):
        # if model_files[i] != 'd1_NestedUNet_250epoch.pth' and model_files[i] != 'd1_my_model4_250epoch.pth':
        #     continue
        liver_dataset = All_Test_Dataset(data_file, transform=x_transforms, target_transform=y_transforms)
        dataloaders = DataLoader(liver_dataset, batch_size=1)

        weight_path = model_path + model_files[i]
        model_name = model_files[i].split('.')[0]
        print(weight_path, model_name)
        makedir("all_test_results/" + model_name)
        model = torch.load(weight_path, map_location='cpu')
        model.eval()
        with torch.no_grad():
            for x, pic_name_i in dataloaders:
                pic_name_i = pic_name_i[0]
                io.imsave("all_test_results/" + model_name + "/" + pic_name_i.split('.')[0] + "_x.png",
                          torch.squeeze(x).numpy())
                predict = model(x)
                predict = torch.squeeze(predict).detach().numpy()
                if save_gray:
                    io.imsave("all_test_results/" + model_name + "/" + pic_name_i.split('.')[0] + "_gray_pre.png",
                              predict)
                predict[predict >= threshold] = 1
                predict[predict < threshold] = 0
                io.imsave("all_test_results/" + model_name + "/" + pic_name_i.split('.')[0] + "_label_pre.png", predict)

# all_test()
# model_forward_visualization(image_path="./d1/val/image/140.png", weight_path="./weights/d1_my_model1_250epoch.pth",
#                             model_name='d1_my_model1_250epoch_2')
# model_forward_visualization(image_path="./d1/val/image/140.png",
#                             weight_path="./weights/Attention_Unet_500epoch_weights_90.pth", model_name='Attention_Unet')
