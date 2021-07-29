from torch.utils.data import Dataset
import PIL.Image as Image
import os


def make_dataset(root, dataset_usage='train'):
    data_set = []
    if dataset_usage == 'train':
        imgs, labels = [], []
        n_image = len(os.listdir(root + '/train/image'))
        n_label = len(os.listdir(root + '/train/label'))
        for i in range(n_image):
            imgs.append(os.path.join(root + '/train/image', os.listdir(root + '/train/image')[i]))
        for i in range(n_label):
            labels.append(os.path.join(root + '/train/label', os.listdir(root + '/train/label')[i]))
        for i in range(len(imgs)):
            data_set.append((imgs[i], labels[i]))
    elif dataset_usage == 'val':
        imgs, labels = [], []
        n_image = len(os.listdir(root + '/val/image'))  #
        n_label = len(os.listdir(root + '/val/label'))
        for i in range(n_image):
            imgs.append(os.path.join(root + '/val/image', os.listdir(root + '/val/image')[i]))
        for i in range(n_label):
            labels.append(os.path.join(root + '/val/label', os.listdir(root + '/val/label')[i]))
        for i in range(len(imgs)):
            data_set.append((imgs[i], labels[i]))
    elif dataset_usage == 'test':
        for i in range(len(os.listdir(root + '/test'))):
            # print(os.listdir(root + '/test')[i])
            data_set.append(os.path.join(root + '/test', os.listdir(root + '/test')[i]))
    elif dataset_usage == 'all_test':
        for i in range(len(os.listdir(root + '/all_test'))):
            # print(os.listdir(root + '/test')[i])
            data_set.append(os.path.join(root + '/all_test', os.listdir(root + '/all_test')[i]))

    return data_set


class Train_Dataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        imgs = make_dataset(root, dataset_usage='train')
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        img_x = Image.open(x_path).convert('L').resize(size=(512, 512))
        img_y = Image.open(y_path).convert('L').resize(size=(512, 512))
        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
        return img_x, img_y

    def __len__(self):
        return len(self.imgs)


class Validation_Dataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        imgs = make_dataset(root, dataset_usage='val')
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        img_x = Image.open(x_path).convert('L').resize(size=(512, 512))
        img_y = Image.open(y_path).convert('L').resize(size=(512, 512))
        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
        return img_x, img_y, x_path

    def __len__(self):
        return len(self.imgs)


class Test_Dataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        imgs = make_dataset(root, dataset_usage='test')
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path = self.imgs[index]
        img_x = Image.open(x_path).convert('L').resize(size=(512, 512))
        if self.transform is not None:
            img_x = self.transform(img_x)
        # print(x_path) # extract test picture name/number
        pic_name = x_path.split(sep='/')[-1].split(sep='\\')[-1]
        # print(type(pic_name), pic_name)
        return img_x, pic_name

    def __len__(self):
        return len(self.imgs)


class All_Test_Dataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        imgs = make_dataset(root, dataset_usage='all_test')
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path = self.imgs[index]
        img_x = Image.open(x_path).convert('L').resize(size=(512, 512))
        if self.transform is not None:
            img_x = self.transform(img_x)
        # print(x_path) # extract test picture name/number
        pic_name = x_path.split(sep='/')[-1].split(sep='\\')[-1]
        # print(type(pic_name), pic_name)
        return img_x, pic_name

    def __len__(self):
        return len(self.imgs)
