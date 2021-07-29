import os
import numpy as np
from skimage import io
import matplotlib.pyplot as plt


def source_tiff_image_to_PNG():
    GF_tiff_names = ['GF3_MDJ_UFS_818032_E122.8_N39.1_20200112_L2_DH_L20004924997',
                     'GF3_MDJ_UFS_818032_E122.9_N39.4_20200112_L2_DH_L20004924983',
                     'GF3_MYN_UFS_018860_E122.6_N39.4_20200310_L2_DH_L20004918498',
                     'GF3_MYN_UFS_018860_E122.7_N39.2_20200310_L2_DH_L20004918494']
    for file_name in GF_tiff_names:
        img = io.imread('./source_images/' + file_name + '.tiff').astype(np.uint8)  # tiff格式可能默认为uint16
        io.imsave('./source_images/' + file_name + '.png', img)


def Mid_label2label():
    # black take place red and white take place other colors in mid_label
    mid_label_names = ['1', '16', '17', '18']
    for k in range(len(mid_label_names)):
        img = io.imread('./mid_dataset/mid_labels/' + str(mid_label_names[k]) + '.png')
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img[i, j, 0] >= 200 and img[i, j, 1] <= 40 and img[i, j, 2] <= 40:
                    img[i, j, 0], img[i, j, 1], img[i, j, 2] = 0, 0, 0
                else:
                    img[i, j, 0], img[i, j, 1], img[i, j, 2] = 255, 255, 255
        io.imsave('./mid_dataset/labels/' + str(mid_label_names[k]) + '_label.png', img)


def crop_images_labels():
    names = ['1', '16', '17', '18']
    for k in range(len(names)):
        img = io.imread('./mid_dataset/images/' + str(names[k]) + '.png', as_gray=True)
        label = io.imread('./mid_dataset/labels/' + str(names[k]) + '_label.png', as_gray=True)

        if not img.shape == label.shape:
            print('Error: shape of image is different with label image.')
            print('shape of image is', img.shape, 'shape of label is ', label.shape)
            print('Check it please!')
            return

        small_pic_name_k = 0
        for i in range(0, (img.shape[0] - img.shape[0] % 1024) + 1, 1024):
            if i + 1024 > img.shape[0]:
                break
            for j in range(0, (img.shape[1] - img.shape[1] % 1024) + 1, 1024):
                if j + 1024 > img.shape[1]:
                    break
                io.imsave('./dataset/images/' + str(names[k]) + '_' + str(small_pic_name_k) + '.png',
                          img[i:i + 1024, j:j + 1024])
                io.imsave('./dataset/labels/' + str(names[k]) + '_' + str(small_pic_name_k) + '.png',
                          label[i:i + 1024, j:j + 1024])
                small_pic_name_k += 1


def rename_dataset():
    # for debug
    # os.remove('./dataset/labels/1_6.png')
    # os.remove('./dataset/labels/1_7.png')
    # os.remove('./dataset/labels/1_8.png')
    image_file_path = './dataset/images/'
    label_file_path = './dataset/labels/'
    image_files = os.listdir(image_file_path).copy()
    label_files = os.listdir(label_file_path).copy()

    for i in range(len(image_files)):
        if image_files[i] not in label_files:
            os.remove(image_file_path + image_files[i])

    image_files = os.listdir(image_file_path).copy()
    label_files = os.listdir(label_file_path).copy()

    for i in range(len(label_files)):
        os.rename(label_file_path + label_files[i], label_file_path + str(i) + '.png')
        os.rename(image_file_path + image_files[i], image_file_path + str(i) + '.png')


def crop_all_test():
    names = [1, 9, 10, 11, 17, 18, 19, 20, 21]
    big_image_file_path = 'F:\历史文件\postgraduate\Cityu courses\\6006\mid_dataset\images/'
    big_image_files = os.listdir(big_image_file_path).copy()
    print(big_image_files)
    for k in range(len(names)):
        img = io.imread(big_image_file_path + str(names[k]) + '.png', as_gray=True)
        small_pic_name_k = 0
        for i in range(0, (img.shape[0] - img.shape[0] % 1024) + 1, 1024):
            if i + 1024 > img.shape[0]:
                break
            for j in range(0, (img.shape[1] - img.shape[1] % 1024) + 1, 1024):
                if j + 1024 > img.shape[1]:
                    break
                io.imsave('F:\历史文件/postgraduate/Cityu courses/6006/mid_dataset/all_test/' + str(names[k]) + '_' + str(
                    small_pic_name_k) + '.png',
                          img[i:i + 1024, j:j + 1024])
                small_pic_name_k += 1


crop_all_test()
# source_tiff_image_to_PNG()  # tiff->png
# rotate and crop images by PhotoShop. Then use PS to add red labels on images to generate mid_label images.
# Mid_label2label()  # transform red-gray labels to black-white labels.
# crop_images_labels()  # crop big images smaller
# drop bad pictures by hand
# rename_dataset()  # rename all pictures
