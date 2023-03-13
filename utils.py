import torchvision.datasets as dsets
import numpy as np
import torch


def image_folder_custom_label(root, transform, idx2label):
    # custom_label
    # type : List
    # index -> label
    # ex) ['tench', 'goldfish', 'great_white_shark', 'tiger_shark']

    old_data = dsets.ImageFolder(root=root, transform=transform)
    old_classes = old_data.classes

    label2idx = {}

    for i, item in enumerate(idx2label):
        label2idx[item] = i

    new_data = dsets.ImageFolder(root=root, transform=transform,
                                 target_transform=lambda x: idx2label.index(old_classes[x]))
    new_data.classes = idx2label
    new_data.class_to_idx = label2idx

    return new_data


def l2_distance(model, images, adv_images, labels, device="cuda"):
    outputs = model(adv_images)
    _, pre = torch.max(outputs.data, 1)
    corrects = (labels.to(device) == pre)
    delta = (adv_images - images.to(device)).view(len(images), -1)
    l2 = torch.norm(delta[~corrects], p=2, dim=1).mean()
    return l2


def rgb2yuv(image):
    """image: tensor(1, 3, 224, 224)"""

    image = image[0].cpu()
    npimg = 255 * (image.numpy())
    img = np.transpose(npimg, (1, 2, 0))

    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    yuv = img.copy()
    yuv[:, :, 0] = np.round(0.299 * R + 0.587 * G + 0.114 * B)
    yuv[:, :, 1] = np.round(- 0.1687 * R - 0.3313 * G + 0.5 * B + 128)
    yuv[:, :, 2] = np.round(0.5 * R - 0.4187 * G - 0.0813 * B + 128)

    # yuv[:, :, 0] = 0.299 * R + 0.587 * G + 0.114 * B
    # yuv[:, :, 1] = - 0.1687 * R - 0.3313 * G + 0.5 * B + 128
    # yuv[:, :, 2] = 0.5 * R - 0.4187 * G - 0.0813 * B + 128

    # yuv1 = yuv.copy()
    # for i in range(224):
    #     for j in range(224):
    #         yuv1[i, j, 0] = round(yuv[i, j, 0])
    #         yuv1[i, j, 1] = round(yuv[i, j, 1])
    #         yuv1[i, j, 2] = round(yuv[i, j, 2])

    return yuv


def yuv2rgb(yuv):
    """yuv, ndarray(224, 224, 3)"""
    Y = yuv[:, :, 0]
    U = yuv[:, :, 1]
    V = yuv[:, :, 2]

    R = np.maximum(0, np.minimum(255, np.round(Y + 1.402 * (V - 128))))
    G = np.maximum(0, np.minimum(255, np.round(Y - 0.34414 * (U - 128) - 0.71414 * (V - 128))))
    B = np.maximum(0, np.minimum(255, np.round(Y + 1.772 * (U - 128))))
    # R = Y + 1.402 * (V - 128)
    # G = Y - 0.34414 * (U - 128) - 0.71414 * (V - 128)
    # B = Y + 1.772 * (U - 128)

    rgb = yuv.copy()
    rgb[:, :, 0] = R
    rgb[:, :, 1] = G
    rgb[:, :, 2] = B

    # rgb1 = rgb.copy()
    # for i in range(224):
    #     for j in range(224):
    #         rgb1[i, j, 0] = round(rgb[i, j, 0])
    #         rgb1[i, j, 1] = round(rgb[i, j, 1])
    #         rgb1[i, j, 2] = round(rgb[i, j, 2])

    # rgb1 = rgb1.astype('uint8')

    return rgb