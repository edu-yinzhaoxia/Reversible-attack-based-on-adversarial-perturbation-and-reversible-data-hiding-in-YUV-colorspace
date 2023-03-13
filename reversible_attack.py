import torch
from torchvision import models
import json
import torchvision.transforms as transforms
from utils import image_folder_custom_label, rgb2yuv, yuv2rgb
import torchattacks
from embed_utils import embed_main
import numpy as np
from torchvision import utils as Vutils
from PIL import Image
import random
from pytorch_grad_cam import CAM


random.seed(2022)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 加载模型
model = models.inception_v3(pretrained=True).to(device)
model.eval()

# load data
class_idx = json.load(open('G:/experiment/Third/imagenet_class_index.json'))
idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]

transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
])

imagnet_data = image_folder_custom_label(root='G:/experiment/Third/final/ori/', transform=transform, idx2label=idx2label)
data_loader = torch.utils.data.DataLoader(imagnet_data, batch_size=1, shuffle=True)

images, labels = iter(data_loader).next()

# adversarial attack
# atk = torchattacks.FGSM(model, eps=2/255)
atk = torchattacks.CW(model, c=1, lr=0.01, steps=100, kappa=100)
# atk = torchattacks.BIM(model, eps=2/255, alpha=1/255, steps=20)

# attention model
model1 = models.resnet50(pretrained=True).to(device)

target_layer = model1.layer4[-1]
cam = CAM(model=model1, target_layer=target_layer, use_cuda=True)


def cam_map(image):

    target_category = None
    grayscale_cam = cam(input_tensor=image,
                        method='gradcam++',
                        target_category=target_category)

    return grayscale_cam


all_count = 0
adv_y_fault = 0

for images, labels in data_loader:

    # 对于原始图像预测正确
    images, labels = images.to(device), labels.to(device)
    ori_outputs = model(images)
    _, ori_pre = torch.max(ori_outputs.data, 1)
    if ori_pre != labels:
        continue

    all_count = all_count + 1

    if all_count > 10000:
        break

    # 保存原始图像
    ori_path = 'G:/experiment/Third/code/torchattack1/result/CW_100/ori/' + str(all_count) + '_' + str(labels.item()) + '.png'
    Vutils.save_image(images, ori_path, normalize=True)

    # 生成注意力图
    ori_map = cam_map(images)
    # 二值化
    hight, width = ori_map.shape
    for i in range(hight):
        for j in range(width):
            if ori_map[i][j] > 0.5:
                ori_map[i][j] = 1
            else:
                ori_map[i][j] = 0

    # 将原始rgb转化为yuv444
    ori_yuv = rgb2yuv(images)

    for max_iteration in range(50):
        adv_images = atk(images, labels)

        # 将对抗rgb转yuv
        adv_yuv = rgb2yuv(adv_images)

        # 保留y通道对抗扰动，生成对抗样本
        advy_yuv = ori_yuv.copy()
        # advy_yuv[:, :, 0] = adv_yuv[:, :, 0]
        for i in range(299):
            for j in range(299):
                if ori_map[i][j] == 1:
                    advy_yuv[:, :, 0][i][j] = adv_yuv[:, :, 0][i][j]

        # 将y通道的对抗扰动嵌入uv通道
        rae_yuv = advy_yuv.copy()
        reversible_yuv = embed_main(ori_yuv, advy_yuv)
        reversible_u = reversible_yuv[:299, :]
        reversible_v = reversible_yuv[299:, :]
        rae_yuv[:, :, 1] = reversible_u
        rae_yuv[:, :, 2] = reversible_v

        reversible_ndarray = yuv2rgb(rae_yuv)
        reversible_image = Image.fromarray(np.uint8(reversible_ndarray))
        save_image = reversible_image.copy()

        # 将生成的可逆对抗样本输入模型进行预测
        reversible_tensor = transform(reversible_image).unsqueeze(0).to(device)
        outputs = model(reversible_tensor)  # 模型输入：tensor(1, 3, 299, 299)

        _, pre = torch.max(outputs.data, 1)
        if pre != labels:
            adv_y_fault = adv_y_fault + 1
            break
        else:
            # 将advy传入下一循环
            advy_ndarray = yuv2rgb(advy_yuv)
            advy_image = Image.fromarray(np.uint8(advy_ndarray))
            advy_tensor = transform(advy_image).unsqueeze(0).to(device)

            images = advy_tensor  # 这里的图像yuv通道都不一样了

    print('all_count, max_iteration', all_count, max_iteration)

    rae_path = 'G:/experiment/Third/code/torchattack1/result/CW_100/rae/' + str(all_count) + '_' + str(
        labels.item()) + '.png'
    Vutils.save_image(reversible_tensor, rae_path, normalize=True)

    # save_result_image2 = Image.fromarray(np.uint8(save_image))
    # save_result_image_name = 'result_FGSM/0.01/rae/' + str(all_count) + '_' + str(labels.item()) + '.png'
    # save_result_image2.save(save_result_image_name)

print('adv_y_fault, all_count', adv_y_fault, all_count)











