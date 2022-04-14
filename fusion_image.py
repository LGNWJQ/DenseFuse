# -*- coding: utf-8 -*-
"""
Writer: WJQpe
Date: 2022 04 06 
"""
import os
import re
import torch
from torchvision.io import read_image, ImageReadMode
from torchvision.utils import save_image
from torchvision import transforms
from matplotlib import pyplot as plt
import numpy as np
import time
from network import Dense_Encoder, CNN_Decoder
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

Gray_Test_list = ['Tno', 'Road']
RGB_test_list = ['Exposure', 'Lytro']

gray = True
if gray:
    IR_image_path = 'test_data/Tno/IR_images/'
    VIS_image_path = 'test_data/Tno/VIS_images/'
    result_path = './fusion_result2/fusion_result_Tno/'

    # IR_image_path = 'test_data/Road/1/'
    # VIS_image_path = 'test_data/Road/2/'
    # result_path = './fusion_result2/fusion_result_Road/'

    weight_path = './weight/0406_19-31_Gray_epoch=6.pt'
else:
    IR_image_path = 'test_data/Lytro/1/'
    VIS_image_path = 'test_data/Lytro/2/'
    result_path = './fusion_result/fusion_result_Lytro/'

    # IR_image_path = 'test_data/Exposure/1/'
    # VIS_image_path = 'test_data/Exposure/2/'
    # result_path = './fusion_result/fusion_result_Exposure/'

    weight_path = './weight/0408_14-02_RGB_epoch=3.pt'


if not os.path.exists(result_path):
    os.makedirs(result_path)

print('获取测试设备...')
print("测试设备为：{}...".format(torch.cuda.get_device_name(0)))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('开始构建网络...')
in_channel = 1 if gray else 3
out_channel = 1 if gray else 3
Encoder = Dense_Encoder(input_nc=in_channel).to(device)
Decoder = CNN_Decoder(output_nc=out_channel).to(device)

print('开始载入权重...')
checkpoint = torch.load(weight_path)
Encoder.load_state_dict(checkpoint['encoder_state_dict'])
Decoder.load_state_dict(checkpoint['decoder_state_dict'])
print('载入完成！！！')

print('设置网络为评估模式...')
Encoder.eval()
Decoder.eval()

print('载入数据...')
IR_image_list = os.listdir(IR_image_path)
VIS_image_list = os.listdir(VIS_image_path)
IR_image_list = sorted(IR_image_list, key=lambda i: int(re.search(r'(\d+)', i).group()))
VIS_image_list = sorted(VIS_image_list, key=lambda i: int(re.search(r'(\d+)', i).group()))

tf_list = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor()
            ]
        )

print('开始融合...')
num = 0
for IR_image_name, VIS_image_name in zip(IR_image_list, VIS_image_list):
    with torch.no_grad():
        time_start = time.time()  # 记录开始时间
        num += 1
        # 读取图像并进行处理
        IR_image = read_image(IR_image_path + IR_image_name, mode=ImageReadMode.GRAY if gray else ImageReadMode.RGB)
        VIS_image = read_image(VIS_image_path + VIS_image_name, mode=ImageReadMode.GRAY if gray else ImageReadMode.RGB)
        IR_image = tf_list(IR_image).unsqueeze(0).to(device)
        VIS_image = tf_list(VIS_image).unsqueeze(0).to(device)

        # 将图片进行编码
        IR_image_EN = Encoder(IR_image)
        VIS_image_EN = Encoder(VIS_image)

        # 进行融合
        Fusion_image_feature = (IR_image_EN + VIS_image_EN) / 2
        # Fusion_image_feature = torch.maximum(IR_image_EN, VIS_image_EN)

        # 进行解码
        Fusion_image = Decoder(Fusion_image_feature)

        # 张量后处理
        Fusion_image = Fusion_image.detach().cpu()
        Fusion_image = Fusion_image[0]
        # plt.axis("off")
        # plt.imshow(np.transpose(Fusion_image, (1, 2, 0)), cmap='gray')
        # plt.show()
        save_image(Fusion_image, result_path+'fusion{}.png'.format(num))
        time_end = time.time()  # 记录结束时间

        time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
        print('输出路径：'+result_path+'fusion{}.png'.format(num)+'融合耗时：{}'.format(time_sum))
