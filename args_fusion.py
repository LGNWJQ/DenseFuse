# -*- coding: utf-8 -*-
"""
Writer: WJQpe
Date: 2022 04 01 
"""
import argparse


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default="WJQ", help="Coder Name")
    parser.add_argument('--image_path', default=r'D:/MyDataset/train2014', type=str, help='训练集路径')
    parser.add_argument('--gray', default=True, type=bool, help='是否使用灰度模式')
    parser.add_argument('--batch_size', default=32, type=int, help='批量大小')
    parser.add_argument('--image_num', default=10000, type=int, help='用于训练的图像数量')
    parser.add_argument('--num_workers', default=8, type=int, help='载入数据集所调用的cpu线程数')
    parser.add_argument('--learning_rate', default=1e-4, type=float, help='学习率')
    parser.add_argument('--epochs', default=1, type=int, help='训练轮数')
    parser.add_argument('--resume_path', default=None, type=str, help='导入已训练好的模型路径')
    parser.add_argument('--save_model_dir', default='./weight', type=str, help='保存权重的路径')
    parser.add_argument('--ssim_weight', default=[1, 10, 100, 1000, 10000], type=list, help='ssim损失的权重')
    parser.add_argument('--tensorboard_step', default=50, type=int, help='tensorboard的更新步数')
    parser.add_argument('--use_lr_scheduler', default=False, type=bool, help='是否使用学习率调度')

    return parser.parse_args()

