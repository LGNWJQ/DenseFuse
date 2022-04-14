# -*- coding: utf-8 -*-
"""
Writer: WJQpe
Date: 2022 04 03 
"""
import os.path
import datetime
from tqdm import tqdm

import torch
from torch.optim import lr_scheduler, Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision

import pytorch_msssim
from network import Train_Module, initialize_weights
from args_fusion import set_args
from dataset import COCO2014, transform_image


def main():
    # 获取计算设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 获取超参数
    args = set_args()
    for arg in vars(args):
        print(arg, ': ', getattr(args, arg))

    # 导入数据集
    coco_dataset = COCO2014(
        images_path=args.image_path,
        transform=transform_image(gray=args.gray),
        image_num=args.image_num
    )
    train_data_loader = DataLoader(
        dataset=coco_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    print('训练数据载入完成...')

    test_data_loader = DataLoader(
        dataset=coco_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=args.num_workers
    )
    # 导入测试图像
    for i, batch in enumerate(test_data_loader):
        test_image = batch
        break
    test_image = test_image.to(device)
    print('测试数据载入完成...')

    # 初始化网络
    in_channel = 1 if args.gray else 3
    out_channel = 1 if args.gray else 3
    Train_network = Train_Module(input_nc=in_channel, output_nc=out_channel).to(device)
    print("Train_network have {} paramerters in total".format(sum(x.numel() for x in Train_network.parameters())))

    # 损失函数和迭代器
    optimizer = Adam(Train_network.parameters(), args.learning_rate)
    if args.use_lr_scheduler:
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[500, 1000, 1500, 3000, 5000], gamma=0.9)
    mse_loss = torch.nn.MSELoss()
    ssim_loss = pytorch_msssim.msssim

    # 是否迁移学习
    if args.resume_path is not None:
        checkpoint = torch.load(args.resume_path)
        Train_network.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        Train_network.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        pre_epoch = checkpoint['epoch']
    else:
        initialize_weights(Train_network)
        pre_epoch = 0
    print('网络模型及优化器构建完成...')

    # 训练过程记录
    train_time = datetime.datetime.now().strftime("%m%d_%H-%M")
    if args.gray:
        logs_path = train_time + '_Gray_' + 'epoch={}'.format(args.epochs + pre_epoch)
    else:
        logs_path = train_time + '_RGB_' + 'epoch={}'.format(args.epochs + pre_epoch)
    writer = SummaryWriter('./logs/' + logs_path)
    print('Tensorboard 构建完成，进入路径：' + './logs/' + logs_path)
    print('然后使用该指令查看训练过程：tensorboard --logdir=./')

    # 训练过程
    print("Start Training on {}...".format(torch.cuda.get_device_name(0)))
    Train_network.train()  # 训练模式
    step = 0  # 迭代次数
    for epoch in range(args.epochs):
        loop = tqdm(train_data_loader)
        for _, image_batch in enumerate(loop):
            # 清空梯度
            optimizer.zero_grad()
            # 载入图像
            image_batch = image_batch.to(device)
            # 复制图像作为标签
            lable = image_batch.data.clone().to(device)
            # 正向传播
            outputs = Train_network(image_batch)

            # 计算损失
            pixel_loss_value = mse_loss(outputs, lable)
            ssim_loss_value = 1 - ssim_loss(outputs, lable, normalize=True)
            total_loss = pixel_loss_value + args.ssim_weight[1] * ssim_loss_value

            # 计算梯度，反向传播
            total_loss.backward()
            optimizer.step()
            if args.use_lr_scheduler:
                scheduler.step()

            # 训练信息输出到进度条
            loop.set_description(f"Epoch [{epoch + 1}/{args.epochs}]")
            loop.set_postfix(
                pixel_loss=pixel_loss_value.item(),
                ssim_loss=ssim_loss_value.item(),
                learning_rate=scheduler.get_last_lr() if args.use_lr_scheduler else args.learning_rate
            )


            step += 1
            # 测试图像重建结果
            if step % args.tensorboard_step == 1:
                with torch.no_grad():
                    writer.add_scalar('pixel_loss', pixel_loss_value.item(), global_step=step)
                    writer.add_scalar('ssim_loss', ssim_loss_value.item(), global_step=step)
                    writer.add_scalar('total_loss', total_loss.item(), global_step=step)

                    rebuild_img = Train_network(test_image)

                    img_grid_real = torchvision.utils.make_grid(
                        test_image, normalize=True, nrow=4
                    )
                    img_grid_rebuild = torchvision.utils.make_grid(
                        rebuild_img, normalize=True, nrow=4
                    )
                    writer.add_image('Real image', img_grid_real, global_step=1)
                    writer.add_image('Rebuild image', img_grid_rebuild, global_step=step)
            if step <= 100:
                with torch.no_grad():
                    rebuild_img_50 = Train_network(test_image)
                    img_grid_rebuild_50 = torchvision.utils.make_grid(
                        rebuild_img_50, normalize=True, nrow=4
                    )
                    writer.add_image('Rebuild image 50', img_grid_rebuild_50, global_step=step)

    print('训练完成...')
    # 保存权重
    if not os.path.exists(args.save_model_dir):
        os.mkdir(args.save_model_dir)

    save_path = args.save_model_dir + './' + logs_path + '.pt'

    torch.save(
        {
            'encoder_state_dict': Train_network.encoder.state_dict(),
            'decoder_state_dict': Train_network.decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': (args.epochs + pre_epoch)
        }, save_path
    )
    print('模型数据已保存在：' + save_path)


if __name__ == "__main__":
    main()
