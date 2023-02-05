# DenseFuse

* 这是[Densefuse](https://arxiv.org/abs/1804.08361)的非官方PyTorch实现，参考官方代码库[DenseFuse](https://github.com/hli1221/imagefusion_densefuse)进行实现，并进行了一定程度的改进（使用较新的PyTorch版本，规避了原代码库中的一些版本较低和不常用的包）

* 论文参考：*H. Li, X. J. Wu, “DenseFuse: A Fusion Approach to Infrared and Visible Images,” IEEE Trans. Image Process., vol. 28, no. 5, pp. 2614–2623, May. 2019.*

  - [IEEEXplore](https://ieeexplore.ieee.org/document/8580578)

  - [arXiv](https://arxiv.org/abs/1804.08361)

## 文件结构

```shell
├─fusion_result     # 使用我训练好的权重对test_data内的图像进行融合的结果 
│  ├─fusion_result_Exposure
│  ├─fusion_result_Lytro
│  ├─fusion_result_Road
│  └─fusion_result_TNO
|
├─logs           	# 用于存储训练过程中产生的Tensorboard文件,这里不提供（文件过大，无法上传）
│  
├─pytorch_msssim 	# 用于计算SSIM损失，来自官方代码库
|
├─test_data      	# 用于测试的不同图片
│  ├─Exposure      		# RGB   多曝光
│  ├─Lytro         		# RGB   多聚焦 
│  ├─Road          		# Gray  可见光+红外
│  └─Tno           		# Gray  可见光+红外
|
├─weight         	# 保存训练好的权重
│ 
├─args_fusion.py 	# 在该文件里修改训练参数
│ 
├─dataset.py        # 该文件为自定义数据集，内含Transforms数据增强方法
│ 
├─fusion_image.py   # 该文件使用训练好的权重将test_data内的测试图像进行融合
│ 
├─network.py        # 该文件里定义了网络模型和初始化方法
│ 
└─train.py          # 该文件用于训练模型

```



## 使用说明

### Trainng

#### 从零开始训练

* 打开args_fusion.py对训练参数进行设置：
* 参数说明：

| 参数名           | 说明                                                         |
| ---------------- | ------------------------------------------------------------ |
| image_path       | 用于训练的数据集的路径                                       |
| gray             | 为`True`时会进入灰度图训练模式，生成的权重用于对单通道灰度图的融合（如test_data里的Road和Tno）; 为`False`时会进入彩色RGB图训练模式，生成的权重用于对三通道彩色图的融合（如test_data里的Exposure和Lytro）; |
| image_num        | `MSCOCO/train2014`数据集包含**82,783**张图像，设置该参数来确定用于训练的图像的数量 |
| num_workers      | 加载数据集时使用的CPU工作进程数量，为0表示仅使用主进程，（在Win10下建议设为0，否则可能报错。Win11下可以根据你的CPU线程数量进行设置来加速数据集加载） |
| learning_rate    | 训练初始学习率                                               |
| epochs           | 训练轮数                                                     |
| resume_path      | 默认为None，设置为已经训练好的**权重文件路径**时可对该权重进行继续训练，注意选择的权重要与**gray**参数相匹配 |
| save_model_dir   | 训练好的权重文件的保存路径                                   |
| ssim_weight      | 对SSIM损失的增益系数列表                                     |
| tensorboard_step | 控制对Tensorboard的写入周期，单位为参数迭代次数              |
| use_lr_scheduler | 是否使用学习率调度，从零开始训练建议设置为True，可以进入train.py的67行左右对其内部参数进行进一步修改 |

* 设置完成参数后，运行**train.py**即可开始训练：

```python
parser.add_argument('--image_path', default='C:/Users/WJQpe/Desktop/DataSets/MSCOCO/train2014', type=str, help='训练集路径')
parser.add_argument('--gray', default=False, type=bool, help='是否使用灰度模式')
parser.add_argument('--batch_size', default=1, type=int, help='批量大小')
parser.add_argument('--image_num', default=20000, type=int, help='用于训练的图像数量')
parser.add_argument('--num_workers', default=0, type=int, help='载入数据集所调用的cpu线程数')
parser.add_argument('--learning_rate', default=1e-4, type=float, help='学习率')
parser.add_argument('--epochs', default=1, type=int, help='训练轮数')
parser.add_argument('--resume_path', default=None, type=str, help='导入已训练好的模型路径')
parser.add_argument('--save_model_dir', default='./weight', type=str, help='保存权重的路径')
parser.add_argument('--ssim_weight', default=[1, 10, 100, 1000, 10000], type=list, help='ssim损失的权重')
parser.add_argument('--tensorboard_step', default=50, type=int, help='tensorboard的更新步数')
parser.add_argument('--use_lr_scheduler', default=True, type=bool, help='是否使用学习率调度')
```

* 你可以在运行窗口看到如下信息：

```
name WJQ
image_path D:/MyDataset/train2014
gray True
batch_size 32
image_num 80000
num_workers 8
learning_rate 0.0001
epochs 5
resume_path None
save_model_dir ./weight
ssim_weight [1, 10, 100, 1000, 10000]
训练数据载入完成...
测试数据载入完成...
网络模型及优化器构建完成...
Tensorboard 构建完成，进入路径：./logs/0404_20-13_Gray_epoch=5
然后使用该指令查看训练过程：tensorboard --logdir=./
Start Training on NVIDIA GeForce RTX 3080 Laptop GPU...
Epoch [1/5]: 100%|██████████| 2500/2500 [13:39<00:00,  3.05it/s, pixel_loss=0.000483, ssim_loss=0.00118]
Epoch [2/5]: 100%|██████████| 2500/2500 [14:11<00:00,  2.93it/s, pixel_loss=0.000132, ssim_loss=0.00029]
Epoch [3/5]: 100%|██████████| 2500/2500 [13:41<00:00,  3.04it/s, pixel_loss=7.08e-5, ssim_loss=0.000103]
Epoch [4/5]: 100%|██████████| 2500/2500 [13:46<00:00,  3.02it/s, pixel_loss=6.24e-5, ssim_loss=3.74e-5]
Epoch [5/5]: 100%|██████████| 2500/2500 [13:49<00:00,  3.01it/s, pixel_loss=4.24e-5, ssim_loss=1.91e-5]
训练完成...
模型数据已保存在：./weight./0404_20-13_Gray_epoch=5.pt

Process finished with exit code 0
```

* Tensorboard查看训练细节：
  * **logs**文件夹下保存Tensorboard文件，当前训练文件的命名格式为：**时间（月-日-时-分) + 训练模式（灰度或彩色）+ 训练轮数**如`0404_20-13_Gray_epoch=5`，训练完成后生成的权重文件命名方式也是这个
  * 进入对于文件夹后使用该指令查看训练过程：`tensorboard --logdir=./`
  * 在浏览器打开生成的链接即可查看训练细节

#### 使用我提供的权重继续训练

* 打开args_fusion.py对训练参数进行设置
* 首先确定训练模式（Gray or RGB）
* 修改**resume_path**的默认值为已经训练过的权重文件路径

* 运行**train.py**即可运行



### Fuse Image

* 打开**fusion_image.py**文件
  * 确定融合模式（Gray or RGB）
  * 确定原图像路径和权重路径
  * 确定保存路径
* 运行**fusion_image.py**
* 你可以在运行窗口看到如下信息：

```shell
获取测试设备...
测试设备为：NVIDIA GeForce GTX 1650...
开始构建网络...
开始载入权重...
载入完成！！！
设置网络为评估模式...
载入数据...
开始融合...
输出路径：./fusion_result2/fusion_result_Tno/fusion1.png融合耗时：2.2450976371765137
输出路径：./fusion_result2/fusion_result_Tno/fusion2.png融合耗时：0.2570321559906006
输出路径：./fusion_result2/fusion_result_Tno/fusion3.png融合耗时：0.1409626007080078
输出路径：./fusion_result2/fusion_result_Tno/fusion4.png融合耗时：0.16060280799865723
输出路径：./fusion_result2/fusion_result_Tno/fusion5.png融合耗时：0.07599830627441406
输出路径：./fusion_result2/fusion_result_Tno/fusion6.png融合耗时：0.2500014305114746
输出路径：./fusion_result2/fusion_result_Tno/fusion7.png融合耗时：0.10700035095214844
输出路径：./fusion_result2/fusion_result_Tno/fusion8.png融合耗时：0.0429995059967041
输出路径：./fusion_result2/fusion_result_Tno/fusion9.png融合耗时：0.2480015754699707
输出路径：./fusion_result2/fusion_result_Tno/fusion10.png融合耗时：0.22699666023254395
输出路径：./fusion_result2/fusion_result_Tno/fusion11.png融合耗时：0.05703163146972656
输出路径：./fusion_result2/fusion_result_Tno/fusion12.png融合耗时：0.14099693298339844
输出路径：./fusion_result2/fusion_result_Tno/fusion13.png融合耗时：0.14799904823303223
输出路径：./fusion_result2/fusion_result_Tno/fusion14.png融合耗时：0.23700404167175293
输出路径：./fusion_result2/fusion_result_Tno/fusion15.png融合耗时：0.2499995231628418
输出路径：./fusion_result2/fusion_result_Tno/fusion16.png融合耗时：0.15399694442749023
输出路径：./fusion_result2/fusion_result_Tno/fusion17.png融合耗时：0.15993571281433105
输出路径：./fusion_result2/fusion_result_Tno/fusion18.png融合耗时：0.15099883079528809
输出路径：./fusion_result2/fusion_result_Tno/fusion19.png融合耗时：0.14999985694885254
输出路径：./fusion_result2/fusion_result_Tno/fusion20.png融合耗时：0.23000025749206543
输出路径：./fusion_result2/fusion_result_Tno/fusion21.png融合耗时：0.24300074577331543
```











