import os
from enum import Enum # 用于定义枚举类型（后面会用来定义数据集的划分：训练集、验证集、测试集）

import PIL # 用于加载图像文件。
import torch
from torchvision import transforms

_CLASSNAMES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
] # MVTec 数据集中的所有类别名称列表。

IMAGENET_MEAN = [0.485, 0.456, 0.406] # ImageNet 数据集的均值，用于图像归一化。
IMAGENET_STD = [0.229, 0.224, 0.225] # ImageNet 数据集的标准差，用于图像归一化。


class DatasetSplit(Enum):
    """
    定义了一个枚举类 DatasetSplit。这是一种为了代码规范的做法，用来代替直接使用字符串 "train" 或 "test"，防止后续代码拼写错误。
    """
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class MVTecDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for MVTec.
    DataLoader 的核心要求是：传入的对象必须实现以下两个方法：
      - __len__(): 返回数据集的大小（样本数量）。
      - __getitem__(idx): 根据索引 idx 返回对应的样本数据。
    任何自定义的 Dataset，只要继承了 PyTorch 的 Dataset，就具备这两个方法。
    """

    def __init__(
        self,
        source, # 数据集的根目录路径。
        classname, # 指定要加载的 MVTec 类别名称。如果为 None，则加载所有类别。
        resize=256, # 加载图片后首先缩放的大小（默认 256）
        imagesize=224, # 最终输入给模型的大小（默认 224），通常是通过中心裁剪得到的
        split=DatasetSplit.TRAIN, # 指定当前数据集是训练集、验证集还是测试集。
        train_val_split=1.0, # 用于划分训练集和验证集的比例（默认 1.0，表示不划分）。
        **kwargs,
    ):
        """
        Args:
            source: [str]. Path to the MVTec data folder.
            classname: [str or None]. Name of MVTec class that should be
                       provided in this dataset. If None, the datasets
                       iterates over all available images.
            resize: [int]. (Square) Size the loaded image initially gets
                    resized to.
            imagesize: [int]. (Square) Size the resized loaded image gets
                       (center-)cropped to.
            split: [enum-option]. Indicates if training or test split of the
                   data should be used. Has to be an option taken from
                   DatasetSplit, e.g. mvtec.DatasetSplit.TRAIN. Note that
                   mvtec.DatasetSplit.TEST will also load mask data.
        """
        super().__init__()
        self.source = source
        self.split = split
        self.classnames_to_use = [classname] if classname is not None else _CLASSNAMES
        self.train_val_split = train_val_split

        #这个方法负责扫描文件夹，找到所有的图片路径，并把它们整理好。
        #imgpaths_per_class: 一个字典结构，按类别存储路径。
        #data_to_iterate: 一个扁平的列表，包含了所有样本的信息，方便 __getitem__ 通过索引直接读取。
        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()

        self.transform_img = [
            transforms.Resize(resize), #按比例缩放短边
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(), # 将图片从 0~255 转成 0~1 的浮点数张量
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), # x = (x - mean) / std
        ] # 标准 ImageNet 预处理流程
        self.transform_img = transforms.Compose(self.transform_img)

        #定义对标签掩码（Ground Truth Mask）的预处理。Mask 不需要 Normalize，因为它只需要 0 或 1 来表示哪里是瑕疵。
        self.transform_mask = [
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)

        self.imagesize = (3, imagesize, imagesize)

    def __getitem__(self, idx):# 获取单个样本
        #根据索引 idx，从 data_to_iterate 列表中取出一个样本的信息。
        #包含：类别名、异常类型（如 "good" 或 "crack"）、图片路径、掩码路径。
        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
        image = PIL.Image.open(image_path).convert("RGB") # 读取图片并强制转为 RGB 模式（防止遇到灰度图导致通道数不对）
        image = self.transform_img(image) # 应用之前定义的 transform_img 进行预处理。

        if self.split == DatasetSplit.TEST and mask_path is not None:
            #如果是 测试集 (TEST) 且有掩码路径（说明是异常样本），则读取掩码图片并做处理。
            mask = PIL.Image.open(mask_path)
            mask = self.transform_mask(mask)
        else:
            #如果是 训练集 或者 正常的样本（没有掩码），则创建一个全零的 Tensor（表示整张图都是正常的，没有瑕疵）。
            mask = torch.zeros([1, *image.size()[1:]])

        #返回一个字典，包含模型训练或测试所需的所有数据。
        return {
            "image": image,
            "mask": mask,
            "classname": classname,
            "anomaly": anomaly,
            "is_anomaly": int(anomaly != "good"),
            "image_name": "/".join(image_path.split("/")[-4:]),
            "image_path": image_path,
        }

    def __len__(self):
        #返回数据集的总样本数。
        return len(self.data_to_iterate)

    def get_image_data(self): # 扫描数据文件, 负责遍历文件夹结构,MVTec 的文件夹结构通常是：dataset/bottle/train/good/xxx.png 或 dataset/bottle/test/broken_large/xxx.png。
        imgpaths_per_class = {}
        maskpaths_per_class = {}

        for classname in self.classnames_to_use:
            classpath = os.path.join(self.source, classname, self.split.value) # 构建当前 split 的路径（例如 .../bottle/train 或 .../bottle/test）。
            maskpath = os.path.join(self.source, classname, "ground_truth") # 构建真值掩码的路径（例如 .../bottle/ground_truth）。
            anomaly_types = os.listdir(classpath) # 列出该路径下的子文件夹（例如 good, broken_large, contamination 等）。

            imgpaths_per_class[classname] = {}
            maskpaths_per_class[classname] = {}

            for anomaly in anomaly_types: # 遍历每个异常类型子文件夹
                anomaly_path = os.path.join(classpath, anomaly)
                anomaly_files = sorted(os.listdir(anomaly_path)) #列出该子文件夹下的所有图片文件名，并排序。
                imgpaths_per_class[classname][anomaly] = [
                    os.path.join(anomaly_path, x) for x in anomaly_files
                ] #读取文件夹里所有的图片文件名，拼成完整路径，存入 imgpaths_per_class。

                if self.train_val_split < 1.0:#如果 train_val_split < 1.0（比如 0.9），说明要把原始训练数据切分
                    n_images = len(imgpaths_per_class[classname][anomaly])
                    train_val_split_idx = int(n_images * self.train_val_split)
                    if self.split == DatasetSplit.TRAIN: #如果当前是训练集，则只保留前 train_val_split_idx 个样本
                        imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
                            classname
                        ][anomaly][:train_val_split_idx]
                    elif self.split == DatasetSplit.VAL: #如果当前是验证集，则只保留后面的样本
                        imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
                            classname
                        ][anomaly][train_val_split_idx:]

                if self.split == DatasetSplit.TEST and anomaly != "good": # 如果是 TEST 集且是异常样本（anomaly != "good"），去 ground_truth 文件夹找对应的掩码文件。
                    anomaly_mask_path = os.path.join(maskpath, anomaly)
                    anomaly_mask_files = sorted(os.listdir(anomaly_mask_path))
                    maskpaths_per_class[classname][anomaly] = [
                        os.path.join(anomaly_mask_path, x) for x in anomaly_mask_files
                    ]
                else: # 否则（训练集或正常样本），掩码路径设为 None。
                    maskpaths_per_class[classname]["good"] = None

        #扁平化数据：
          # 上面得到的 imgpaths_per_class 是嵌套字典，不方便直接通过索引 idx 取数据。
          # 这一步通过三重循环，把每一张图片的信息打包成一个列表 data_tuple。
          # data_tuple 结构：[类别, 异常类型, 图片路径, 掩码路径]。
          # 最后将所有 tuple 放入 data_to_iterate 列表并返回。
        # Unrolls the data dictionary to an easy-to-iterate list.
        data_to_iterate = []
        for classname in sorted(imgpaths_per_class.keys()):
            for anomaly in sorted(imgpaths_per_class[classname].keys()):
                for i, image_path in enumerate(imgpaths_per_class[classname][anomaly]):
                    data_tuple = [classname, anomaly, image_path]
                    if self.split == DatasetSplit.TEST and anomaly != "good":
                        data_tuple.append(maskpaths_per_class[classname][anomaly][i])
                    else:
                        data_tuple.append(None)
                    data_to_iterate.append(data_tuple)

        return imgpaths_per_class, data_to_iterate
