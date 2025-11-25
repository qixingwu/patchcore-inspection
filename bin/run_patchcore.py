import contextlib
import logging
import os
import sys

import click
import numpy as np
import torch

import patchcore.backbones
import patchcore.common
import patchcore.metrics
import patchcore.patchcore
import patchcore.sampler
import patchcore.utils

LOGGER = logging.getLogger(__name__)

_DATASETS = {"mvtec": ["patchcore.datasets.mvtec", "MVTecDataset"]}


# 定义一个命令集合，下面可以再注册多个子命令
@click.group(chain=True) #跑完链中的每个子命令（收集返回值），最后把这些返回值一次性交给 result_callback
#  argument通常少量（1-2），必须参数，必须按顺序写；option通常较多（3个及以上），可选参数，不需要按顺序写
@click.argument("results_path", type=str) # 结果保存路径
@click.option("--gpu", type=int, default=[0], multiple=True, show_default=True) # 使用的GPU编号
@click.option("--seed", type=int, default=0, show_default=True) # 随机种子
@click.option("--log_group", type=str, default="group") # 日志组别
@click.option("--log_project", type=str, default="project") # 日志项目名称
@click.option("--save_segmentation_images", is_flag=True) # 是否保存分割图像
@click.option("--save_patchcore_model", is_flag=True) # 是否保存PatchCore模型
def main(**kwargs):
    pass


@main.result_callback() # 把子命令的返回值收集到 methods 里
def run(
    methods,
    results_path,# results_path = 'results'
    gpu, # gpu = (0,)
    seed, # seed = 0
    log_group,# log_group = 'IM224_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0'
    log_project, # log_project = 'MVTecAD_Results'
    save_segmentation_images,# save_segmentation_images = False
    save_patchcore_model, # save_patchcore_model = True
):
    methods = {key: item for (key, item) in methods}
    """
    methods = [
    ('get_patchcore', <function patch_core.<locals>.get_patchcore at ...>),
    ('get_sampler', <function sampler.<locals>.get_sampler at ...>),
    ('get_dataloaders', <function dataset.<locals>.get_dataloaders at ...>)
]
    """
    run_save_path = patchcore.utils.create_storage_folder(
        results_path, log_project, log_group, mode="iterate"
    ) # run_save_path = 'results/MVTecAD_Results/IM224_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0_2'

    list_of_dataloaders = methods["get_dataloaders"](seed)

    device = patchcore.utils.set_torch_device(gpu) # device = device(type='cuda',index=0),gpu = (0,) 
    # Device context here is specifically set and used later
    # because there was GPU memory-bleeding which I could only fix with
    # context managers.
    #这里必须显式创建一个 CUDA 设备上下文（with 块），否则 PatchCore 在多次推理中会出现 GPU 内存泄漏（memory bleeding），即用完的显存无法被自动释放。
    device_context = (
        torch.cuda.device("cuda:{}".format(device.index))
        if "cuda" in device.type.lower()
        else contextlib.suppress() # Python 标准库里提供的 一个“空的上下文管理器”
    )

    result_collect = []
    # 遍历数据集: 开始循环，依次处理每一个子数据集（例如 bottle, cable 等）。result_collect 用于收集每个数据集的评估指标。
    for dataloader_count, dataloaders in enumerate(list_of_dataloaders):
        LOGGER.info(
            "Evaluating dataset [{}] ({}/{})...".format(
                dataloaders["training"].name,
                dataloader_count + 1,
                len(list_of_dataloaders),
            )
        )
        # 固定种子: 在处理每个数据集开始前重新固定随机种子，确保每个数据集的训练过程都是独立且可复现的。
        #Python 中,只要对象不是空的、不是 0、不是 None，就会被当作 True。
        patchcore.utils.fix_seeds(seed, device)

        dataset_name = dataloaders["training"].name # dataset_name = 'mvtec_bottle'

        with device_context:
            torch.cuda.empty_cache() # 释放之前可能占用的显存
            imagesize = dataloaders["training"].dataset.imagesize # imagesize = (3, 224, 224)
            sampler = methods["get_sampler"](
                device,
            )
            PatchCore_list = methods["get_patchcore"](imagesize, sampler, device)
            if len(PatchCore_list) > 1:
                LOGGER.info(
                    "Utilizing PatchCore Ensemble (N={}).".format(len(PatchCore_list))
                )
            for i, PatchCore in enumerate(PatchCore_list):
                torch.cuda.empty_cache()
                if PatchCore.backbone.seed is not None:
                    patchcore.utils.fix_seeds(PatchCore.backbone.seed, device)
                LOGGER.info(
                    "Training models ({}/{})".format(i + 1, len(PatchCore_list))
                )
                torch.cuda.empty_cache()
                PatchCore.fit(dataloaders["training"])

            torch.cuda.empty_cache()
            aggregator = {"scores": [], "segmentations": []}
            for i, PatchCore in enumerate(PatchCore_list):
                torch.cuda.empty_cache()
                LOGGER.info(
                    "Embedding test data with models ({}/{})".format(
                        i + 1, len(PatchCore_list)
                    )
                )
                scores, segmentations, labels_gt, masks_gt = PatchCore.predict(
                    dataloaders["testing"]
                )
                aggregator["scores"].append(scores)
                aggregator["segmentations"].append(segmentations)

            scores = np.array(aggregator["scores"])
            min_scores = scores.min(axis=-1).reshape(-1, 1)
            max_scores = scores.max(axis=-1).reshape(-1, 1)
            scores = (scores - min_scores) / (max_scores - min_scores)
            scores = np.mean(scores, axis=0)

            segmentations = np.array(aggregator["segmentations"])
            min_scores = (
                segmentations.reshape(len(segmentations), -1)
                .min(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            max_scores = (
                segmentations.reshape(len(segmentations), -1)
                .max(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            segmentations = (segmentations - min_scores) / (max_scores - min_scores)
            segmentations = np.mean(segmentations, axis=0)

            anomaly_labels = [
                x[1] != "good" for x in dataloaders["testing"].dataset.data_to_iterate
            ]

            # (Optional) Plot example images.
            if save_segmentation_images:
                image_paths = [
                    x[2] for x in dataloaders["testing"].dataset.data_to_iterate
                ]
                mask_paths = [
                    x[3] for x in dataloaders["testing"].dataset.data_to_iterate
                ]

                def image_transform(image):
                    in_std = np.array(
                        dataloaders["testing"].dataset.transform_std
                    ).reshape(-1, 1, 1)
                    in_mean = np.array(
                        dataloaders["testing"].dataset.transform_mean
                    ).reshape(-1, 1, 1)
                    image = dataloaders["testing"].dataset.transform_img(image)
                    return np.clip(
                        (image.numpy() * in_std + in_mean) * 255, 0, 255
                    ).astype(np.uint8)

                def mask_transform(mask):
                    return dataloaders["testing"].dataset.transform_mask(mask).numpy()

                image_save_path = os.path.join(
                    run_save_path, "segmentation_images", dataset_name
                )
                os.makedirs(image_save_path, exist_ok=True)
                patchcore.utils.plot_segmentation_images(
                    image_save_path,
                    image_paths,
                    segmentations,
                    scores,
                    mask_paths,
                    image_transform=image_transform,
                    mask_transform=mask_transform,
                )

            LOGGER.info("Computing evaluation metrics.")
            auroc = patchcore.metrics.compute_imagewise_retrieval_metrics(
                scores, anomaly_labels
            )["auroc"]

            # Compute PRO score & PW Auroc for all images
            pixel_scores = patchcore.metrics.compute_pixelwise_retrieval_metrics(
                segmentations, masks_gt
            )
            full_pixel_auroc = pixel_scores["auroc"]

            # Compute PRO score & PW Auroc only images with anomalies
            sel_idxs = []
            for i in range(len(masks_gt)):
                if np.sum(masks_gt[i]) > 0:
                    sel_idxs.append(i)
            pixel_scores = patchcore.metrics.compute_pixelwise_retrieval_metrics(
                [segmentations[i] for i in sel_idxs],
                [masks_gt[i] for i in sel_idxs],
            )
            anomaly_pixel_auroc = pixel_scores["auroc"]

            result_collect.append(
                {
                    "dataset_name": dataset_name,
                    "instance_auroc": auroc,
                    "full_pixel_auroc": full_pixel_auroc,
                    "anomaly_pixel_auroc": anomaly_pixel_auroc,
                }
            )

            for key, item in result_collect[-1].items():
                if key != "dataset_name":
                    LOGGER.info("{0}: {1:3.3f}".format(key, item))

            # (Optional) Store PatchCore model for later re-use.
            # SAVE all patchcores only if mean_threshold is passed?
            if save_patchcore_model:
                patchcore_save_path = os.path.join(
                    run_save_path, "models", dataset_name
                )
                os.makedirs(patchcore_save_path, exist_ok=True)
                for i, PatchCore in enumerate(PatchCore_list):
                    prepend = (
                        "Ensemble-{}-{}_".format(i + 1, len(PatchCore_list))
                        if len(PatchCore_list) > 1
                        else ""
                    )
                    PatchCore.save_to_path(patchcore_save_path, prepend)

        LOGGER.info("\n\n-----\n")

    # Store all results and mean scores to a csv-file.
    result_metric_names = list(result_collect[-1].keys())[1:]
    result_dataset_names = [results["dataset_name"] for results in result_collect]
    result_scores = [list(results.values())[1:] for results in result_collect]
    patchcore.utils.compute_and_store_final_results(
        run_save_path,
        result_scores,
        column_names=result_metric_names,
        row_names=result_dataset_names,
    )

# 当用户在命令行输入 patch_core 子命令时，调用此函数
@main.command("patch_core") 
# Pretraining-specific parameters.
@click.option("--backbone_names", "-b", type=str, multiple=True, default=[]) # 使用的backbone名称
@click.option("--layers_to_extract_from", "-le", type=str, multiple=True, default=[]) # 从哪些层提取特征
# Parameters for Glue-code (to merge different parts of the pipeline.
@click.option("--pretrain_embed_dimension", type=int, default=1024) # 预训练嵌入维度（即 backbone 输出的通道数, 一般不需要改）,把高维特征投影到一个更低维的空间
@click.option("--target_embed_dimension", type=int, default=1024) # 目标嵌入维度,将预训练模型输出的特征（pretrain embedding）通过一个线性映射（或降维层）投影到的目标特征维度
@click.option("--preprocessing", type=click.Choice(["mean", "conv"]), default="mean") # 预处理方法，如均值或卷积
@click.option("--aggregation", type=click.Choice(["mean", "mlp"]), default="mean") # 聚合方法，如均值或多层感知机
# Nearest-Neighbour Anomaly Scorer parameters.
@click.option("--anomaly_scorer_num_nn", type=int, default=5) # 最近邻数量
# Patch-parameters.
@click.option("--patchsize", type=int, default=3) # 补丁大小
@click.option("--patchscore", type=str, default="max") # 补丁评分方法
@click.option("--patchoverlap", type=float, default=0.0) # 补丁重叠比例
@click.option("--patchsize_aggregate", "-pa", type=int, multiple=True, default=[]) # 聚合补丁大小
# NN on GPU.
@click.option("--faiss_on_gpu", is_flag=True) # 是否在GPU上使用Faiss
@click.option("--faiss_num_workers", type=int, default=8) # Faiss使用的工作线程数量
def patch_core(
    backbone_names,
    layers_to_extract_from,
    pretrain_embed_dimension,
    target_embed_dimension,
    preprocessing,
    aggregation,
    patchsize,
    patchscore,
    patchoverlap,
    anomaly_scorer_num_nn,
    patchsize_aggregate,
    faiss_on_gpu,
    faiss_num_workers,
):
    backbone_names = list(backbone_names) #Click 的 multiple=True 选项会把参数当元组，这里转成 list，后续好处理。 backbone_names = ('wideresnet50',)
    if len(backbone_names) > 1:
        layers_to_extract_from_coll = [[] for _ in range(len(backbone_names))]
        for layer in layers_to_extract_from:
            idx = int(layer.split(".")[0])
            layer = ".".join(layer.split(".")[1:])
            layers_to_extract_from_coll[idx].append(layer)
    else:
        layers_to_extract_from_coll = [layers_to_extract_from] #layers_to_extract_from = ('layer2', 'layer3')

    def get_patchcore(input_shape, sampler, device): 
        #get_patchcore 这个“内置函数”在定义时就把外层的变量一起“打包”进去了，以后调用它的时候会直接用你在 patch_core() 里处理好的这些值。
        loaded_patchcores = []
        for backbone_name, layers_to_extract_from in zip(
            backbone_names, layers_to_extract_from_coll
        ):
            backbone_seed = None
            if ".seed-" in backbone_name:
                backbone_name, backbone_seed = backbone_name.split(".seed-")[0], int(
                    backbone_name.split("-")[-1]
                )
            backbone = patchcore.backbones.load(backbone_name)
            backbone.name, backbone.seed = backbone_name, backbone_seed # 在 Python 中，大多数对象（尤其是自定义类的实例）都是动态可扩展的

            nn_method = patchcore.common.FaissNN(faiss_on_gpu, faiss_num_workers) #构造基于 Faiss 的最近邻搜索器，支持 GPU/多线程。

            patchcore_instance = patchcore.patchcore.PatchCore(device)
            patchcore_instance.load(
                backbone=backbone,
                layers_to_extract_from=layers_to_extract_from,
                device=device,
                input_shape=input_shape,
                pretrain_embed_dimension=pretrain_embed_dimension,
                target_embed_dimension=target_embed_dimension,
                patchsize=patchsize,
                featuresampler=sampler,
                anomaly_scorer_num_nn=anomaly_scorer_num_nn,
                nn_method=nn_method,
            )
            loaded_patchcores.append(patchcore_instance)
        return loaded_patchcores

    return ("get_patchcore", get_patchcore)


@main.command("sampler")
@click.argument("name", type=str)
@click.option("--percentage", "-p", type=float, default=0.1, show_default=True)
def sampler(name, percentage):
    def get_sampler(device):
        if name == "identity":
            return patchcore.sampler.IdentitySampler()
        elif name == "greedy_coreset":
            return patchcore.sampler.GreedyCoresetSampler(percentage, device)
        elif name == "approx_greedy_coreset":
            return patchcore.sampler.ApproximateGreedyCoresetSampler(percentage, device)

    return ("get_sampler", get_sampler)


@main.command("dataset")
@click.argument("name", type=str)
@click.argument("data_path", type=click.Path(exists=True, file_okay=False))
@click.option("--subdatasets", "-d", multiple=True, type=str, required=True)
@click.option("--train_val_split", type=float, default=1, show_default=True)
@click.option("--batch_size", default=2, type=int, show_default=True)
@click.option("--num_workers", default=8, type=int, show_default=True)
@click.option("--resize", default=256, type=int, show_default=True)
@click.option("--imagesize", default=224, type=int, show_default=True)
@click.option("--augment", is_flag=True)
def dataset(
    name, # name = 'mvtec'
    data_path, # data_path = '/mnt/e/Dataset/mvtec'
    subdatasets, # subdatasets = ('bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper')
    train_val_split, # train_val_split = 1.0
    batch_size, # batch_size = 2
    resize, # resize = 256
    imagesize, # imagesize = 224
    num_workers, # num_workers = 8
    augment, #  augment = False
):
    dataset_info = _DATASETS[name] # dataset_info = ['patchcore.datasets.mvtec', 'MVTecDataset'],_DATASETS = {'mvtec': ['patchcore.datasets.mvtec', 'MVTecDataset']},name = 'mvtec'
    dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]]) #dataset_library = <module 'patchcore.datasets.mvtec'> 动态地加载模块

    def get_dataloaders(seed):
        """
        根据给定的种子（seed）和全局配置，为每一个子数据集（subdataset）分别创建训练集、测试集和验证集的数据加载器。
        它接受一个参数 seed（随机种子），通常用于保证数据分割和增强的可复现性。
        """
        dataloaders = []
        # subdatasets 是一个在函数外部定义的列表（全局变量），里面包含不同的类别名称（例如 ['bottle', 'cable', 'capsule']）。代码会为列表中的每一个类别分别创建一套数据加载器。
        for subdataset in subdatasets:
            #等价于DatasetClass = getattr(dataset_library, dataset_info[1])
            train_dataset = dataset_library.__dict__[dataset_info[1]](
                data_path, # data_path = '/mnt/e/Dataset/mvtec'
                classname=subdataset, # classname = 'bottle'（当前循环的子数据集名称）
                resize=resize, # resize = 256
                train_val_split=train_val_split, # train_val_split = 1.0
                imagesize=imagesize, # imagesize = 224
                split=dataset_library.DatasetSplit.TRAIN, # split = 'train'（表示这是训练集）
                seed=seed, # seed = 0
                augment=augment, # augment = False
            )

            test_dataset = dataset_library.__dict__[dataset_info[1]](
                data_path,
                classname=subdataset,
                resize=resize,
                imagesize=imagesize,
                split=dataset_library.DatasetSplit.TEST,
                seed=seed,
            )

            train_dataloader = torch.utils.data.DataLoader(
                train_dataset, # train_dataset = <patchcore.datasets.mvtec.MVTecDataset object at 0x7ec6f807a5b0>
                batch_size=batch_size, # batch_size = 2
                shuffle=False, # 训练集通常会打乱顺序以增加数据多样性，但这里设为 False 可能是因为某些特殊需求
                num_workers=num_workers,
                pin_memory=True, # 将数据锁在内存中，加快数据从 CPU 传输到 GPU 的速度。
            )

            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

            train_dataloader.name = name
            if subdataset is not None:
                train_dataloader.name += "_" + subdataset

            if train_val_split < 1:
                val_dataset = dataset_library.__dict__[dataset_info[1]](
                    data_path,
                    classname=subdataset,
                    resize=resize,
                    train_val_split=train_val_split,
                    imagesize=imagesize,
                    split=dataset_library.DatasetSplit.VAL,
                    seed=seed,
                )

                val_dataloader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                )
            else:
                val_dataloader = None
            dataloader_dict = {
                "training": train_dataloader,
                "validation": val_dataloader,
                "testing": test_dataloader,
            }

            dataloaders.append(dataloader_dict)
        return dataloaders

    return ("get_dataloaders", get_dataloaders)


if __name__ == "__main__":

    # # 设置日志系统的基础配置，让程序里的 logging.info(...)、logging.warning(...) 等语句能在控制台输出。
    # logging.basicConfig(level=logging.INFO)

    # ===== 交互选择日志去向 =====
    try:
        choice = input("日志输出到哪里？[yes=控制台, no=文件, 回车=控制台+文件] ").strip().lower() 
    except EOFError:
        # 有些非交互环境（比如某些 IDE 配置）拿不到输入，默认双写
        choice = ""

    # 日志文件名（你也可以改成放在 results 目录，简单起见先放当前目录）
    log_file = os.path.abspath("./log/patchcore_run.log")

    # 拿到 root logger 并清空默认 handler，避免重复打印
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # 统一的日志格式
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(name)s: %(message)s", datefmt="%H:%M:%S"
    )

    # 控制台/文件 handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(fmt)

    # 根据选择添加 handler
    if choice in ("yes", "y"):
        logger.addHandler(console_handler)
        print("✅ 日志仅输出到控制台。")
    elif choice in ("no", "n"):
        logger.addHandler(file_handler)
        print(f"✅ 日志仅输出到文件：{log_file}")
    else:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        print(f"✅ 日志将同时输出到控制台和文件：{log_file}")

    # 下面照常启动程序
    # --------------------------

    # 打印命令行参数
    LOGGER.info("Command line arguments: {}".format(" ".join(sys.argv)))
    main()
