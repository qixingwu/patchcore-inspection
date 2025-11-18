"""PatchCore and PatchCore detection methods."""
import logging
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import tqdm

import patchcore
import patchcore.backbones
import patchcore.common
import patchcore.sampler

LOGGER = logging.getLogger(__name__)


class PatchCore(torch.nn.Module):
    """
    PatchCore 这个类，把 PatchCore 整个“异常检测流程”都封装起来了：
    从特征提取 → 切 patch → 建记忆库（memory bank）→ KNN 打分 → 生成图像级分数和像素级掩码 → 支持保存和加载。
    """
    def __init__(self, device):
        """PatchCore anomaly detection class."""
        super(PatchCore, self).__init__()
        self.device = device

    def load(
        self,
        backbone,
        layers_to_extract_from,
        device,
        input_shape,
        pretrain_embed_dimension,
        target_embed_dimension,
        patchsize=3,
        patchstride=1,
        anomaly_score_num_nn=1,
        featuresampler=patchcore.sampler.IdentitySampler(),
        nn_method=patchcore.common.FaissNN(False, 4),
        **kwargs,
    ):
        """
        配好模型结构
        """
        self.backbone = backbone.to(device)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape

        self.device = device
        self.patch_maker = PatchMaker(patchsize, stride=patchstride)

        self.forward_modules = torch.nn.ModuleDict({})

        feature_aggregator = patchcore.common.NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device
        )
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        self.forward_modules["feature_aggregator"] = feature_aggregator

        preprocessing = patchcore.common.Preprocessing(
            feature_dimensions, pretrain_embed_dimension
        )
        self.forward_modules["preprocessing"] = preprocessing

        self.target_embed_dimension = target_embed_dimension
        preadapt_aggregator = patchcore.common.Aggregator(
            target_dim=target_embed_dimension
        ) 

        _ = preadapt_aggregator.to(self.device)

        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        self.anomaly_scorer = patchcore.common.NearestNeighbourScorer(
            n_nearest_neighbours=anomaly_score_num_nn, nn_method=nn_method
        )

        self.anomaly_segmentor = patchcore.common.RescaleSegmentor(
            device=self.device, target_size=input_shape[-2:]
        )

        self.featuresampler = featuresampler

    def embed(self, data):
        """
        它是“外部接口”，让用户能用一种简单统一的方式给 PatchCore 提供数据，然后得到 embedding（特征向量）。
        PatchCore 的训练（fit）和推理（predict）都需要“把图像 → 转成 patch-level embedding”。
        而 embedding 流程非常复杂（用了 _embed），所以对外提供一个简单接口是必要的。
        它解决了三个重要问题：
        ✔ 1. 让使用者不用关心数据类型是 Tensor 还是 DataLoader
        ✔ 2. 加强“数据清洗”能力：自动取出 image 字段
        ✔ 3. 保证 embedding 流程统一 —— 外面永远调用 _embed
        """
        if isinstance(data, torch.utils.data.DataLoader):
            features = []
            for image in data:
                if isinstance(image, dict):
                    image = image["image"] # 自动取出 image 字段
                with torch.no_grad():
                    input_image = image.to(torch.float).to(self.device) # 转成 float Tensor 并搬到正确设备上
                    features.append(self._embed(input_image))
            return features
        return self._embed(data)

    def _embed(self, images, detach=True, provide_patch_shapes=False):
        """Returns feature embeddings for images.
        _embed 是 PatchCore 的核心特征流水线：
        把一批图片 images → 通过 backbone 抽特征 → 各层切成 patch → 把多层特征对齐到同一patch 网格 → 做预处理和降维 → 得到统一的 patch-level embedding（给 KNN 用）。
        """

        def _detach(features):
            """
            如果 detach=True,对每个 tensor：.detach()（断开梯度）,.cpu()（丢回 CPU）,.numpy()（变成 numpy）
            """
            if detach:
                return [x.detach().cpu().numpy() for x in features]
            return features

        _ = self.forward_modules["feature_aggregator"].eval()
        with torch.no_grad():
            features = self.forward_modules["feature_aggregator"](images)

        features = [features[layer] for layer in self.layers_to_extract_from]

        features = [
            self.patch_maker.patchify(x, return_spatial_info=True) for x in features
        ] # 对每一层的 feature map（x）调用 PatchMaker.patchify，把它切成 patches，同时返回 patch 的空间布局信息。
        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0] # PatchCore 要把所有层的 patch 网格尺寸对齐到同一个大小，而这个大小用第一层的 patch grid 当作标准。

        for i in range(1, len(features)):
            _features = features[i]        # shape: [B, N_i, C_i, P, P]
            patch_dims = patch_shapes[i]   # patch_dims = [H_i, W_i] 且 N_i = H_i * W_i

            # TODO(pgehler): Add comments
            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            ) # [B, N_i, C_i, P, P] -> [B, H_i, W_i, C_i, P, P]
            _features = _features.permute(0, -3, -2, -1, 1, 2) # [B, H_i, W_i, C_i, P, P] -> [B, C_i, P, P, H_i, W_i]
            perm_base_shape = _features.shape #记录当前形状，用于后面 reshape 回来
            _features = _features.reshape(-1, *_features.shape[-2:]) # [B*C_i*P*P, H_i, W_i],把前面所有维度合并，只保留 H_i, W_i 准备插值
            _features = F.interpolate(
                _features.unsqueeze(1), #在前面加一个 channel 维度，符合 F.interpolate 的输入格式
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            ) # 把 H_i, W_i 插值到参考尺寸 ref_num_patches
            _features = _features.squeeze(1) # 把多余的 channel 维度挤掉
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            ) # 把前面合并的维度 reshape 回去
            _features = _features.permute(0, -2, -1, 1, 2, 3) #[B, C_i, P, P, H_ref, W_ref] -> [B, H_ref, W_ref, C_i, P, P]
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:]) #再次把二维 patch 网格拉平成一个维度,得到 [B, N_ref, C_i, P, P]
            features[i] = _features #实现了[B, H_i * W_i, C_i, P, P] -> [B, H_ref * W_ref, C_i, P, P]
        features = [x.reshape(-1, *x.shape[-3:]) for x in features] #循环结束后，对所有层 flatten 成 “所有 patch × 通道 × patch内部”，[B, H_ref * W_ref, C_i, P, P] -> [B * H_ref * W_ref, C_i, P, P]

        # As different feature backbones & patching provide differently
        # sized features, these are brought into the correct form here.
        features = self.forward_modules["preprocessing"](features) # Preprocessing 不是对“每个 patch 降维”，而是对“每个层的所有 patch 特征做线性映射 + 求平均”，最终得到每层一个 embedding 向量（D_pretrain）。
        features = self.forward_modules["preadapt_aggregator"](features) # 把多层特征整合成一个统一长度的向量，[B * H_ref * W_ref, D_pretrain] -> [B * H_ref * W_ref, D_target]

        if provide_patch_shapes:
            return _detach(features), patch_shapes
        return _detach(features)

    def fit(self, training_data):
        """PatchCore training.

        This function computes the embeddings of the training data and fills the
        memory bank of SPADE.
        """
        self._fill_memory_bank(training_data)

    def _fill_memory_bank(self, input_data):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()

        def _image_to_features(input_image):
            with torch.no_grad():
                input_image = input_image.to(torch.float).to(self.device)
                return self._embed(input_image)

        features = []
        with tqdm.tqdm(
            input_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    image = image["image"]
                features.append(_image_to_features(image))

        features = np.concatenate(features, axis=0)
        features = self.featuresampler.run(features)

        self.anomaly_scorer.fit(detection_features=[features])

    def predict(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader(data)
        return self._predict(data)

    def _predict_dataloader(self, dataloader):
        """This function provides anomaly scores/maps for full dataloaders."""
        _ = self.forward_modules.eval()

        scores = []
        masks = []
        labels_gt = []
        masks_gt = []
        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    labels_gt.extend(image["is_anomaly"].numpy().tolist())
                    masks_gt.extend(image["mask"].numpy().tolist())
                    image = image["image"]
                _scores, _masks = self._predict(image)
                for score, mask in zip(_scores, _masks):
                    scores.append(score)
                    masks.append(mask)
        return scores, masks, labels_gt, masks_gt

    def _predict(self, images):
        """Infer score and mask for a batch of images."""
        images = images.to(torch.float).to(self.device)
        _ = self.forward_modules.eval()

        batchsize = images.shape[0]
        with torch.no_grad():
            features, patch_shapes = self._embed(images, provide_patch_shapes=True)
            features = np.asarray(features)

            patch_scores = image_scores = self.anomaly_scorer.predict([features])[0]
            image_scores = self.patch_maker.unpatch_scores(
                image_scores, batchsize=batchsize
            )
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            image_scores = self.patch_maker.score(image_scores)

            patch_scores = self.patch_maker.unpatch_scores(
                patch_scores, batchsize=batchsize
            )
            scales = patch_shapes[0]
            patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])

            masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores)

        return [score for score in image_scores], [mask for mask in masks]

    @staticmethod
    def _params_file(filepath, prepend=""):
        return os.path.join(filepath, prepend + "patchcore_params.pkl")

    def save_to_path(self, save_path: str, prepend: str = "") -> None:
        LOGGER.info("Saving PatchCore data.")
        self.anomaly_scorer.save(
            save_path, save_features_separately=False, prepend=prepend
        )
        patchcore_params = {
            "backbone.name": self.backbone.name,
            "layers_to_extract_from": self.layers_to_extract_from,
            "input_shape": self.input_shape,
            "pretrain_embed_dimension": self.forward_modules[
                "preprocessing"
            ].output_dim,
            "target_embed_dimension": self.forward_modules[
                "preadapt_aggregator"
            ].target_dim,
            "patchsize": self.patch_maker.patchsize,
            "patchstride": self.patch_maker.stride,
            "anomaly_scorer_num_nn": self.anomaly_scorer.n_nearest_neighbours,
        }
        with open(self._params_file(save_path, prepend), "wb") as save_file:
            pickle.dump(patchcore_params, save_file, pickle.HIGHEST_PROTOCOL)

    def load_from_path(
        self,
        load_path: str,
        device: torch.device,
        nn_method: patchcore.common.FaissNN(False, 4),
        prepend: str = "",
    ) -> None:
        LOGGER.info("Loading and initializing PatchCore.")
        with open(self._params_file(load_path, prepend), "rb") as load_file:
            patchcore_params = pickle.load(load_file)
        patchcore_params["backbone"] = patchcore.backbones.load(
            patchcore_params["backbone.name"]
        )
        patchcore_params["backbone"].name = patchcore_params["backbone.name"]
        del patchcore_params["backbone.name"]
        self.load(**patchcore_params, device=device, nn_method=nn_method)

        self.anomaly_scorer.load(load_path, prepend)


# Image handling classes.
class PatchMaker:
    """
    把一张（或多张）特征图切成很多小块 patch，并对这些 patch 的结果做一些整合。
    """
    def __init__(self, patchsize, stride=None):
        self.patchsize = patchsize
        self.stride = stride

    def patchify(self, features, return_spatial_info=False): # [B, C, H, W] -> [B, N_patches, C, P, P]
        """把特征图切成很多小 patchConvert a tensor into a tensor of respective patches.
        Args:
            x: [torch.Tensor, bs x c x w x h]
        Returns:
            x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
            patchsize]
        """
        padding = int((self.patchsize - 1) / 2) # 为了让每个patch都能中心对齐
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        ) # 拿一个滑动窗口，在特征图上滑动，把每一次窗口覆盖的区域（patch）拉平成一列，最终得到所有 patch 的集合。 dilation 决定 patch 内的点是不是连续取的。
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]: # 输入的特征图在高和宽方向各能切出多少个 patch。
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        ) # 把 [B, C*P*P, N_patches] reshape 成 [B, C, P, P, N_patches]，即把第二维展开成 3 个维度。
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3) # 把 patch 这一维（N_patches）从最后一维移到第二维

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batchsize):
        """
        把拍扁过的一长条的 patch 分数，重新按 batch 组织好。
        当你把所有 batch 的 patch 分数堆成了一长条后，这个函数帮你按 batch 把它重新分组。
        """
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        """
        把高维分数张量反复做 max pooling，压缩成低维/标量分数
        """
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x) # 内部统一用 PyTorch 来算，最后再按需要转回 numpy。
        while x.ndim > 1:
            x = torch.max(x, dim=-1).values
        if was_numpy:
            return x.numpy()
        return x
