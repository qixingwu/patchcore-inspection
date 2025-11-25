# 抽象基类库 abc, 它的作用是让你能在 Python 中创建“抽象类”和“抽象方法”。
# 抽象类是一种只能被继承、不能被直接创建对象的类。
# 使用 @abc.abstractmethod 修饰的方法，子类必须实现，否则会报错。
# 目的：为了让所有采样器（Greedy、Approximate、Random）都有统一接口
import abc 
# Union 的作用：表示“可以是多种类型之一”
# Union[A, B] 代表类型可以是 A 或 B
from typing import Union

import numpy as np
import torch
import tqdm


class IdentitySampler: # 不进行任何下采样
    def run(
        self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        return features


class BaseSampler(abc.ABC):
    """
    这是一个父类，为所有采样器定义了标准接口和通用工具。
    要让 Python 识别它是抽象类，必须继承：abc.ABC
    """
    def __init__(self, percentage: float):
        if not 0 < percentage < 1:
            raise ValueError("Percentage value not in (0, 1).")
        self.percentage = percentage

    @abc.abstractmethod
    def run(
        self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        pass

    def _store_type(self, features: Union[torch.Tensor, np.ndarray]) -> None:
        self.features_is_numpy = isinstance(features, np.ndarray)
        if not self.features_is_numpy:
            self.features_device = features.device

    def _restore_type(self, features: torch.Tensor) -> Union[torch.Tensor, np.ndarray]:
        if self.features_is_numpy:
            return features.cpu().numpy()
        return features.to(self.features_device)


class GreedyCoresetSampler(BaseSampler):
    """
    贪婪核心集采样
    """
    def __init__(
        self,
        percentage: float,
        device: torch.device,
        # 在进行 Greedy Coreset 时，你需要计算：所有样本之间的距离或每个样本到选择点的距离
        # 这些运算的复杂度跟维度 D 有关，如果你的特征是：D = 512（常见）D = 2048（ResNet 特征）D = 4096（VGG 特征）
        # 维度太高会导致：计算距离非常慢 内存占用巨大（尤其是 N×N 距离矩阵）为了提高速度、降低内存，就需要降维。
        dimension_to_project_features_to=128, # 投影维度，默认是128维
    ):
        """Greedy Coreset sampling base class."""
        super().__init__(percentage)

        self.device = device
        self.dimension_to_project_features_to = dimension_to_project_features_to

    def _reduce_features(self, features): # 进行降维
        if features.shape[1] == self.dimension_to_project_features_to:
            return features
        # 注意：这里没有训练这个层，而是使用随机初始化的权重。 根据 Johnson-Lindenstrauss 引理，随机投影能在低维空间保持高维空间中的距离关系。
        mapper = torch.nn.Linear(
            features.shape[1], self.dimension_to_project_features_to, bias=False
        )
        _ = mapper.to(self.device)
        features = features.to(self.device)
        return mapper(features)

    def run(
        self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """Subsamples features using Greedy Coreset.

        Args:
            features: [N x D]
        """
        if self.percentage == 1:
            return features
        self._store_type(features)
        # 如果输入是 NumPy 数组，就把它转换成 PyTorch Tensor
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features)
        reduced_features = self._reduce_features(features) # 首先降维到128
        sample_indices = self._compute_greedy_coreset_indices(reduced_features) # 调用 _compute_greedy_coreset_indices 算出应该保留哪些索引。
        features = features[sample_indices]
        return self._restore_type(features)

    @staticmethod
    def _compute_batchwise_differences(
        matrix_a: torch.Tensor, matrix_b: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes batchwise Euclidean distances using PyTorch
        这是一个利用矩阵运算加速计算欧氏距离的数学技巧。公式为：$$\|A - B\|^2 = A^2 + B^2 - 2AB$$
        假设：matrix_a.shape = [N, D]  matrix_b.shape = [M, D]
        最终我们想得到：[N, M] 的距离矩阵
        """
        a_times_a = matrix_a.unsqueeze(1).bmm(matrix_a.unsqueeze(2)).reshape(-1, 1)
        b_times_b = matrix_b.unsqueeze(1).bmm(matrix_b.unsqueeze(2)).reshape(1, -1)
        a_times_b = matrix_a.mm(matrix_b.T)
        # clamp(0, None) 限制最小值为0（防止浮点误差导致负数）
        return (-2 * a_times_b + a_times_a + b_times_b).clamp(0, None).sqrt()

    def _compute_greedy_coreset_indices(self, features: torch.Tensor) -> np.ndarray:
        """Runs iterative greedy coreset selection.

        Args:
            features: [NxD] input feature bank to sample.

        这个算法的直观理解是：
          随便找一个点（或者离中心最远的点）作为第一个基站。
          在地图上找离这个基站最远的地方，建立第二个基站。
          现在有两个基站了，对地图上每个人来说，离他们最近的基站距离变了。
          再在地图上找一个“离最近基站还是最远”的地方，建立第三个基站。
          重复直到基站数量足够。
        """
        # 计算所有点对所有点的距离矩阵 (N x N)。注意：这非常消耗显存。
        distance_matrix = self._compute_batchwise_differences(features, features)
        # 初始化一个距离向量
        # 如果一个点对应的行向量范数很大，说明它和其他所有点的距离总和很大。
        # 这意味着这个点是一个“离群点”或者“边缘点”。
        # 在贪婪算法里，如果你不知道从哪里开始，选一个最“边缘”的点作为起始点通常是不错的策略（这是一种启发式 Heuristic）。
        coreset_anchor_distances = torch.norm(distance_matrix, dim=1) # 每个样本到所有样本距离的“综合大小”（L2 norm），仅仅是为了选出第 1 个点。

        coreset_indices = []
        num_coreset_samples = int(len(features) * self.percentage)

        for _ in range(num_coreset_samples):
            # 我们总是选择距离当前核心集最远的那个点。这正是“贪婪”的体现——每次都选那个当前“最孤立”、“最独特”的样本，以最大化覆盖范围。
            select_idx = torch.argmax(coreset_anchor_distances).item()
            # 严格来说，这段代码不会重复选到同一个 select_idx，但它并不是通过显式“去重”做到的，而是通过算法的“距离更新规则”自动实现的。
            # 当某个点被选进 coreset 之后，会发生两件事：
            # 1. 这个点本身不会再被选中（因为它到自己的距离是0，不可能是最大的）。
            # 2. 之后每轮都用 torch.argmax(coreset_anchor_distances) 选“最远的点”时，这个点的距离值已经被更新为0（因为它到核心集的距离是0），所以不会再被选中。
            coreset_indices.append(select_idx)

            # 先用 cat 把旧的距离和新的距离凑成 [N, 2]，再对第二维取 min，就等效于 “对每个点，保留它到已有 coreset 和新点的距离里更小的那个
            coreset_select_distance = distance_matrix[
                :, select_idx : select_idx + 1  # noqa E203
            ]
            coreset_anchor_distances = torch.cat(
                [coreset_anchor_distances.unsqueeze(-1), coreset_select_distance], dim=1
            )
            coreset_anchor_distances = torch.min(coreset_anchor_distances, dim=1).values

        return np.array(coreset_indices)


class ApproximateGreedyCoresetSampler(GreedyCoresetSampler):
    """
    为了解决上面 GreedyCoresetSampler 计算 N x N 矩阵显存爆炸的问题，这个类做了一个近似版本。
    """
    def __init__(
        self,
        percentage: float,
        device: torch.device,
        number_of_starting_points: int = 10,
        dimension_to_project_features_to: int = 128,
    ):
        """Approximate Greedy Coreset sampling base class."""
        self.number_of_starting_points = number_of_starting_points
        super().__init__(percentage, device, dimension_to_project_features_to)

    def _compute_greedy_coreset_indices(self, features: torch.Tensor) -> np.ndarray:
        """Runs approximate iterative greedy coreset selection.

        This greedy coreset implementation does not require computation of the
        full N x N distance matrix and thus requires a lot less memory, however
        at the cost of increased sampling times.

        Args:
            features: [NxD] input feature bank to sample.
        """
        number_of_starting_points = np.clip(
            self.number_of_starting_points, None, len(features)
        )
        start_points = np.random.choice(
            len(features), number_of_starting_points, replace=False
        ).tolist()

        approximate_distance_matrix = self._compute_batchwise_differences(
            features, features[start_points]
        )
        approximate_coreset_anchor_distances = torch.mean(
            approximate_distance_matrix, axis=-1
        ).reshape(-1, 1)
        coreset_indices = []
        num_coreset_samples = int(len(features) * self.percentage)

        with torch.no_grad():
            for _ in tqdm.tqdm(range(num_coreset_samples), desc="Subsampling..."):
                select_idx = torch.argmax(approximate_coreset_anchor_distances).item()
                coreset_indices.append(select_idx)
                coreset_select_distance = self._compute_batchwise_differences(
                    features, features[select_idx : select_idx + 1]  # noqa: E203
                )
                approximate_coreset_anchor_distances = torch.cat(
                    [approximate_coreset_anchor_distances, coreset_select_distance],
                    dim=-1,
                )
                approximate_coreset_anchor_distances = torch.min(
                    approximate_coreset_anchor_distances, dim=1
                ).values.reshape(-1, 1)

        return np.array(coreset_indices)


class RandomSampler(BaseSampler):
    def __init__(self, percentage: float):
        super().__init__(percentage)

    def run(
        self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """Randomly samples input feature collection.

        Args:
            features: [N x D]
        """
        num_random_samples = int(len(features) * self.percentage)
        # np.random.choice：NumPy 的随机选择函数。
        # 第一个参数 len(features)：表示从 0 到 N-1 的范围内进行选择。
        # 第二个参数 num_random_samples：表示要选多少个（即上面算出来的数量）。
        # replace=False：表示不放回抽样。选过的索引不会再被选中，保证选出的索引是唯一的。
        subset_indices = np.random.choice(
            len(features), num_random_samples, replace=False
        )
        subset_indices = np.array(subset_indices)
        return features[subset_indices]
