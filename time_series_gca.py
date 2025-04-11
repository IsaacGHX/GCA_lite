from GCA_base import GCABase
import time
import torch
import numpy as np
from functools import wraps
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from models.model import Generator_gru, Generator_lstm, Generator_transformer, Discriminator3
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

from utils.multiGAN_trainer import train_multi_gan
from typing import List, Optional


def log_execution_time(func):
    """装饰器：记录函数的运行时间，并动态获取函数名"""

    @wraps(func)  # 保留原函数的元信息（如 __name__）
    def wrapper(*args, **kwargs):
        start_time = time.time()  # 记录开始时间
        result = func(*args, **kwargs)  # 执行目标函数
        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time  # 计算耗时

        # 动态获取函数名（支持类方法和普通函数）
        func_name = func.__name__
        print(f"GCA_time_series - '{func_name}' elapse time: {elapsed_time:.4f} sec")
        return result

    return wrapper


class GCA_time_series(GCABase):
    def __init__(self, N_pairs: int, batch_size: int, num_epochs: int,
                 generators_names: List, discriminators_names: Optional[List],
                 ckpt_path: str, output_path: str,
                 window_sizes: int,
                 initial_learning_rate: float = 2e-5,
                 train_split: float = 0.8,
                 do_distill: bool = False,
                 precise=torch.float32,
                 device=None,
                 seed: int = None,
                 gan_weights=None,
                 ):
        """
        初始化必备的超参数。

        :param N_pairs: 生成器or对抗器的个数
        :param batch_size: 小批次处理
        :param num_epochs: 预定训练轮数
        :param initial_learning_rate: 初始学习率
        :param generators_names: list object，包括了表示具有不同特征的生成器的名称
        :param discriminators_names: list object，包括了表示具有不同判别器的名称，如果没有就不写默认一致
        :param ckpt_path: 各模型检查点
        :param output_path: 可视化、损失函数的log等输出路径
        """
        super().__init__(N_pairs, batch_size, num_epochs,
                         generators_names, discriminators_names,
                         ckpt_path, output_path,
                         initial_learning_rate,
                         train_split,
                         precise,
                         do_distill,
                         device,
                         seed)  # 调用父类初始化

        self.window_sizes = window_sizes
        self.generator_dict = {
            "gru": Generator_gru,
            "lstm": Generator_lstm,
            "transformer": Generator_transformer,
        }

        self.discriminator_dict = {
            "default": Discriminator3,
        }
        self.gan_weights = gan_weights

        self.init_hyperparameters()

    @log_execution_time
    def process_data(self, data_path, target_columns, feature_columns):
        """
        Process the input data by loading, splitting, and normalizing it.

        Args:
            data_path (str): Path to the CSV data file
            target_columns (list): Indices of target columns
            feature_columns (list): Indices of feature columns

        Returns:
            tuple: (train_x, test_x, train_y, test_y, y_scaler)
        """
        print(f"Processing data with seed: {self.seed}")  # Using self.seed

        # Load data
        data = pd.read_csv(data_path)

        # Select target columns
        y = data.iloc[:, target_columns].values
        target_column_names = data.columns[target_columns]
        print("Target columns:", target_column_names)

        # Select feature columns
        x = data.iloc[:, feature_columns].values
        feature_column_names = data.columns[feature_columns]
        print("Feature columns:", feature_column_names)

        # Data splitting using self.train_split
        train_size = int(data.shape[0] * self.train_split)
        train_x, test_x = x[:train_size], x[train_size:]
        train_y, test_y = y[:train_size], y[train_size:]

        # Normalization
        self.x_scaler = MinMaxScaler(feature_range=(0, 1))  # Store scaler as instance variable
        self.y_scaler = MinMaxScaler(feature_range=(0, 1))  # Store scaler as instance variable

        self.train_x = self.x_scaler.fit_transform(train_x)
        self.test_x = self.x_scaler.transform(test_x)

        self.train_y = self.y_scaler.fit_transform(train_y)
        self.test_y = self.y_scaler.transform(test_y)

        return self.train_x, self.test_x, self.train_y, self.test_y, self.y_scaler

    def create_sequences_combine(self, x, y, window_size, start):
        x_ = []
        y_ = []
        y_gan = []
        for i in range(start, x.shape[0]):
            tmp_x = x[i - window_size: i, :]
            tmp_y = y[i]
            tmp_y_gan = y[i - window_size: i + 1]
            x_.append(tmp_x)
            y_.append(tmp_y)
            y_gan.append(tmp_y_gan)
        x_ = torch.from_numpy(np.array(x_)).float()
        y_ = torch.from_numpy(np.array(y_)).float()
        y_gan = torch.from_numpy(np.array(y_gan)).float()
        return x_, y_, y_gan

    @log_execution_time
    def init_dataloader(self):
        """初始化用于训练与评估的数据加载器"""

        # Sliding Window Processing
        # 分别生成不同 window_size 的序列数据
        train_data_list = [
            self.create_sequences_combine(self.train_x, self.train_y, w, self.window_sizes[-1])
            for w in self.window_sizes
        ]

        test_data_list = [
            self.create_sequences_combine(self.test_x, self.test_y, w, self.window_sizes[-1])
            for w in self.window_sizes
        ]

        # 分别提取 x、y、y_gan 并堆叠
        self.train_x_all = [x.to(self.device) for x, _, _ in train_data_list]
        self.train_y_all = train_data_list[0][1]  # 所有 y 应该相同，取第一个即可，不用cuda因为要eval
        self.train_y_gan_all = [y_gan.to(self.device) for _, _, y_gan in train_data_list]

        self.test_x_all = [x.to(self.device) for x, _, _ in test_data_list]
        self.test_y_all = test_data_list[0][1]
        self.test_y_gan_all = [y_gan.to(self.device) for _, _, y_gan in test_data_list]

        assert all(torch.equal(train_data_list[0][1], y) for _, y, _ in train_data_list), "Train y mismatch!"
        assert all(torch.equal(test_data_list[0][1], y) for _, y, _ in test_data_list), "Test y mismatch!"

        """
        train_x_all.shape  # (N, N, W, F)  不同 window_size 会导致 W 不一样，只能在 W 相同时用 stack
        train_y_all.shape  # (N,)
        train_y_gan_all.shape  # (3, N, W+1)
        """

        self.dataloaders = []

        for i, (x, y_gan) in enumerate(zip(self.train_x_all, self.train_y_gan_all)):
            shuffle_flag = ("transformer" in self.generator_names[i])  # 最后一个设置为 shuffle=True，其余为 False
            dataloader = DataLoader(
                TensorDataset(x, y_gan),
                batch_size=self.batch_size,
                shuffle=shuffle_flag,
                generator=torch.manual_seed(self.seed)
            )
            self.dataloaders.append(dataloader)

    def init_model(self):
        """模型结构初始化"""
        assert len(self.generator_names) == self.N, "Generators and Discriminators mismatch!"
        assert isinstance(self.generator_names, list)
        for i in range(self.N):
            assert isinstance(self.generator_names[i], str)

        self.generators = []
        self.discriminators = []

        for i, name in enumerate(self.generator_names):
            # 获取对应的 x, y
            x = self.train_x_all[i]
            y = self.train_y_all[i]

            # 初始化生成器
            GenClass = self.generator_dict[name]
            if "transformer" in name:
                gen_model = GenClass(x.shape[-1], output_len = y.shape[-1]).to(self.device)
            else:
                gen_model = GenClass(x.shape[-1], y.shape[-1]).to(self.device)

            self.generators.append(gen_model)

            # 初始化判别器（默认只用 Discriminator3）
            DisClass = self.discriminator_dict[
                "default" if self.discriminators_names is None else self.discriminators_names[i]]
            dis_model = DisClass(self.window_sizes[i], out_size=y.shape[-1]).to(self.device)
            self.discriminators.append(dis_model)

    def init_hyperparameters(self, ):
        """初始化训练所需的超参数"""
        # 初始化：对角线上为1，其余为0，最后一列为1.0
        self.init_GDweight = []
        for i in range(self.N):
            row = [0.0] * self.N
            row[i] = 1.0
            row.append(1.0)  # 最后一列为 scale
            self.init_GDweight.append(row)

        if self.gan_weights is None:
            # 最终：均分组合，最后一列为1.0
            final_row = [round(1.0 / self.N, 3)] * self.N + [1.0]
            self.final_GDweight = [final_row[:] for _ in range(self.N)]
        else:
            pass

        self.g_learning_rate = self.initial_learning_rate
        self.d_learning_rate = self.initial_learning_rate
        self.adam_beta1, self.adam_beta2 = (0.9, 0.999)
        self.schedular_factor = 0.1
        self.schedular_patience = 16
        self.schedular_min_lr = 1e-7

    def train(self):
        train_multi_gan(self.generators, self.discriminators, self.dataloaders,
                        self.window_sizes,
                        self.y_scaler, self.train_x_all, self.train_y_all, self.test_x_all, self.test_y_all,
                        self.do_distill,
                        self.num_epochs,
                        self.output_path,
                        self.device,
                        init_GDweight=self.init_GDweight,
                        final_GDweight=self.final_GDweight)

    def distill(self):
        """评估模型性能并可视化结果"""
        pass

    def visualize_and_evaluate(self):
        """评估模型性能并可视化结果"""
        pass

    def init_history(self):
        """初始化训练过程中的指标记录结构"""
        pass
