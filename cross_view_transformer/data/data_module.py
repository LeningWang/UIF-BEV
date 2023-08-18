import torch
import pytorch_lightning as pl

from . import get_dataset_module_by_name

# 处理数据！！！！！！！！！！！！！！！
class DataModule(pl.LightningDataModule):
    def __init__(self, dataset: str, data_config: dict, loader_config: dict):
        super().__init__()

        self.get_data = get_dataset_module_by_name(dataset).get_data

        self.data_config = data_config           # 数据集配置
        self.loader_config = loader_config        # 导入配置文件

    def get_split(self, split, loader=True, shuffle=False):       # 切分数据
        datasets = self.get_data(split=split, **self.data_config)
# datasets：是读取了很多的数据
        if not loader:
            return datasets
                                                     #  连接不同的数据集以构成更大的新数据集：class torch.utils.data.ConcatDataset( [datasets, … ] )
        dataset = torch.utils.data.ConcatDataset(datasets)

        loader_config = dict(self.loader_config)

        if loader_config['num_workers'] == 0:
            loader_config['prefetch_factor'] = 2

        return torch.utils.data.DataLoader(dataset, shuffle=shuffle, **loader_config)

    def train_dataloader(self, shuffle=False):
        return self.get_split('train', loader=True, shuffle=shuffle)

    def val_dataloader(self, shuffle=False):
        return self.get_split('val', loader=True, shuffle=shuffle)
