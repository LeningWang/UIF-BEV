import json
import torch

from pathlib import Path
from .common import get_split
from .transforms import Sample, LoadDataTransform


def get_data(
    dataset_dir,   # 数据集的路径
    labels_dir,    # 标签的路径
    split,
    version,
    num_classes,
    augment='none',
    image=None,                         # image config
    dataset='unused',                   # ignore
    **dataset_kwargs
):
    dataset_dir = Path(dataset_dir)   # 数据集路径
    labels_dir = Path(labels_dir)    # 标签路径

    # Override augment if not training
    augment = 'none' if split != 'train' else augment
    transform = LoadDataTransform(dataset_dir, labels_dir, image, num_classes, augment)

    # Format the split name
    split = f'mini_{split}' if version == 'v1.0-mini' else split
    split_scenes = get_split(split, 'nuscenes')

    return [NuScenesGeneratedDataset(s, labels_dir, transform=transform) for s in split_scenes]


class NuScenesGeneratedDataset(torch.utils.data.Dataset):
    """
    Lightweight dataset wrapper around contents of a JSON file

    Contains all camera info, image_paths, label_paths ...
    that are to be loaded in the transform
    """
    def __init__(self, scene_name, labels_dir, transform=None):
        self.samples = json.loads((Path(labels_dir) / f'{scene_name}.json').read_text()) #读取json数据
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        #print("####################################idx",idx)
        
        data01 = Sample(**self.samples[idx])  #每次的采样，为一个场景下的，不同时间的数据，不要被最后几位数字迷糊！！！！
        data02 = Sample(**self.samples[idx])

        #data = data+Sample(**self.samples[3])
        #print("data01.images",data01.scene,data01.images)
        #print(type(data))

        if self.transform is not None:

            #print(data01)
            #print(type(data01))
            #print("auxxxxxxx",data01['aux'])

            data = self.transform(data01)

            data02 = self.transform(data02)
            #data["image"]=torch.cat([data["image"],data02["image"]],1)
            #data["image"]=data["image"]+data02["image"]
            ######data["bev"]=data["bev"]+data02["bev"]


            data["cam_idx"]=torch.cat([data["cam_idx"],data02["cam_idx"]],0)
            data["image"]=torch.cat([data["image"],data02["image"]],0)  # 必须有
            data["intrinsics"]=torch.cat([data["intrinsics"],data02["intrinsics"]],0)  # 必须有
            data["extrinsics"]=torch.cat([data["extrinsics"],data02["extrinsics"]],0) # 必须有
            data["view"]=torch.cat([data["view"],data02["view"]],0)
            data["center"]=torch.cat([data["center"],data02["center"]],0) # 可以无


            

        return data
