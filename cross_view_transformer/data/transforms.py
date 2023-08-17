import pathlib

import torch
import torchvision
import numpy as np

from PIL import Image
from .common import encode, decode
from .augmentations import StrongAug, GeometricAug



class Sample(dict):  # 相当于建立了一个字典 然后入参
    def __init__(
        self,
        token,   # 唯一标签
        scene,
        intrinsics,   #
        extrinsics,  # 外参
        images,
        view,
        bev,     #似乎也是一个标签
        #ego_pose,
        **kwargs
    ):
        super().__init__(**kwargs)

        # Used to create path in save/load
        self.token = token
        self.scene = scene

        self.view = view #
        self.bev = bev

        self.images = images  # 将这个image 改为12张
        self.intrinsics = intrinsics  # 内参
        self.extrinsics = extrinsics  # 外参

        #self.ego_pose=ego_pose # 车辆动态参数

    def __getattr__(self, key):
        return super().__getitem__(key)

    def __setattr__(self, key, val):
        self[key] = val

        return super().__setattr__(key, val)


class SaveDataTransform:
    """
    All data to be saved to .json must be passed in as native Python lists
    """
    def __init__(self, labels_dir):
        self.labels_dir = pathlib.Path(labels_dir)

    def get_cameras(self, batch: Sample):
        return {
            'images': batch.images,
            'intrinsics': batch.intrinsics,
            'extrinsics': batch.extrinsics,
            #'ego_pose': batch.ego_pose
        }

    def get_bev(self, batch: Sample):   #  获取bev文件当中的数据
        result = {
            'view': batch.view,
        }

        scene_dir = self.labels_dir / batch.scene

        bev_path = f'bev_{batch.token}.png'
        Image.fromarray(encode(batch.bev)).save(scene_dir / bev_path)

        result['bev'] = bev_path

        # Auxilliary labels
        if batch.get('aux') is not None:
            aux_path = f'aux_{batch.token}.npz'
            np.savez_compressed(scene_dir / aux_path, aux=batch.aux)

            result['aux'] = aux_path

        # Visibility mask
        if batch.get('visibility') is not None:
            visibility_path = f'visibility_{batch.token}.png'
            Image.fromarray(batch.visibility).save(scene_dir / visibility_path)

            result['visibility'] = visibility_path

        return result

    def __call__(self, batch):
        """
        Save sensor/label data and return any additional info to be saved to json
        """
        result = {}
        result.update(self.get_cameras(batch))
        result.update(self.get_bev(batch))
        result.update({k: v for k, v in batch.items() if k not in result})

        return result


class LoadDataTransform(torchvision.transforms.ToTensor):    #导入数据
    def __init__(self, dataset_dir, labels_dir, image_config, num_classes, augment='none'):
        super().__init__()

        self.dataset_dir = pathlib.Path(dataset_dir)  #数据集路径
        self.labels_dir = pathlib.Path(labels_dir)    #标签路径
        self.image_config = image_config
        self.num_classes = num_classes

        xform = {
            'none': [],
            'strong': [StrongAug()],
            'geometric': [StrongAug(), GeometricAug()],
        }[augment] + [torchvision.transforms.ToTensor()]

        self.img_transform = torchvision.transforms.Compose(xform)
        self.to_tensor = super().__call__        # totensor!! 将数据输入到张量

    def get_cameras(self, sample: Sample, h, w, top_crop):
        """
        Note: we invert I and E here for convenience.
        """
        images = list()   # 图像数据
        intrinsics = list()  # 内参数据

        for image_path, I_original in zip(sample.images, sample.intrinsics):  # 内参
            h_resize = h + top_crop
            w_resize = w

            image = Image.open(self.dataset_dir / image_path)

            # print("************image", self.dataset_dir/image_path)
            # print("************sample_token", sample.token)

            ''' ************image_path samples/CAM_FRONT_LEFT/   n008-2018-09-18-14-43-59-0400__CAM_FRONT_LEFT  __1537296396604799.jpg
                ************image_path samples/CAM_FRONT/        n008-2018-09-18-14-43-59-0400__CAM_FRONT       __1537296396612404.jpg
                ************image_path samples/CAM_FRONT_RIGHT/  n008-2018-09-18-14-43-59-0400__CAM_FRONT_RIGHT __1537296396620482.jpg
                ************image_path samples/CAM_BACK_LEFT/    n008-2018-09-18-14-43-59-0400__CAM_BACK_LEFT   __1537296396647405.jpg
                ************image_path samples/CAM_BACK/         n008-2018-09-18-14-43-59-0400__CAM_BACK        __1537296396637558.jpg
                ************image_path samples/CAM_BACK_RIGHT/   n008-2018-09-18-14-43-59-0400__CAM_BACK_RIGHT  __1537296396628113.jpg'''
            # 注意照片时间 地点的提取方式
            image_new = image.resize((w_resize, h_resize), resample=Image.BILINEAR)
            image_new = image_new.crop((0, top_crop, image_new.width, image_new.height))
            #print("************I_original", I_original)
            '''
            ************I_original [[1256.7485116440405, 0.0, 817.7887570959712], [0.0, 1256.7485116440403, 451.9541780095127], [0.0, 0.0, 1.0]]
            ************I_original [[1259.5137405846733, 0.0, 807.2529053838625], [0.0, 1259.5137405846733, 501.19579884916527], [0.0, 0.0, 1.0]]
            ************I_original [[1256.7414812095406, 0.0, 792.1125740759628], [0.0, 1256.7414812095406, 492.7757465151356], [0.0, 0.0, 1.0]]
            ************I_original [[1249.9629280788233, 0.0, 825.3768045375984], [0.0, 1249.9629280788233, 462.54816385708756], [0.0, 0.0, 1.0]]
            ************I_original [[1254.9860565800168, 0.0, 829.5769333630991], [0.0, 1254.9860565800168, 467.1680561863987], [0.0, 0.0, 1.0]]
            ************I_original [[809.2209905677063, 0.0, 829.2196003259838], [0.0, 809.2209905677063, 481.77842384512485], [0.0, 0.0, 1.0]]
            ************I_original [[1272.5979470598488, 0.0, 826.6154927353808], [0.0, 1272.5979470598488, 479.75165386361925], [0.0, 0.0, 1.0]]
            ************I_original [[796.8910634503094, 0.0, 857.7774326863696], [0.0, 796.8910634503094, 476.8848988407415], [0.0, 0.0, 1.0]]
            ************I_original [[1259.5137405846733, 0.0, 807.2529053838625], [0.0, 1259.5137405846733, 501.19579884916527], [0.0, 0.0, 1.0]]
            ************I_original [[1256.4720761102153, 0.0, 759.9201772536986], [0.0, 1256.472076110215, 418.2347543062189], [0.0, 0.0, 1.0]]
            ************I_original [[1266.417203046554, 0.0, 816.2670197447984], [0.0, 1266.417203046554, 491.50706579294757], [0.0, 0.0, 1.0]]
'''
            I = np.float32(I_original)
            I[0, 0] *= w_resize / image.width
            I[0, 2] *= w_resize / image.width
            I[1, 1] *= h_resize / image.height
            I[1, 2] *= h_resize / image.height
            I[1, 2] -= top_crop


            images.append(self.img_transform(image_new))
            # 将图片，转化成了list   print("************images", images)，已经实现了归一化处理
            intrinsics.append(torch.tensor(I))

        return {
            'cam_idx': torch.LongTensor(sample.cam_ids),
            'image': torch.stack(images, 0),
            'intrinsics': torch.stack(intrinsics, 0),
            'extrinsics': torch.tensor(np.float32(sample.extrinsics)),
        }

    def get_bev(self, sample: Sample):    # 除了获取车辆图像数据，还需要获取车辆BEV数据

        scene_dir = self.labels_dir / sample.scene
        #print("scene_dir", scene_dir)             ##########################
        bev = None

        if sample.bev is not None:
            bev = Image.open(scene_dir / sample.bev)
            #print("scene_dir / sample.bev",scene_dir / sample.bev)            ###############3333
            #bev.show()
            bev = decode(bev, self.num_classes)
            bev = (255 * bev).astype(np.uint8)
            bev = self.to_tensor(bev)   #将BEV数据也输入至tensor

        result = {
            'bev': bev,
            'view': torch.tensor(sample.view),
        }

        if 'visibility' in sample:
            visibility = Image.open(scene_dir / sample.visibility)
            result['visibility'] = np.array(visibility, dtype=np.uint8)

        if 'aux' in sample:
            aux = np.load(scene_dir / sample.aux)['aux']
            result['center'] = self.to_tensor(aux[..., 1])

        if 'pose' in sample:
            result['pose'] = np.float32(sample['pose'])

        return result

    def __call__(self, batch):
        if not isinstance(batch, Sample):
            batch = Sample(**batch)
        #print('batch_01', batch)
        result = dict()
        result.update(self.get_cameras(batch, **self.image_config))
        result.update(self.get_bev(batch))
        #print('batch_02', batch)
        #print('batch_result',result)  #“cam_idx” 相机编号[0, 1, 2, 3, 4, 5]；；image图像；；
                                      # intrinsics内参（6个）；；extrinsics外参；；
                                      #  bev（全为0） ;;view;;visibility（visibility用于定义
                                     #  某个annotation在相机拍摄的照片中的可见程度，分成四组。可用visibility查看）;;center（全为0）；；pose

        return result
