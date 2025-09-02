import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.utils.data as data
from torchvision import transforms
from toolbox.datasets.augmentations import Resize, Compose, ColorJitter, RandomHorizontalFlip, RandomCrop, RandomScale, \
    RandomRotation


class FMB(data.Dataset):

    def __init__(self, cfg, mode='train', do_aug=True):

        assert mode in ['train', 'val', 'test', 'test_day', 'test_night', 'train_day', 'train_night', 'trainval'], f'{mode} not support.'
        self.mode = mode

        ## pre_feature_all-processing
        self.im_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.dp_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.449, 0.449, 0.449], [0.226, 0.226, 0.226]),
        ])

        self.root = cfg['root']
        self.n_classes = cfg['n_classes']

        scale_range = tuple(float(i) for i in cfg['scales_range'].split(' '))
        crop_size = tuple(int(i) for i in cfg['crop_size'].split(' '))

        self.aug = Compose([
            ColorJitter(
                brightness=cfg['brightness'],
                contrast=cfg['contrast'],
                saturation=cfg['saturation']),
            RandomHorizontalFlip(cfg['p']),
            RandomScale(scale_range),
            RandomCrop(crop_size, pad_if_needed=True)
        ])

        # self.val_resize = Resize(crop_size)

        self.mode = mode
        self.do_aug = do_aug

        if cfg['class_weight'] == 'enet':
            self.class_weight = np.array(
                [1.5105, 16.6591, 29.4238, 34.6315, 40.0845, 41.4357, 47.9794, 45.3725, 44.9000])
            self.binary_class_weight = np.array([1.5121, 10.2388])
        elif cfg['class_weight'] == 'median_freq_balancing':
            self.class_weight = np.array(
                [0.0118, 0.2378, 0.7091, 1.0000, 1.9267, 1.5433, 0.9057, 3.2556, 1.0686])
            self.binary_class_weight = np.array([0.5454, 6.0061])
        else:
            raise (f"{cfg['class_weight']} not support.")

        with open(os.path.join(self.root, f'{mode}.txt'), 'r') as f:
            self.infos = f.readlines()


    def __len__(self):
        return len(self.infos)

    def __getitem__(self, index):
        image_path = self.infos[index].strip()

        # image = Image.open(os.path.join(self.root, 'images', image_path + '.png'))
        # label = Image.open(os.path.join(self.root, 'labels', image_path + '.png'))

        image = Image.open(os.path.join(self.root, f'{self.mode}', 'Visible', image_path))
        thermal = Image.open(os.path.join(self.root, f'{self.mode}', 'Infrared', image_path)).convert('RGB')
        label = Image.open(os.path.join(self.root, f'{self.mode}', 'Label', image_path))
        # bound = Image.open(os.path.join(self.root, 'bound', image_path+'.png'))
        # binary_label = Image.open(os.path.join(self.root, 'binary_labels', image_path + '.png'))
        # attention_map = Image.open(os.path.join(self.root, 'attention_map', image_path + '.png'))

        # image = np.asarray(image)
        # im = image[:, :, :3]
        # dp = image[:, :, 3:]
        # dp = np.concatenate([dp, dp, dp], axis=2)
        # image = Image.fromarray(im)
        # depth = Image.fromarray(dp)

        sample = {
            'image': image,
            'thermal': thermal,
            'label': label,
            # 'boundary_label': bound,
            # 'binary_label': binary_label,
            # 'attention_map': attention_map,
        }

        if self.mode in ['train'] and self.do_aug:  # 只对训练集增强
            sample = self.aug(sample)

        sample['image'] = self.im_to_tensor(sample['image'])
        sample['thermal'] = self.dp_to_tensor(sample['thermal'])
        sample['label'] = torch.from_numpy(np.asarray(sample['label'], dtype=np.int64)).long()
        # sample['boundary_label'] = torch.from_numpy(np.asarray(sample['boundary_label'], dtype=np.int64) / 255.).long()
        # sample['binary_label'] = torch.from_numpy(np.asarray(sample['binary_label'], dtype=np.int64) / 255.).long()
        # sample['attention_map'] = torch.from_numpy(np.asarray(sample['attention_map'], dtype=np.int64) / 255.).long()

        sample['label_path'] = image_path.strip().split('/')[-1]  # 后期保存预测图时的文件名和label文件名一致
        return sample

    @property
    def cmap(self):
        return [
        (0, 0, 0),
        (179, 228, 228),  # road
        (181, 57, 133),  # sidewalk
        (67, 162, 177),  # building
        (200, 178, 50),  # lamp
        (132, 45, 199),  # sign
        (66, 172, 84),  # vegetation
        (179, 73, 79),  # sky
        (76, 99, 166),  # person
        (66, 121, 253),  # car
        (137, 165, 91),  # truck
        (155, 97, 152),  # bus
        (105, 153, 140),  # motocycle
        (222, 215, 158),  # bicycle
        (135, 113, 90),  # pole
        ]


if __name__ == '__main__':
    import json

    path = '/home/ubuntu/code/wild/configs/FMB.json'
    with open(path, 'r') as fp:
        cfg = json.load(fp)
    # cfg['root'] = '/home/guoxiaodong/code/seg/Semantic_Segmentation_Street_Scenes/irseg'

    dataset = FMB(cfg, mode='train', do_aug=True)

    print(len(dataset))

    from toolbox.utils import ClassWeight

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=cfg['ims_per_gpu'], shuffle=True,
                                               num_workers=cfg['num_workers'], pin_memory=True)
    classweight = ClassWeight('enet')  # enet, median_freq_balancing
    class_weight = classweight.get_weight(train_loader, 15)
    class_weight = torch.from_numpy(class_weight).float()

    print(class_weight)

# [10.2249,  9.6609, 32.8497,  6.0635, 48.1396, 44.9108,  4.4491,  3.1748,
#        43.9271, 15.9236, 43.1266, 44.8469, 48.6038, 50.4826, 27.1057]