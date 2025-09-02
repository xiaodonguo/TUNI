import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.utils.data as data
from torchvision import transforms
from toolbox.datasets.augmentations import Resize, Compose, ColorJitter, RandomHorizontalFlip, RandomCrop, RandomScale, \
    RandomRotation

# os.environ["CUDA_VISIBLE_DEVICES"] = "5"


class SUS(data.Dataset):

    def __init__(self, cfg, mode='train', do_aug=True):

        assert mode in ['train', 'val', 'test', 'train_day', 'train_night', 'test_day', 'test_night'], f'{mode} not support.'
        self.mode = mode

        ## pre_feature_all-processing
        self.im_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.ir_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.449, 0.449, 0.449], [0.226, 0.226, 0.226]),
        ])

        self.root = cfg['root']
        self.n_classes = cfg['n_classes']

        self.scale_range = tuple(float(i) for i in cfg['scales_range'].split(' '))
        crop_size = tuple(int(i) for i in cfg['crop_size'].split(' '))

        self.aug = Compose([
            ColorJitter(
                brightness=cfg['brightness'],
                contrast=cfg['contrast'],
                saturation=cfg['saturation']),
            RandomHorizontalFlip(cfg['p']),
            RandomScale(self.scale_range),
            RandomCrop(crop_size, pad_if_needed=True)
        ])

        # self.val_resize = Resize(crop_size)

        self.mode = mode
        self.do_aug = do_aug

        if cfg['class_weight'] == 'enet':
            self.class_weight = np.array(
                [2.0013,  4.2586, 27.5196, 22.8228, 11.2535, 31.3595])
            # self.binary_class_weight = np.array([1.5121, 10.2388])
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

        image = Image.open(os.path.join(self.root, 'seperated_images', image_path + '_rgb.jpg'))
        thermal = Image.open(os.path.join(self.root, 'seperated_images', image_path + '_th.jpg')).convert('RGB')
        label = Image.open(os.path.join(self.root, 'labels', image_path + '.png'))
        bound = Image.open(os.path.join(self.root, 'boundary', image_path + '.png'))
        binary = Image.open(os.path.join(self.root, 'binary', image_path + '.png'))


        sample = {
            'image': image,
            'thermal': thermal,
            'label': label,
            'binary': binary,
            'boundary': bound,
        }

        if self.mode in ['train'] and self.do_aug:  # 只对训练集增强
            sample = self.aug(sample)

        sample['image'] = self.im_to_tensor(sample['image'])
        sample['thermal'] = self.ir_to_tensor(sample['thermal'])
        sample['label'] = torch.from_numpy(np.asarray(sample['label'], dtype=np.int64)).long()
        sample['boundary'] = torch.from_numpy(np.asarray(sample['boundary'], dtype=np.int64) / 255.).long()
        sample['binary'] = torch.from_numpy(np.asarray(sample['binary'], dtype=np.int64) / 255.).long()
        sample['label_path'] = image_path.strip().split('/')[-1] + '.png'  # 后期保存预测图时的文件名和label文件名一致
        return sample

    @property
    def cmap(self):
        return [
            (0, 0, 0),  # unlabelled
            (128, 0, 0),  # road
            (0, 128, 0),  # sidewalk
            (128, 128, 0),  # person
            (0, 0, 128),  # vechile
            (128, 0, 128)]  # bicycle
        #     (64, 64, 128),  # guardrail
        #     (192, 128, 128),  # color_cone
        #     (192, 64, 0),  # bump
        # ]


if __name__ == '__main__':
    import json

    path = '/home/ubuntu/code/wild/configs/SUS.json'
    with open(path, 'r') as fp:
        cfg = json.load(fp)
    dataset = SUS(cfg, mode='train', do_aug=True)
    print(dataset[0]['image'].shape)
    print(len(dataset[0]))

    from toolbox.utils import ClassWeight

    # train_loader = torch.utils.data.DataLoader(dataset, batch_size=cfg['ims_per_gpu'], shuffle=True,
    #                                            num_workers=cfg['num_workers'], pin_memory=True, drop_last=True)
    # classweight = ClassWeight('enet')  # enet, median_freq_balancing
    # class_weight = classweight.get_weight(train_loader, 2)
    # class_weight = torch.from_numpy(class_weight).float()
    #
    # print(class_weight)

    # [1.9884, 4.3964, 26.8930, 22.1750, 11.1487, 30.6316] semantic
    # [1.9689, 3.1175] binary
    # [1.4574, 19.0466] boundary
    # [1.4491, 22.1858] person