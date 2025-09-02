import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.utils.data as data
from torchvision import transforms
from toolbox.datasets.augmentations import Resize, Compose, ColorJitter, RandomHorizontalFlip, RandomCrop, RandomScale, \
    RandomRotation


class NYUv2(data.Dataset):

    def __init__(self, cfg, mode='train', do_aug=True):
        assert mode in ['train', 'test']

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

        if cfg['class_weight'] == 'enet':
            self.class_weight = np.array([5.7207, 4.7615, 9.4599, 12.7489, 16.7200, 18.6128, 23.0818, 24.5198,
                                          25.3707, 23.7553, 27.1135, 25.0198, 30.0310, 26.6835, 32.8046, 33.5390,
                                          32.1986, 34.4739, 35.8989, 34.4298, 37.8402, 38.2055, 29.4286, 38.4170,
                                          40.4630, 40.5733, 42.4383, 42.2503, 42.1083, 43.2440, 42.5954, 44.7177,
                                          44.5846, 44.6458, 44.4627, 45.0588, 43.8649, 45.6229, 25.6861, 25.8775,
                                          15.8056])
            # self.binary_class_weight = np.asarray([20, 20])
        elif cfg['class_weight'] == 'median_freq_balancing':
            self.class_weight = np.array([0.1048, 0.0839, 0.1959, 0.2881, 0.3726, 0.5093, 0.6121, 0.8205, 0.8337,
                                          0.7482, 0.6981, 0.8627, 1.1372, 0.8133, 1.1034, 1.4300, 1.0000, 0.9371,
                                          2.0121, 0.9904, 1.9158, 2.0197, 1.1565, 2.5478, 1.6299, 2.0030, 4.3178,
                                          2.5112, 0.9394, 4.5706, 0.8648, 2.5528, 4.1016, 2.2452, 4.7550, 6.3821,
                                          1.5921, 7.2107, 0.9008, 0.9054, 0.3956])
            self.binary_class_weight = np.asarray([1, 1])
        else:
            raise (f"{cfg['class_weight']} not support.")

        self.mode = mode
        self.do_aug = do_aug
        self.train_ids, self.test_ids = train_test_split(np.arange(1449), train_size=795, random_state=3)

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_ids)
        else:
            return len(self.test_ids)

    def __getitem__(self, index):
        if self.mode == 'train':
            image_index = self.train_ids[index]
        else:
            image_index = self.test_ids[index]

        image_path = f'all_data/image/{image_index}.jpg'
        depth_path = f'all_data/depth/{image_index}.png'
        label_path = f'all_data/label/{image_index}.png'
        binary_label_path = f'all_data/binary_labels/{image_index}.png'
        bound_path = f'all_data/bound/{image_index}.png'

        image = Image.open(os.path.join(self.root, image_path))  # RGB_T 0~255
        depth = Image.open(os.path.join(self.root, depth_path)).convert('RGB_T')  # 1 channel -> 3
        label = Image.open(os.path.join(self.root, label_path))  # 1 channel 0~40
        binary_label = Image.open(os.path.join(self.root, binary_label_path))
        bound = Image.open(os.path.join(self.root, bound_path)).convert('L')

        sample = {
            'image': image,
            'depth': depth,
            'label': label,
            'bound': bound,
            'binary_label': binary_label,
        }

        if self.mode == 'train' and self.do_aug:  # 只对训练集增强
            sample = self.aug(sample)

        sample['image'] = self.im_to_tensor(np.array(sample['image']))
        sample['depth'] = self.dp_to_tensor(np.array(sample['depth']))
        sample['label'] = torch.from_numpy(np.array(sample['label'], dtype=np.int64)).long()
        sample['bound'] = torch.from_numpy(np.array(sample['bound'], dtype=np.float32) / 255.).long()
        sample['binary_label'] = torch.from_numpy(np.asarray(sample['binary_label'], dtype=np.int64) / 255.).long()

        sample['label_path'] = label_path.strip().split('/')[-1]  # 后期保存预测图时的文件名和label文件名一致
        return sample

    @property
    def cmap(self):
        return [(0, 0, 0),
                (128, 0, 0), (0, 128, 0), (128, 128, 0),
                (0, 0, 128), (128, 0, 128), (0, 128, 128), (128, 128, 128),
                (64, 0, 0), (192, 0, 0), (64, 128, 0),
                (192, 128, 0), (64, 0, 128), (192, 0, 128),
                (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0),
                (0, 192, 0), (128, 192, 0), (0, 64, 128), (128, 64, 128),
                (0, 192, 128), (128, 192, 128), (64, 64, 0), (192, 64, 0),
                (64, 192, 0), (192, 192, 0), (64, 64, 128), (192, 64, 128),
                (64, 192, 128), (192, 192, 128), (0, 0, 64), (128, 0, 64),
                (0, 128, 64), (128, 128, 64), (0, 0, 192), (128, 0, 192),
                (0, 128, 192), (128, 128, 192), (64, 0, 64)]  # 41


if __name__ == '__main__':
    import json

    path = '/home/dtrimina/Desktop/lxy/Segmentation_final/configs/cccmodel/nyuv2_cccmodel.json'
    with open(path, 'r') as fp:
        cfg = json.load(fp)
    cfg['root'] = '/home/dtrimina/Desktop/lxy/database/nyuv2'
    # dataset = NYUv2(cfg, mode='train')
    # from toolbox.utils import class_to_RGB
    # import matplotlib.pyplot as plt
    #
    # for i in range(len(dataset)):
    #     sample = dataset[i]
    #
    #     image = sample['image']
    #     depth = sample['depth']
    #     label = sample['label']
    #
    #     image = image.numpy()
    #     image = image.transpose((1, 2, 0))
    #     image *= np.asarray([0.229, 0.224, 0.225])
    #     image += np.asarray([0.485, 0.456, 0.406])
    #
    #     depth = depth.numpy()
    #     depth = depth.transpose((1, 2, 0))
    #     depth *= np.asarray([0.226, 0.226, 0.226])
    #     depth += np.asarray([0.449, 0.449, 0.449])
    #
    #     label = label.numpy()
    #     label = class_to_RGB(label, N=41, cmap=dataset.cmap)
    #
    #     plt.subplot('131')
    #     plt.imshow(image)
    #     plt.subplot('132')
    #     plt.imshow(depth)
    #     plt.subplot('133')
    #     plt.imshow(label)
    #
    #     plt.show()
    #
    #     if i == 5:
    #         break

    dataset = NYUv2(cfg, mode='train', do_aug=False)
    from toolbox.utils import ClassWeight

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=cfg['ims_per_gpu'], shuffle=True,
                                               num_workers=cfg['num_workers'], pin_memory=True)
    classweight = ClassWeight('enet')  # enet, median_freq_balancing
    class_weight = classweight.get_weight(train_loader, 2)
    class_weight = torch.from_numpy(class_weight).float()
    # class_weight[cfg['id_unlabel']] = 0

    print(class_weight)
    # 1.5844, 6.7297

    # # median_freq_balancing
    # tensor([0.1048, 0.0839, 0.1959, 0.2881, 0.3726, 0.5093, 0.6121, 0.8205, 0.8337,
    #         0.7482, 0.6981, 0.8627, 1.1372, 0.8133, 1.1034, 1.4300, 1.0000, 0.9371,
    #         2.0121, 0.9904, 1.9158, 2.0197, 1.1565, 2.5478, 1.6299, 2.0030, 4.3178,
    #         2.5112, 0.9394, 4.5706, 0.8648, 2.5528, 4.1016, 2.2452, 4.7550, 6.3821,
    #         1.5921, 7.2107, 0.9008, 0.9054, 0.3956])

    # # enet
    # tensor([5.7207, 4.7615, 9.4599, 12.7489, 16.7200, 18.6128, 23.0818, 24.5198,
    #         25.3707, 23.7553, 27.1135, 25.0198, 30.0310, 26.6835, 32.8046, 33.5390,
    #         32.1986, 34.4739, 35.8989, 34.4298, 37.8402, 38.2055, 29.4286, 38.4170,
    #         40.4630, 40.5733, 42.4383, 42.2503, 42.1083, 43.2440, 42.5954, 44.7177,
    #         44.5846, 44.6458, 44.4627, 45.0588, 43.8649, 45.6229, 25.6861, 25.8775,
    #         15.8056])

    # bound classweight = [1.5844, 6.7297]
