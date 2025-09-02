import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import random
import torch
import torch.utils.data as data
from torchvision import transforms
from toolbox.datasets.augmentations import Resize, Compose, ColorJitter, RandomHorizontalFlip, RandomCrop, RandomScale, \
    RandomRotation
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as F
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"


class SUS(data.Dataset):

    def __init__(self, cfg, mode='train', do_aug=True):

        assert mode in ['train', 'val', 'test', 'test_day', 'test_night'], f'{mode} not support.'
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
        self.crop_size = tuple(int(i) for i in cfg['crop_size'].split(' '))

        self.aug = Compose([
            ColorJitter(
                brightness=cfg['brightness'],
                contrast=cfg['contrast'],
                saturation=cfg['saturation']),
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
        box = read_boxes_from_txt(os.path.join(self.root, 'box', image_path + '.txt'))


        sample = {
            'image': image,
            'thermal': thermal,
            'label': label,
            'binary': binary,
            'boundary': bound,
        }

        if self.mode in ['train'] and self.do_aug:  # 只对训练集增强
            # 颜色映射
            sample = self.aug(sample)

            # 随机翻转
            if random.random() < 0.5:
                for key in sample.keys():
                    sample[key] = F.hflip(sample[key])
                box = [horizontal_flip_box(i, image.width) for i in box]

            # 随机缩放
            w, h = sample['image'].size
            scale_factor = random.uniform(self.scale_range[0], self.scale_range[1])
            size = (int(round(h * scale_factor)), int(round(w * scale_factor)))
            for key in sample.keys():
                # BILINEAR for image
                if key in ['image']:
                    sample[key] = F.resize(sample[key], size, interpolation=InterpolationMode.BILINEAR)
                # NEAREST for depth, label, bound
                else:
                    sample[key] = F.resize(sample[key], size, interpolation=InterpolationMode.NEAREST)
            box = [scale_box(i, scale_factor) for i in box]


            # 随机裁剪
            # --- 填充（如果图像尺寸小于裁剪尺寸）---
            crop_x = random.randint(0, max(0, size[0] - self.crop_size[0]))
            crop_y = random.randint(0, max(0, size[1] - self.crop_size[1]))
            for key in sample.keys():
                pad_width = max(0, self.crop_size[0] - size[0])
                pad_height = max(0, self.crop_size[1] - size[1])
                if pad_width > 0 or pad_height > 0:
                    # 对称填充（推荐）或零填充
                    padding = (0, 0, pad_width, pad_height)  # (左, 上, 右, 下)
                    sample[key] = F.pad(sample[key], padding, fill=0)  # fill=0 是零填充
                sample[key] = F.crop(sample[key], crop_x, crop_y, self.crop_size[0], self.crop_size[1])

            box = [crop_box(i, crop_y, crop_x, self.crop_size[1], self.crop_size[0]) for i in box]



        sample['image'] = self.im_to_tensor(sample['image'])
        sample['thermal'] = self.ir_to_tensor(sample['thermal'])
        sample['label'] = torch.from_numpy(np.asarray(sample['label'], dtype=np.int64)).long()
        sample['boundary'] = torch.from_numpy(np.asarray(sample['boundary'], dtype=np.int64) / 255.).long()
        sample['binary'] = torch.from_numpy(np.asarray(sample['binary'], dtype=np.int64) / 255.).long()
        sample['box'] = torch.tensor(process_boxes(box), dtype=torch.int64)
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

def read_boxes_from_txt(file_path):
    boxes = []  # 用于存储所有box的列表
    with open(file_path, 'r') as file:  # 打开文件
        for line in file:  # 逐行读取
            # 去除行首尾的空白字符，并按空格分割成列表
            box_values = line.strip().split()
            # 将每个值转换为整数，并组合成元组
            box_tuple = tuple(map(int, box_values[:4]))
            boxes.append(box_tuple)  # 将元组添加到列表中
    return boxes

def process_boxes(boxes, threshold=10):
    # 如果boxes的数量小于阈值，则通过复制使其达到阈值
    if len(boxes) == 0:
        boxes = [(0, 0, 0, 0) for _ in range(10)]
    elif len(boxes) < threshold:
        # 计算需要复制的次数
        repeat_times = threshold // len(boxes)
        remainder = threshold % len(boxes)
        # 复制boxes
        boxes = boxes * repeat_times + boxes[:remainder]
    # 如果boxes的数量大于阈值，则只保留前threshold个
    elif len(boxes) > threshold:
        boxes = boxes[:threshold]
    return boxes

def horizontal_flip_box(box, image_width):
    """
    对边界框进行水平翻转。
    box: [x1, y1, x2, y2]
    image_width: 图像的宽度
    """
    x1, y1, x2, y2 = box
    new_x1 = image_width - x2
    new_x2 = image_width - x1
    return (new_x1, y1, new_x2, y2)

def scale_box(box, scale_factor):
    """
    对边界框进行缩放。
    box: [x1, y1, x2, y2]
    scale_factor: 缩放比例
    """
    x1, y1, x2, y2 = box
    return (int(x1 * scale_factor), int(y1 * scale_factor), int(x2 * scale_factor), int(y2 * scale_factor))

def crop_box(box, crop_x, crop_y, crop_width, crop_height):
    """
    对边界框进行裁剪。
    box: [x1, y1, x2, y2]
    crop_x, crop_y: 裁剪区域的左上角坐标
    crop_width, crop_height: 裁剪区域的宽高
    """
    x1, y1, x2, y2 = box
    # 计算裁剪后的新坐标
    new_x1 = max(x1 - crop_x, 0)
    new_y1 = max(y1 - crop_y, 0)
    new_x2 = min(x2 - crop_x, crop_width)
    new_y2 = min(y2 - crop_y, crop_height)
    return (new_x1, new_y1, new_x2, new_y2)



if __name__ == '__main__':
    import json

    path = '/home/ubuntu/code/wild/configs/SUS.json'
    with open(path, 'r') as fp:
        cfg = json.load(fp)
    dataset = SUS(cfg, mode='train', do_aug=True)
    print(dataset[0])
    print(len(dataset))

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

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import torch
    from torchvision.transforms import functional as F


    def visualize_sample(sample):
        """
        在图像上绘制边界框
        sample: 从 dataset 取出的样本, 包含 'image' 和 'box'
        """
        image = sample['image']
        boxes = sample['box']  # 这里 box 可能是 Tensor 或 list

        # 处理图像格式: (C, H, W) -> (H, W, C)
        image = F.to_pil_image(image)

        # 创建画布
        fig, ax = plt.subplots(1, figsize=(8, 6))
        ax.imshow(image)

        # 遍历绘制所有边界框
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)  # 确保坐标是整数
            width, height = x2 - x1, y2 - y1
            rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

        ax.axis("off")
        plt.show()


    # 取出 dataset 的一个样本
    # sample = dataset[0]  # dataset 是你的 `SUS` 数据集对象
    # for i, sample in enumerate(dataset):
    #     if i <= 20:
    #         print(sample['image'].shape)
    #         print(sample['label_path'])
    #         visualize_sample(sample)

