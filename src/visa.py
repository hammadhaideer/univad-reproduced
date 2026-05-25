import json
import os
import random

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image


class VisaDataset(data.Dataset):
    def __init__(
        self,
        root,
        transform,
        target_transform,
        mode='test',
        k_shot=0,
        save_dir=None,
        obj_name=None,
        image_size=448,
    ):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.image_size = image_size
        self.data_all = []

        with open(os.path.join(self.root, 'meta.json'), 'r') as f:
            meta_info = json.load(f)

        meta_info = meta_info[mode]

        if mode == 'train':
            self.cls_names = [obj_name]
            save_path = os.path.join(save_dir, 'k_shot.txt')
            with open(save_path, 'a') as f:
                for cls_name in self.cls_names:
                    data_tmp = meta_info[cls_name]
                    indices = torch.randint(0, len(data_tmp), (k_shot,))
                    for i in range(len(indices)):
                        self.data_all.append(data_tmp[indices[i]])
                        f.write(data_tmp[indices[i]]['img_path'] + '\n')
        else:
            self.cls_names = list(meta_info.keys())
            for cls_name in self.cls_names:
                self.data_all.extend(meta_info[cls_name])

        self.length = len(self.data_all)

    def __len__(self):
        return self.length

    def get_cls_names(self):
        return self.cls_names

    def __getitem__(self, index):
        data = self.data_all[index]
        img_path = data['img_path']
        mask_path = data['mask_path']
        cls_name = data['cls_name']
        specie_name = data['specie_name']
        anomaly = data['anomaly']

        img_pil = Image.open(
            os.path.join(self.root, img_path)
        ).convert('RGB')

        if anomaly == 0:
            img_mask = Image.fromarray(
                np.zeros((img_pil.size[0], img_pil.size[1])), mode='L'
            )
        else:
            img_mask = np.array(
                Image.open(
                    os.path.join(self.root, mask_path)
                ).convert('L')
            ) > 0
            img_mask = Image.fromarray(
                img_mask.astype(np.uint8) * 255, mode='L'
            )

        img = self.transform(img_pil) if self.transform is not None else img_pil
        img_mask = (
            self.target_transform(img_mask)
            if self.target_transform is not None and img_mask is not None
            else img_mask
        )
        if img_mask is None:
            img_mask = torch.zeros(1, self.image_size, self.image_size)

        return {
            'img_pil': np.array(
                img_pil.resize((self.image_size, self.image_size))
            ),
            'img': img,
            'img_mask': img_mask,
            'cls_name': cls_name.replace('_', ' '),
            'anomaly': anomaly,
            'anomaly_class': specie_name.replace('_', ' '),
            'img_path': os.path.join(self.root, img_path),
        }