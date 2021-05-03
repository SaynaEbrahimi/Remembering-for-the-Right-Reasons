from __future__ import print_function

from collections import defaultdict
from itertools import chain
from torchvision.datasets.folder import default_loader

import torch.utils.data as data
import pandas as pd
import numpy as np
import os
import torch
from sklearn.model_selection import train_test_split
from PIL import Image

from torchvision.transforms import transforms




class CUB(torch.utils.data.Dataset):
    num_classes = 200
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    def __init__(self, root, train, transform, target_transform=None):

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        self.root = os.path.join(self.root, 'CUB_200_2011')

        f_images = pd.read_csv(os.path.join(self.root, 'images.txt'), header=None)
        images_data = {}
        for row in f_images.values[:, 0].tolist():
            images_data[int(row.split(' ')[0]) - 1] = row.split(' ')[1]

        f_splits = pd.read_csv(os.path.join(self.root, 'train_test_split.txt'), header=None)
        split_dict = {}
        for row in f_splits.values[:, 0].tolist():
            split_dict[int(row.split(' ')[0]) - 1] = int(row.split(' ')[1])

        images = defaultdict(list)
        for k, v in chain(images_data.items(), split_dict.items()):
            images[k].append(v)

        # the following should be used if the regular data split for CUB is to be used (half train half test) for
        # results shown in Table 1b in the main paper.

        train_data, train_targets, test_data, test_targets = [], [], [], []
        for k, v in images.items():
            is_training = v[1]

            if is_training > 0:
                train_data.append(v[0])
                train_targets.append(int(v[0][:3]) - 1)
            else:
                test_data.append(v[0])
                test_targets.append(int(v[0][:3]) - 1)

        if train:
            self.data = train_data
            self.targets = train_targets
        else:
            self.data = test_data
            self.targets = test_targets

    def __getitem__(self, index):
        path, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        img = default_loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.data)


class CUBFSCI(torch.utils.data.Dataset):
    """
    CUB Dataset for few-shot class incremental experiment (arxiv.org/pdf/2004.10956.pdf)

    Uses training images listed from paper's repo: github.com/xyutao/fscil/tree/master/data/index_list
    """
    # TODO: Add memory capabilities
    img_name_path = '../data/fscil_index_list'
    mean = CUB.mean  # [0.485, 0.456, 0.406]
    std = CUB.std  # [0.229, 0.224, 0.225]
    num_classes = CUB.num_classes

    def __init__(self, root, task_id, train, transform, target_transform=None):

        self.root = os.path.expanduser(root)
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        # Assuming task_id's are 0-indexed
        # self.task_id = task_id + 1
        self.task_id = task_id+1

        self.cub_root = os.path.join(self.root, 'CUB_200_2011')

        f_images = pd.read_csv(os.path.join(self.cub_root, 'images.txt'), delim_whitespace=True, names=['id', 'path'],
                               header=None)
        f_targets = pd.read_csv(os.path.join(self.cub_root, 'image_class_labels.txt'), delim_whitespace=True,
                                names=['id', 'class'], header=None)

        if train:
            df_image_names = pd.read_csv(os.path.join(CUBFSCI.img_name_path, 'session_{}.txt'.format(self.task_id)),
                                         names=['relative_path'])
        else:
            df_image_names = pd.read_csv(os.path.join(CUBFSCI.img_name_path, 'test_{}.txt'.format(self.task_id)),
                                         names=['relative_path'])

        self.data = df_image_names['relative_path'].to_list()

        l = len('CUB_200_2011/images') + 1
        data_ids = [f_images.loc[f_images['path'] == path[l:], 'id'].iloc[0] for path in self.data]
        self.targets = [f_targets.loc[f_targets['id'] == id, 'class'].iloc[0] for id in data_ids]
        # print (self.targets)

        self.targets = [target - 1 for target in self.targets]
        print (task_id, max(self.targets), min(self.targets))


        self.classes = set(self.targets)
        self.tt = [task_id for _ in range(len(self.data))]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path, target,tt = os.path.join(self.root, self.data[index]), self.targets[index], self.tt[index]

        img = default_loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img, target, tt


class iCUBFSCI(CUBFSCI):
    """
    iCUB for CUBFSCI to match design of codebase.
    """

    def __init__(self, root, classes_, train, seed, transform, multi_head, task_id, num_samples_per_class,
                 target_transform=None):
        self.train = train
        super(iCUBFSCI, self).__init__(root, task_id, train, transform=transform, target_transform=transform)

        f_classes = os.path.join(self.cub_root, 'classes.txt')
        class_names = np.loadtxt(f_classes, str, delimiter='\t')

        # SAME as iCUB except CUBFSCI has self.classes already
        self.classes_ = classes_
        self.class_mapping_mh = {c: i for i, c in enumerate(self.classes_)}
        self.class_mapping_sh = {c: i + task_id for i, c in enumerate(self.classes_)}
        self.class_mapping = self.class_mapping_mh if multi_head else self.class_mapping_sh

        # self.class_names = {c: ' '.join(class_names[c].split(',')[0].split()[1:])[4:] for c in self.classes}
        # self.idx_to_class = {i: c for i, c in enumerate(list(self.class_names.values()))}

        # original iCub code
        # TODO: Incorporate original iCub Code for memory

        data = []
        labels = []
        for i in range(len(self.data)):
            if self.targets[i] in self.classes:
                data.append(self.data[i])
                if multi_head:
                    labels.append(self.class_mapping_mh[self.targets[i]])
                else:
                    labels.append(self.class_mapping_sh[self.targets[i]])

        self.targets = labels
        self.data = [os.path.join(self.root, path) for path in data]

        if num_samples_per_class and train:
            x, y = [], []
            for l in self.classes:
                indices_with_label_l = []
                for i in range(len(self.data)):
                    if multi_head:
                        if self.targets[i] == self.class_mapping_mh[l]:
                            indices_with_label_l.append(i)
                    else:
                        if self.targets[i] == self.class_mapping_sh[l]:
                            indices_with_label_l.append(i)

                x_with_label_l = [self.data[item] for item in indices_with_label_l]

                np.random.seed(seed=seed)
                shuffled_indices = np.random.permutation(len(x_with_label_l))[:num_samples_per_class]
                x_with_label_l = [x_with_label_l[item] for item in shuffled_indices]

                if multi_head:
                    mapped_label = self.class_mapping_mh[l]
                else:
                    mapped_label = self.class_mapping_sh[l]

                y_with_label_l = [mapped_label] * len(x_with_label_l)

                x.append(x_with_label_l)
                y.append(y_with_label_l)

            data = sum(x, [])
            targets = sum(y, [])

            self.data = data
            self.targets = targets

        self.tt = [task_id for _ in range(len(self.data))]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path, target, tt = self.data[index], self.targets[index], self.tt[index]

        img = default_loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img, target, tt


class iCUB(CUB):

    def __init__(self, root, classes_, train, seed, transform, multi_head, task_id, num_samples_per_class,
                 target_transform=None):
        # train = split == 'train'

        split = 'train' if train else 'val'
        super(iCUB, self).__init__(root, train, transform=transform, target_transform=target_transform)

        f_classes = os.path.join(self.root, 'classes.txt')
        class_names = np.loadtxt(f_classes, str, delimiter='\t')

        self.classes_ = classes_
        self.class_mapping_mh = {c: i for i, c in enumerate(classes_)}
        self.class_mapping_sh = {c: i + task_id for i, c in enumerate(classes_)}
        self.class_mapping = self.class_mapping_mh if multi_head else self.class_mapping_sh

        self.class_names = {c: ' '.join(class_names[c].split(',')[0].split()[1:])[4:] for c in self.classes_}
        self.idx_to_class = {i: c for i, c in enumerate(list(self.class_names.values()))}

        data = []
        labels = []
        for i in range(len(self.data)):
            if self.targets[i] in self.classes_:
                data.append(self.data[i])
                if multi_head:
                    labels.append(self.class_mapping_mh[self.targets[i]])
                else:
                    labels.append(self.class_mapping_sh[self.targets[i]])

        self.targets = labels
        self.data = [os.path.join(self.root, 'images', path) for path in data]

        if num_samples_per_class and train:
            x, y = [], []
            for l in classes_:
                indices_with_label_l = []
                for i in range(len(self.data)):
                    if multi_head:
                        if self.targets[i] == self.class_mapping_mh[l]:
                            indices_with_label_l.append(i)
                    else:
                        if self.targets[i] == self.class_mapping_sh[l]:
                            indices_with_label_l.append(i)

                x_with_label_l = [self.data[item] for item in indices_with_label_l]

                np.random.seed(seed=seed)
                shuffled_indices = np.random.permutation(len(x_with_label_l))[:num_samples_per_class]
                x_with_label_l = [x_with_label_l[item] for item in shuffled_indices]

                if multi_head:
                    mapped_label = self.class_mapping_mh[l]
                else:
                    mapped_label = self.class_mapping_sh[l]

                y_with_label_l = [mapped_label] * len(x_with_label_l)

                x.append(x_with_label_l)
                y.append(y_with_label_l)

            data = sum(x, [])
            targets = sum(y, [])

            self.data = data
            self.targets = targets

            # self.data = np.concatenate((self.data, data))
            # self.targets = np.concatenate((self.targets, targets))

        self.tt = [task_id for _ in range(len(self.data))]

    def __getitem__(self, index):
        path, target, tt = self.data[index], self.targets[index], self.tt[index]

        # doing this so that it is consistent with all other datasets
        img = default_loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, target, tt

    def __len__(self):
        return len(self.data)



class iCUBSegmentation(iCUB):
    """ Same as iCUB, except adds segmentation """

    def __init__(self, root, classes_, train, seed, transform, multi_head, task_id, num_samples_per_class,
                 target_transform=None):
        split = 'train' if train else 'val'
        super(iCUBSegmentation, self).__init__(root, classes_, train, seed, transform, multi_head, task_id,
                                               num_samples_per_class=None, target_transform=None)

        self.segmentations = np.array([
            path.replace('images', 'segmentations').replace('.jpg', '.png') for path in self.data])

        if self.transform is not None:
            seg_transforms = []
            for t in self.transform.transforms:
                if not isinstance(t, transforms.Normalize):
                    seg_transforms.append(t)
            self.seg_transform = transforms.Compose(seg_transforms)
        else:
            self.seg_transform = None

    def __getitem__(self, index):
        path, target, tt = self.data[index], self.targets[index], self.tt[index]

        # doing this so that it is consistent with all other datasets
        img = default_loader(path)
        if self.transform is not None:
            img = self.transform(img)

        segmentation_path = self.segmentations[index]
        segmentation = Image.open(segmentation_path)
        if self.seg_transform is not None:
            segmentation = self.seg_transform(segmentation)
        return img, target, tt, segmentation


class CUBEvidence(CUB):
    """docstring for CUBTask"""

    def __init__(self, root, train, class_task_mapping, transform):
        super(CUBEvidence, self).__init__(root, train, transform=transform, target_transform=None)

        self.task_ids = [class_task_mapping[class_number] for class_number in self.targets]
        self.transform = transform

        self.data = [os.path.join(self.root, 'images', p) for p in self.data]

    def __getitem__(self, index):
        path, target, task_id = self.data[index], self.targets[index], self.task_ids[index]

        # doing this so that it is consistent with all other datasets
        img = default_loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img, target, task_id
