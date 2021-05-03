
from __future__ import print_function
from PIL import Image
import os
import os.path
from torchvision import transforms
import numpy as np
from math import ceil
import torch
from tqdm import tqdm
from torchvision.datasets.folder import default_loader

class DatasetGen(object):
    """docstring for DatasetGen"""

    def __init__(self, args):
        super(DatasetGen, self).__init__()
        self.args = args
        self.dataset = self.args.experiment.dataset

        if self.args.experiment.dataset == 'cub':
            if self.args.experiment.segmentation:
                print ("Loading segmentated birds...")
                from .cub import iCUBSegmentation as iData
            if self.args.experiment.fscil:
                print("Loading Few-Shot Class Incremental birds...")
                from .cub import iCUBFSCI as iData
            else:
                from .cub import iCUB as iData


        elif self.args.experiment.dataset == 'miniimagenet':
            from .miniimagenet import iMiniImageNet as iData
        elif self.args.experiment.dataset == 'imagenet100':
            from .imagenet import iImageNet100 as iData
        elif self.args.experiment.dataset == 'cifar100':
            from .cifar100 import iCIFAR100 as iData
        else:
            raise NotImplementedError

        if self.args.experiment.raw_memory_only or self.args.experiment.xai_memory:
            main_path = os.path.join(self.args.path.checkpoint, "memory")
            if not os.path.exists(main_path):
                os.mkdir(main_path)


        self.inputsize = [3, args.experiment.im_size, args.experiment.im_size]
        self.iData = iData

        self.mean, self.std = iData.mean, iData.std
        self.num_classes = iData.num_classes

        basic_transform = [transforms.ToTensor(), transforms.Normalize(mean=self.mean, std=self.std)]

        transform_train = []
        transform_test = []

        if self.args.experiment.augmentation:
            if self.args.experiment.dataset == 'cifar100':
                transforms_crop = transforms.RandomCrop(self.args.experiment.im_size, padding=4)
            else:
                transforms_crop = transforms.RandomResizedCrop(self.args.experiment.im_size)

            transform_train += [
                transforms_crop,
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
            ]
        else:
            transform_train += [
                transforms.Resize((args.experiment.im_size, args.experiment.im_size))
            ]

        transform_test += [
                transforms.Resize((args.experiment.im_size, args.experiment.im_size))
            ]
        transform_test += basic_transform
        self.test_transformation =  transforms.Compose(transform_test)

        transform_train += basic_transform
        self.train_transformation = transforms.Compose(transform_train)

        self.num_workers = args.device.workers
        self.pin_memory = True


        self.seed = args.seed
        print ("args.experiment.fs_batch", args.experiment.fs_batchfs_batch)
        if args.experiment.fscil:
            self.batch_size = [args.experiment.fs_batch for _ in range(args.experiment.ntasks)]
            self.batch_size[0] = args.train.batch_size
        else:
            self.batch_size = [args.train.batch_size for _ in range(args.experiment.ntasks)]

        self.pc_valid=args.train.pc_valid
        self.root = args.path.data

        self.num_tasks = args.experiment.ntasks
        if (self.args.experiment.fscil):
            if self.dataset == 'cub':
                self.taskcla = [[0,100]] + [[t,10] for t in range(1,11)]
        else:
            self.taskcla = [[t, int(self.num_classes/self.num_tasks)] for t in range(self.num_tasks)]

        print ("self.taskcla: ", self.taskcla)

        self.use_memory = self.args.experiment.use_memory

        self.indices = {}
        self.idx={}

        self.num_workers = args.device.workers
        self.pin_memory = True

        if self.args.train.shuffle_tasks:
            np.random.seed(self.seed)
            task_ids = np.split(np.random.permutation(self.num_classes), self.num_tasks)
        elif not self.args.train.shuffle_tasks:
            if self.args.experiment.fscil:
                if self.dataset == 'cub':
                    # task_ids = [np.arange(1,101)] + [np.arange(101+10*i, 101+10*(i+1)) for i in range(0,10)]
                    task_ids = [np.arange(0,100)] + [np.arange(100+10*i, 100+10*(i+1)) for i in range(0,10)]
            else:
                task_ids = np.split(np.arange(self.num_classes), self.num_tasks)
            # task_ids = [list(item) for item in task_ids]
            # task_ids = task_ids[::-1]
        self.task_ids = [list(arr) for arr in task_ids]

        self.class_task_mapping = {}
        for idx, task in enumerate(task_ids):
            for cls in task:
                self.class_task_mapping[cls] = idx

        self.print = True



    def get_class_names(self, task_id):
        classes_ = self.task_ids[task_id]
        if self.args.experiment.dataset == 'cub':
            f_classes = os.path.join(self.args.path.data, 'CUB_200_2011', 'classes.txt')
            f_names = np.loadtxt(f_classes, str, delimiter='\t')
            class_names = {c: ' '.join(f_names[c].split(',')[0].split()[1:])[4:] for c in classes_}
            idx_to_class = {i: c for i, c in enumerate(list(class_names.values()))}
            return idx_to_class


    def get_task_datasets(self, task_id):
        multi_head = self.args.architecture.multi_head
        train_set = self.iData(root=self.root, classes_=self.task_ids[task_id],
                               train=True, seed=self.seed, multi_head=multi_head, task_id=task_id,
                               num_samples_per_class=None,
                               transform=self.train_transformation)

        test_set = self.iData(root=self.root, classes_=self.task_ids[task_id],
                              train=False, seed=self.seed, multi_head=multi_head, task_id=task_id,
                              num_samples_per_class=None,
                              transform=self.test_transformation)



        return train_set, test_set


    def get_memory_sets(self, task_id):
        '''
        Generate memory sets for each prior task with the updated spc.
        Note that each dataset generator uses the same indices to select data and hence this effectively reduce the
        memory size for each task and saves the new memory sets into the memory

        :param task_id:  Task ID of the last finished task until now

        '''
        if not self.args.experiment.fscil:
            classes_to_divide_budget_between = sum([item[1] for item in self.taskcla[:task_id + 1]])
            num_samples_per_class = self.args.experiment.memory_budget // classes_to_divide_budget_between

            multi_head = self.args.architecture.multi_head

            for t in tqdm(range(task_id+1), ascii=True, desc='Collecting {} spc: '.format(num_samples_per_class)):
                memory_set = self.iData(root=self.root, classes_=self.task_ids[t],
                                       train=True, seed=self.seed, multi_head=multi_head, task_id=t,
                                       num_samples_per_class=num_samples_per_class,
                                       transform=self.train_transformation)
                print ("Memory collected for task {} is {} samples".format(t,len(memory_set)))
                memory_path = os.path.join(self.args.path.checkpoint, 'memory', 'mem_{}.pth'.format(t))
                torch.save(memory_set, memory_path)
        else:
            # In the few shot CIL problem we have a fixed number of samples per session (400 samples from base classes
            # and 1 sample per class in the remaining tasks)
            if task_id == 0:
                nspc = 4
            else:
                nspc = 1

            memory_set = self.iData(root=self.root, classes_=self.task_ids[task_id],
                                    train=True, seed=self.seed, multi_head=self.args.architecture.multi_head,
                                    task_id=task_id,
                                    num_samples_per_class=nspc,
                                    transform=self.train_transformation)
            print ("Memory collected for task {} is {} samples".format(task_id,len(memory_set)))
            memory_path = os.path.join(self.args.path.checkpoint, 'memory', 'mem_{}.pth'.format(task_id))
            torch.save(memory_set, memory_path)



    def load_memory(self, task_id):
        memory = []
        for t in range(task_id):
            memory_path = os.path.join(self.args.path.checkpoint, 'memory', 'mem_{}.pth'.format(t))
            memory_set = torch.load(memory_path)
            memory.append(memory_set)
        return memory





    def get(self, task_id=None):

        dataloaders = {}
        train_set, test_set = self.get_task_datasets(task_id)
        self.class_names = self.get_class_names(task_id)

        if self.pc_valid > 0:
            split = int(np.floor(self.pc_valid * len(train_set)))
            train_split, valid_split = torch.utils.data.random_split(train_set, [len(train_set) - split, split])
            train_set = train_split

            valid_batch_size = int(self.batch_size[task_id] * self.pc_valid)
            if (valid_batch_size == 0):
                valid_batch_size = 1
            valid_loader = torch.utils.data.DataLoader(valid_split, batch_size=valid_batch_size,
                                                       num_workers=self.num_workers, shuffle=True)
            dataloaders['valid'] = valid_loader


        if task_id > 0 and self.use_memory:
            print ("--"*30)
            print ("Loading memory ... ")
            memory = self.load_memory(task_id)
            new_set = memory + [train_set]
            train_set = torch.utils.data.ConcatDataset(new_set)

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size[task_id],
                                                   num_workers=self.num_workers, shuffle=True)

        test_loader = torch.utils.data.DataLoader(test_set, batch_size=self.batch_size[task_id],
                                                  num_workers=self.num_workers, shuffle=True)

        dataloaders['train'] = train_loader
        dataloaders['test'] = test_loader
        dataloaders['name'] = '{}-{}-{}'.format(self.args.experiment.dataset, task_id,self.task_ids[task_id])
        dataloaders['random_chance'] = 100. / self.num_classes
        dataloaders['classes'] = self.num_classes
        dataloaders['task_ids'] = self.task_ids
        dataloaders['class_names'] = self.class_names

        if self.print:
            print ("Task Name: ", task_id)
            print ("# Total Tasks:             {}".format(self.num_tasks))
            print ("# Classes per task:                  {}".format(self.num_classes // self.num_tasks))
            print ("# Total number of classes to learn:  {}".format(self.num_classes))
            # print ("# Total Memory Images per Task       {}".format(
            #     self.args.experiment.spc*self.num_classes//self.num_tasks))
            # print ("# Images per Class per Task:         {}".format(self.args.experiment.spc))
            print ("---"*30)
            print ("Training set size:   {} images of {}x{}".format(len(train_loader.dataset),self.inputsize[1],self.inputsize[1]))
            if self.pc_valid>0:
                print ("Validation set size: {} images of {}x{}".format(len(valid_loader.dataset),self.inputsize[1],self.inputsize[1]))
                print ("Train+Val  set size: {} images of {}x{}".format(len(valid_loader.dataset)+len(train_loader.dataset),self.inputsize[1],self.inputsize[1]))
            print ("Test set size:       {} images of {}x{}".format(len(test_loader.dataset),self.inputsize[1],self.inputsize[1]))
            print ("---"*30)
            print()

        return dataloaders





    def generate_evidence_loaders(self, task_id):

        evidence_set = []
        for t in range(task_id+1):
            evidence = EvidenceSet(self.args, t, self.train_transformation)
            evidence_set.append(evidence)

        evidence_sets = torch.utils.data.ConcatDataset(evidence_set)

        evidence_loader = torch.utils.data.DataLoader(evidence_sets,
                                                      batch_size=self.args.train.memory_batch_size,
                                                      num_workers=self.args.device.workers,
                                                      shuffle=True)

        print ("Checking saiency sets size: ", len(evidence_sets))
        return evidence_loader



class EvidenceSet(torch.utils.data.Dataset):

    def __init__(self, args, task_id, transforms):
        self.args = args
        sal_path = os.path.join(self.args.path.checkpoint, 'memory', 'sal_{}.pth'.format(task_id))
        mem_path = os.path.join(self.args.path.checkpoint, 'memory', 'mem_{}.pth'.format(task_id))
        pred_path = os.path.join(self.args.path.checkpoint, 'memory', 'pred_{}.pth'.format(task_id))

        memory_set = torch.load(mem_path)
        self.data = memory_set.data
        self.target = memory_set.targets
        self.tt = memory_set.tt
        self.saliencies =torch.load(sal_path)
        self.predictions = torch.load(pred_path)
        self.transform = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.target[idx]
        saliency, tt, pred = self.saliencies[idx], self.tt[idx], self.predictions[idx]

        if self.args.experiment.dataset == 'cub':
            if not torch.is_tensor(img):
                img = default_loader(img)
                img = self.transform(img)

        elif self.args.experiment.dataset == 'miniimagenet':
            if not torch.is_tensor(img):
                img = Image.fromarray(img)
                img = self.transform(img)

        elif self.args.experiment.dataset == 'cifar100':
            if not torch.is_tensor(img):
                img = Image.fromarray(img)
                img = self.transform(img)
        else:
            raise NotImplementedError

        return img, target, saliency, tt, pred




