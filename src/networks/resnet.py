import sys
import torch
import torchvision
import utils



class Net(torch.nn.Module):

    def __init__(self,args):
        super(Net,self).__init__()
        self.args=args
        self.ncha,size,_=args.inputsize
        self.taskcla=args.taskcla
        self.num_tasks=args.experiment.ntasks
        self.device=args.device.name
        self.multi_head = args.architecture.multi_head
        self.pretrained = args.architecture.pretrained
        self.softmax = torch.nn.Softmax(dim=1)

        model = torchvision.models.__dict__[args.architecture.backbone](self.pretrained)

        # if args.architecture.backbone.startswith('densenet'):
        #     num_ftrs = model.classifier.in_features

        if args.architecture.backbone == 'resnet18':
            num = -2
            if args.experiment.dataset == 'miniimagenet':
                num_ftrs = 4608
            elif args.experiment.dataset == 'cub' or args.experiment.dataset == 'imagenet100':
                num_ftrs = 25088  #without average pool
            elif args.experiment.dataset == 'cifar100' or args.experiment.dataset == 'cifar10':
                num_ftrs = 512  # without average pool

        elif args.architecture.backbone == 'alexnet':
            num = -2
            if args.experiment.dataset == 'miniimagenet':
                num_ftrs = 256
            elif args.experiment.dataset == 'cub':
                num_ftrs = 9216
        elif args.architecture.backbone.startswith('densenet'):
            num = -1
            if args.experiment.dataset == 'miniimagenet':
                num_ftrs = 4096
            elif args.experiment.dataset == 'cub':
                num_ftrs = 50176
        elif args.architecture.backbone.startswith('vgg'):
            num = -1
            if args.experiment.dataset == 'miniimagenet':
                num_ftrs = 256
            elif args.experiment.dataset == 'cub':
                num_ftrs = 25088
        elif args.architecture.backbone.startswith('squeezenet'):
            num = -1
            if args.experiment.dataset == 'miniimagenet':
                num_ftrs = 12800
            elif args.experiment.dataset == 'cub':
                num_ftrs = 86528
        elif args.architecture.backbone.startswith('googlenet'):
            num = -1
            if args.experiment.dataset == 'miniimagenet':
                num_ftrs = 12800
            elif args.experiment.dataset == 'cub':
                num_ftrs = 1024
        else:
            raise NotImplementedError

        self.features = torch.nn.Sequential(*list(model.children())[:num])
        if self.multi_head:
            self.head = torch.nn.ModuleList()
            for t, n in self.taskcla:
                self.head.append(torch.nn.Sequential(
                    torch.nn.Linear(num_ftrs, n)
                )
            )
        else:
            # num_classes_first_task = self.taskcla[0][1]
            # print ("num_classes_first_taskL ", num_classes_first_task)
            self.head = torch.nn.Linear(num_ftrs, args.experiment.total_num_classes)


        return


    def forward(self,x,task_id=None):
        # inp = x.view_as(x)
        features = self.features(x) # [B, 512, 7, 7]
        h = features.view(x.size(0), -1) # [B 25088]

        if self.multi_head:
            if torch.is_tensor(task_id):
                return torch.stack([self.head[task_id[i]].forward(h[i]) for i in range(h.size(0))])
            else:
                y = []
                for i, _ in self.taskcla:
                    y.append(self.head[i](h))
                # if self.rise:
                #     return self.softmax(y[task_id])
                # else:
                return y[task_id]
        else:
            # if torch.is_tensor(task_id):
            #     return torch.stack([self.head[task_id[i]].forward(h[i]) for i in range(h.size(0))])
            # else:
            return self.head(h)

    def increment_classes(self, task_id):
        assert self.head.out_features == sum([item[1] for item in self.taskcla[:task_id]])

        """Add n classes in the final fc layer"""
        in_features = self.head.in_features
        out_features = self.head.out_features
        weight = self.head.weight.data

        self.head = torch.nn.Linear(in_features, out_features+self.taskcla[task_id][1], bias=False)
        self.head.weight.data[:out_features] = weight

        assert self.head.out_features == sum([item[1] for item in self.taskcla[:task_id+1]])


    def print_model_size(self):
        count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print('Num parameters in the entire model    = %s ' % (self.human_format(count)))


    def human_format(self,num):
        magnitude=0
        while abs(num)>=1000:
            magnitude+=1
            num/=1000.0
        return '%.2f%s'%(num,['','K','M','G','T','P'][magnitude])


class _CustomDataParallel(torch.nn.DataParallel):
    def __init__(self, model):
        super(_CustomDataParallel, self).__init__(model)

    def __getattr__(self, name):
        try:
            return super(_CustomDataParallel, self).__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)