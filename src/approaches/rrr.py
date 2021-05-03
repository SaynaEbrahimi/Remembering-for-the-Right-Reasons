import sys,time,os
import numpy as np
import torch

import os
import utils

from matplotlib import pyplot as plt
import seaborn as sns; sns.set()
import torch.nn.functional as F
import wandb
import torch.utils.data
from torch.optim.optimizer import Optimizer

from copy import deepcopy
import math
from utils import RAdam

########################################################################################################################

class RRR(object):

    def __init__(self,model,args,dataset,network,clipgrad=10000):
        self.args=args
        self.dataset = dataset
        self.model=model
        self.nepochs=args.train.nepochs
        self.sbatch=args.train.batch_size

        self.lrs = [args.train.lr for _ in range(args.experiment.ntasks)]
        if self.args.experiment.fscil:
            self.lrs[1:] = [self.args.experiment.lr_multiplier * lr for lr in self.lrs[1:]]


        # self.lrs = [item[1] for item in args.lrs]
        self.lrs_exp = [args.saliency.lr for _ in range(args.experiment.ntasks)]
        self.lr_min=[lr/1000. for lr in self.lrs]
        self.lr_factor=args.train.lr_factor
        self.lr_patience=args.train.lr_patience
        self.clipgrad=clipgrad
        self.checkpoint=args.path.checkpoint
        self.device=args.device.name

        # self.args.train.schedule = [20, 30,40]
        self.args.train.schedule = [20, 40, 60]

        self.criterion=torch.nn.CrossEntropyLoss().to(device=self.args.device.name)

        if self.args.saliency.loss == "l1":
            self.sal_loss = torch.nn.L1Loss().to(device=self.args.device.name)
        elif self.args.saliency.loss == "l2":
            self.sal_loss = torch.nn.MSELoss().to(device=self.args.device.name)
        else:
            raise NotImplementedError

        if self.args.train.l1_reg:
            self.l1_reg = torch.nn.L1Loss(reduction='sum')

        self.get_optimizer(task_id=0)
        self.get_optimizer_explanations(task_id=0)



        self.network=network
        self.inputsize=args.inputsize
        self.taskcla=args.taskcla

        self.memory_loaders = {}
        self.test_loader = {}


        self.memory_paths = []
        self.saliency_loaders = None

        if self.args.experiment.raw_memory_only or self.args.experiment.xai_memory:
            self.use_memory = True
        else:
            self.use_memory = False

        # XAI
        if self.args.saliency.method == 'gc':
            print ("Using GradCAM to obtain saliency maps")
            from approaches.explanations import GradCAM as Explain
        elif self.args.saliency.method == 'smooth':
            print ("Using SmoothGrad to obtain saliency maps")
            from approaches.explanations import SmoothGrad as Explain
        elif self.args.saliency.method == 'bp':
            print ("Using BackPropagation to obtain saliency maps")
            from approaches.explanations import BackPropagation as Explain
        elif self.args.saliency.method == 'gbp':
            print ("Using Guided BackPropagation to obtain saliency maps")
            from approaches.explanations import GuidedBackPropagation as Explain
        elif self.args.saliency.method == 'deconv':
            from approaches.explanations import Deconvnet as Explain

        self.explainer = Explain(self.args)


    def get_optimizer(self,task_id, lr=None):
        if lr is None: lr=self.lrs[task_id]

        if (self.args.train.optimizer=="radam"):
            self.optimizer = RAdam(self.model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0)
        elif(self.args.train.optimizer=="adam"):
            self.optimizer= torch.optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08,
                               weight_decay=0.0, amsgrad=False)
        elif(self.args.train.optimizer=="sgd"):
            self.optimizer= torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=0.001)
            self.scheduler_opt = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                        patience=self.lr_patience,
                                                                        factor=self.lr_factor / 10,
                                                                        min_lr=self.lr_min[task_id], verbose=True)



    def adjust_learning_rate(self, epoch):
        if epoch in self.args.train.schedule:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= self.args.train.gamma
            print("Reducing learning rate to ", param_group['lr'])


    def get_optimizer_explanations(self, task_id, lr=None):
        if lr is None: lr=self.lrs_exp[task_id]

        if self.args.train.optimizer=="sgd":
            self.optimizer_explanations = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=self.args.train.wd)
            self.scheduler_exp_opt = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_explanations,
                                                                         patience=self.lr_patience,
                                                                          factor=self.lr_factor/10,
                                                                          min_lr=self.lr_min[task_id], verbose=True)
        elif(self.args.train.optimizer=="adam"):
            self.optimizer_explanations= torch.optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08,
                               weight_decay=0.0, amsgrad=False)

        elif (self.args.train.optimizer=="radam"):
            self.optimizer_explanations = RAdam(self.model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0)




    def train(self,task_id,performance):
        # if task_id > 0 and not self.args.architecture.multi_head:
        #     self.model.increment_classes(task_id)
        #     self.model  = self.model.to(self.device)

        print('*'*100)
        data_loader = self.dataset.get(task_id=task_id)
        self.test_loader[task_id] = data_loader['test']

        print(' '*85, 'Run #{:2d} - Dataset {:2d} ({:s})'.format(self.args.seed+1, task_id+1,data_loader['name']))

        print('*'*100)

        best_loss=np.inf
        if self.args.train.nepochs > 1:
            best_model=utils.get_model(self.model)
        lr = self.lrs[task_id]
        lr_exp = self.lrs_exp[task_id]
        patience=self.lr_patience
        best_acc = 0

        self.get_optimizer(task_id=task_id)
        self.get_optimizer_explanations(task_id=task_id)

        # Loop epochs

        for e in range(self.nepochs):
            self.epoch = e

            # Train
            clock0=time.time()
            self.train_epoch(data_loader['train'], task_id)
            clock1=time.time()

            # Valid
            if self.args.train.pc_valid > 0:
                valid_res = self.eval(data_loader['valid'], task_id, set_name='valid')
                utils.report_valid(valid_res)
            else:
                valid_res = self.eval(data_loader['test'], task_id, set_name='test')
                print ("Epoch {}/{} | Test acc {}: {:.2f}".format(e+1, self.nepochs, task_id, valid_res['acc']))

            if self.args.wandb.log:
                wandb.log({"Best Acc on Task {}".format(task_id): valid_res['acc']})

            if (self.args.optimizer == "sgd"):
                self.scheduler_opt.step(valid_res['loss'])
                self.scheduler_exp_opt.step(valid_res['loss'])

            is_best = valid_res['acc'] > best_acc
            if is_best:
                best_model = utils.get_model(self.model)
            best_acc = max(valid_res['acc'], best_acc)



        # Restore best validation model
        if self.args.train.nepochs > 1:
            self.model.load_state_dict(deepcopy(best_model))

        self.save_model(task_id, deepcopy(self.model.state_dict()))

        if task_id == 1 and self.args.experiment.xai_memory:
            self.compute_memory()

        if self.use_memory and task_id < self.args.experiment.ntasks:
            self.update_memory(task_id)


    def train_epoch(self,train_loader,task_id):
        self.adjust_learning_rate(self.epoch)

        self.model.train()

        if task_id > 0 and self.args.experiment.xai_memory:
            for idx, (data, target, sal, tt, _) in enumerate(self.saliency_loaders):

                x = data.to(device=self.device, dtype=torch.float)
                s = sal.to(device=self.device, dtype=torch.float)

                explanations, self.model , _, _ = self.explainer(x, self.model, task_id)

                self.saliency_size = explanations.size()

                # To make predicted explanations (Bx7x7) same as ground truth ones (Bx1x7x7)
                sal_loss = self.sal_loss(explanations.view_as(s), s)
                sal_loss *= self.args.saliency.regularizer

                if self.args.wandb.log:
                    wandb.log({"Saliency loss": sal_loss.item()})

                try:
                    sal_loss.requires_grad = True
                except:
                    continue

                self.optimizer_explanations.zero_grad()
                sal_loss.backward(retain_graph=True)
                self.optimizer_explanations.step()

        # Loop batches
        for batch_idx, (x, y, tt) in enumerate(train_loader):

            images = x.to(device=self.device, dtype=torch.float)
            targets = y.to(device=self.device, dtype=torch.long)
            tt = tt.to(device=self.device, dtype=torch.long)
            # Forward
            if self.args.architecture.multi_head:
                output=self.model.forward(images, tt)
            else:
                output = self.model.forward(images)

            loss=self.criterion(output,targets)

            # L1 regularize
            if self.args.train.l1_reg:
                reg_loss = self.l1_regularizer()
                factor = self.args.train.l1_reg_factor
                loss += factor * reg_loss

            loss *= self.args.train.task_loss_reg


            # Backward
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)

            # Apply step
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.clipgrad)
            self.optimizer.step()


    def eval(self,data_loader, task_id, set_name='valid'):
        total_loss=0
        total_acc=0
        total_num=0
        self.model.eval()
        res={}
        old_tasks_loss, sal_loss = 0, 0

        # Loop batches
        with torch.no_grad():

            for batch_idx, (x, y, tt) in enumerate(data_loader):
                # Fetch x and y labels
                images=x.to(device=self.device, dtype=torch.float)
                targets=y.to(device=self.device, dtype=torch.long)
                tt=tt.to(device=self.device, dtype=torch.long)

                # Forward
                if self.args.architecture.multi_head:
                    output = self.model.forward(images, tt)
                else:
                    output = self.model.forward(images)

                loss = self.criterion(output,targets)
                _, pred=output.max(1)
                hits = (pred==targets).float()

                # Log
                total_loss += loss
                total_acc += hits.sum().item()
                total_num += targets.size(0)

        res['loss'], res['acc'] = total_loss/(batch_idx+1), 100*total_acc/total_num
        res['size'] = self.loader_size(data_loader)

        return res


    def test(self, model, test_id, model_id=None):
        total_loss=0
        total_acc=0
        total_num=0
        model.eval()
        res={}

        # Loop batches
        with torch.no_grad():

            for batch_idx, (x, y, tt) in enumerate(self.test_loader[test_id]):
                # Fetch x and y labels
                images=x.to(device=self.device, dtype=torch.float32)
                targets=y.to(device=self.device, dtype=torch.long)

                # Forward
                # output= model.forward(images,test_id)
                if self.args.architecture.multi_head:
                    output = self.model.forward(images, test_id)
                else:
                    output = self.model.forward(images)

                _, pred = output.max(1)

                loss=self.criterion(output,targets)
                hits=(pred==targets).float()

                # Log
                total_loss+=loss
                total_acc+=hits.sum().item()
                total_num+=targets.size(0)

        res['loss'], res['acc'] = total_loss/(batch_idx+1), 100*total_acc/total_num


        return res['loss'], res['acc']


    def update_memory(self, task_id):
        start = time.time()
        # Get memory set for each task seen so far with the updated samples per class and return new spc
        self.dataset.get_memory_sets(task_id)
        midway = time.time()
        print('[Storing memory time = {:.1f} min ]'.format((midway - start) / (60)))


        if self.args.experiment.xai_memory:
            self.update_saliencies(task_id)
            self.saliency_loaders = self.dataset.generate_evidence_loaders(task_id)
            # Be careful if you comment out the sanity check below. It extremely slows down the training
            # self.check_saliency_loaders(task_id)
        print('[Storing saliency time = {:.1f} min ]'.format((time.time() - midway) / (60)))

    def update_saliencies(self, task_id):
        save_images = False
        ims = [] if save_images else None
        images, preds = [], []

        # Generate saliency for images from the last seen task
        memory_path = os.path.join(self.args.path.checkpoint, 'memory', 'mem_{}.pth'.format(task_id))
        memory_set = torch.load(memory_path)
        num_samples = len(memory_set)
        single_loader = torch.utils.data.DataLoader(memory_set, batch_size=1,
                                                    num_workers=self.args.device.workers,
                                                    shuffle=False)

        saliencies, predictions = [], []
        for idx, (img, y, tt) in enumerate(single_loader):

            img = img.to(self.args.device.name)
            sal, self.model, _, _ = self.explainer(img, self.model, task_id)
            if self.args.architecture.multi_head:
                output = self.model.forward(img, task_id)
            else:
                output = self.model.forward(img)

            _, pred = output.max(1)

            saliencies.append(sal)
            predictions.append(pred)

        sal_path = os.path.join(self.args.path.checkpoint, 'memory', 'sal_{}.pth'.format(task_id))
        pred_path = os.path.join(self.args.path.checkpoint, 'memory', 'pred_{}.pth'.format(task_id))
        torch.save(saliencies, sal_path)
        torch.save(predictions, pred_path)

        if not self.args.experiment.fscil:
            # Reduce previous saliencies
            for t in range(task_id):
                # Read the stored saliency file
                sal_path = os.path.join(self.args.path.checkpoint, 'memory', 'sal_{}.pth'.format(t))
                saliencies = torch.load(sal_path)
                before = len(saliencies)

                pred_path = os.path.join(self.args.path.checkpoint, 'memory', 'pred_{}.pth'.format(t))
                predictions = torch.load(pred_path)

                # Extract the required number of samples and save them again
                saliencies = saliencies[:num_samples]
                after = len(saliencies)
                torch.save(saliencies, sal_path)
                print ("Reduced saliencies for task {} from {} to {}".format(t, before, after))

                predictions = predictions[:num_samples]
                torch.save(predictions, pred_path)



    def l1_regularizer(self):
        reg_loss = 0
        for param in self.model.parameters():
            target = torch.zeros_like(param)
            reg_loss += self.l1_reg(param, target)
        return reg_loss


    def save_model(self,t,best_model):
        torch.save({'model_state_dict': best_model,
                    }, os.path.join(self.checkpoint, 'model_run_id_{}_task_id_{}.pth.tar'.format(self.args.seed,t)))


    def loader_size(self,data_loader):
        return data_loader.dataset.__len__()


    def load_model(self, task_id):
        net=self.network.Net(self.args).to(device=self.args.device.name)
        # net = self.network._CustomDataParallel(net, self.args.device.name_ids)
        if self.args.device.multi:
            net = torch.nn.DataParallel(net)
        checkpoint=torch.load(os.path.join(self.checkpoint, 'model_run_id_{}_task_id_{}.pth.tar'.format(self.args.seed,task_id)))
        net.load_state_dict(checkpoint['model_state_dict'])
        net = net.to(device=self.args.device.name)
        # net = self.network._CustomDataParallel(net, self.args.device.name_ids)
        return net

    def load_singlehead_model(self,current_model_id):
        return self.model

    def load_multihead_model(self, test_id, current_model_id):
        # Load a previous model
        old_model=self.network.Net(self.args)
        if self.args.device.multi:
            old_model = torch.nn.DataParallel(old_model)
        checkpoint=torch.load(os.path.join(self.checkpoint, 'model_run_id_{}_task_id_{}.pth.tar'.format(self.args.seed,test_id)))
        old_model.load_state_dict(checkpoint['model_state_dict'])

        # Load a current model
        current_model=self.network.Net(self.args)
        if self.args.device.multi:
            current_model = torch.nn.DataParallel(current_model)
        checkpoint=torch.load(os.path.join(self.checkpoint, 'model_run_id_{}_task_id_{}.pth.tar'.format(self.args.seed,current_model_id)))
        current_model.load_state_dict(checkpoint['model_state_dict'])

        # Change the current_model head with the old head
        if self.args.device.multi:
            old_head=deepcopy(old_model.module.head.state_dict())
            current_model.module.head.load_state_dict(old_head)
        else:
            old_head = deepcopy(old_model.head.state_dict())
            current_model.head.load_state_dict(old_head)

        current_model=current_model.to(self.args.device.name)
        return current_model



    def load_checkpoint(self, task_id):
        print("Loading checkpoint for task {} ...".format(task_id))

        # Load a previous model
        net=self.network.Net(self.args)
        net = net.to(device=self.args.device.name)

        checkpoint=torch.load(os.path.join(self.checkpoint, 'model_run_id_{}_task_id_{}.pth.tar'.format(self.args.seed,task_id)))
        net.load_state_dict(checkpoint['model_state_dict'])
        net=net.to(device=self.args.device.name)

        return net


    def check_saliency_loaders(self, task_id):

        path = os.path.join(self.args.path.checkpoint, 'memory')
        for idx, (images, targets, saliencies, tt, preds) in enumerate(self.saliency_loaders):
            # loop over batch
            for i in range(len(images)):

                fig = plt.figure(dpi=100)
                # fig.subplots_adjust(hspace=0.01, wspace=0.01)

                img = images[i].unsqueeze(0)  # [1, 3, 224, 224]
                saliency = saliencies[i].unsqueeze(0)  # shape: [1, 1, 7, 7]
                pred = preds[i]  # shape: [1, 7, 7]
                target = targets[i] #
                task = tt[i]
                # saliency = saliency.unsqueeze(0)  # shape:
                image_size = img.shape[2:] # [32x32]
                saliency = F.interpolate(saliency, size=image_size, mode="bilinear", align_corners=False)

                B, C, H, W = saliency.shape
                saliency = saliency.view(B, -1)
                saliency_max = saliency.max(dim=1, keepdim=True)[0]
                saliency_max[torch.where(saliency_max == 0)] = 1.  # prevent divide by 0
                saliency -= saliency.min(dim=1, keepdim=True)[0]
                saliency /= saliency_max
                saliency = saliency.view(B, C, H, W)
                saliency = saliency.squeeze(0).squeeze(0)
                saliency = saliency.detach().cpu().numpy()

                img = img.squeeze(0)
                img = img.cpu().numpy().transpose((1, 2, 0))  # (224, 224, 3)
                mean = np.array(self.dataset.mean)
                std = np.array(self.dataset.std)
                img = std * img + mean
                img = np.clip(img, 0, 1)

                # plt.subplot(1, 10, idx + 1)
                plt.axis('off')
                result = 'Correct' if pred.item() == target else "Wrong"
                plt.title('{} | Pred:{}, Truth:{}, Task:{}'.format(result, pred.item(), target, task), fontsize=9)

                plt.imshow(img)
                plt.savefig(os.path.join(path, 'Img-batch-{}-ID-{}-task-{}.png'.format(idx, i, task_id)))
                plt.imshow(saliency, cmap='jet', alpha=0.5)
                plt.colorbar(fraction=0.046, pad=0.04)

                fig.tight_layout()
                plt.savefig(os.path.join(path, 'Sal-batch-{}-ID-{}-task-{}.png'.format(idx, i, task_id)))
                plt.close()





    def visualize(self, images, saliencies, task_id, preds, y, path):

        # fig = plt.figure(figsize=(10, 2), dpi=500)
        # fig.subplots_adjust(hspace=0.01, wspace=0.01)

        for idx in range(len(saliencies)):
            saliency, img = saliencies[idx], images[idx]

            # img = img.unsqueeze(0)  # [1, 3, 224, 224]
            saliency = saliency.unsqueeze(0)  # shape: [1, 7, 7]
            # saliency = saliency.unsqueeze(0)  # shape: [1, 1, 7, 7]

            image_size = img.shape[2:]
            # print (saliency.size(), img.size(), image_size)
            saliency = F.interpolate(saliency, size=image_size, mode="bilinear", align_corners=False)

            B, C, H, W = saliency.shape
            saliency = saliency.view(B, -1)
            saliency_max = saliency.max(dim=1, keepdim=True)[0]
            saliency_max[torch.where(saliency_max == 0)] = 1.  # prevent divide by 0
            saliency -= saliency.min(dim=1, keepdim=True)[0]
            saliency /= saliency_max
            saliency = saliency.view(B, C, H, W)
            saliency = saliency.squeeze(0).squeeze(0)
            saliency = saliency.detach().cpu().numpy()

            img = img.squeeze(0)
            img = img.cpu().numpy().transpose((1, 2, 0))  # (224, 224, 3)
            mean = np.array(self.dataset.mean)
            std = np.array(self.dataset.std)
            img = std * img + mean
            img = np.clip(img, 0, 1)

            fig = plt.figure(dpi=500)
            # plt.subplot(1,1, 1)
            plt.axis('off')

            result = 'Correct' if preds[idx].item() == y[idx].item() else "Wrong"
            plt.title('{}-Pred:{},Truth:{}'.format(result, preds[idx].item(), y[idx].item()), fontsize=7)
            plt.imshow(img)
            plt.imshow(saliency, cmap='jet', alpha=0.5)
            plt.colorbar(fraction=0.046, pad=0.04)
            fig.tight_layout()

            print('saving to: {}'.format(os.path.join(path, 'sal-task-{}-mem-{}.png'.format(task_id, idx))))
            plt.savefig(os.path.join(path, 'sal-task-{}-mem-{}.png'.format(task_id, idx)))
            plt.close()

    def compute_memory(self):
        ssize = 1
        print ("***"*200)
        print ("saliency_size", self.saliency_size)
        for s in self.saliency_size:
            ssize *= s
        saliency_memory = 4 * self.args.experiment.memory_budget * ssize

        ncha, size, _ = self.args.inputsize
        image_size = ncha * size * size

        samples_memory = 4 * self.args.experiment.memory_budget * image_size
        count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print('Num parameters in the entire model    = %s ' % (utils.human_format(count)))
        architecture_memory = 4 * count

        print("-------------------------->  Saliency memory size: (%sB)" % utils.human_format(saliency_memory))
        print("-------------------------->  Episodic memory size: (%sB)" % utils.human_format(samples_memory))
        print("------------------------------------------------------------------------------")
        print("                             TOTAL:  %sB" % utils.human_format(
            architecture_memory+samples_memory+saliency_memory))




