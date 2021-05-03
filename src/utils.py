import os
import numpy as np
from copy import deepcopy
import pickle
import json
import torch
from subprocess import call
import math
from torch.optim.optimizer import Optimizer
########################################################################################################################

def human_format(num):
    magnitude=0
    while abs(num)>=1000:
        magnitude+=1
        num/=1000.0
    return '%.1f%s'%(num,['','K','M','G','T','P'][magnitude])



def report_train(res, e, clock0, clock1):
    # Training performance
    print('| Epoch {:3d}, time={:0.2f}s  | Train loss={:.3f} | Acc={:5.2f}% |'.format(
            e + 1, (clock1 - clock0), res['loss'], res['acc']), end='')

def report_valid(res):
    # Validation performance
    print('|| Task Valid loss={:.3f} | Acc={:5.2f}% |'.format(res['loss'], res['acc']), end='')


########################################################################################################################

def get_model(model):
    return deepcopy(model.state_dict())

########################################################################################################################


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return

def fisher_matrix_diag(task_id,train_loader,model,criterion,args,sbatch=20):
    # Init
    fisher={}
    for n,p in model.named_parameters():
        fisher[n]=0*p.data
    # Compute
    model.train()

    for batch_idx, (x, y, tt) in enumerate(train_loader):
        images = x.to(device=args.device.name, dtype=torch.float)
        targets = y.to(device=args.device.name, dtype=torch.long)
        tt = tt.to(device=args.device.name, dtype=torch.long)

        model.zero_grad()

        # Forward
        output = model.forward(images, tt)
        loss = criterion(output, targets, task_id)
        loss.backward()

        # Get gradients
        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += sbatch * p.grad.data.pow(2)
    # Mean
    for n, _ in model.named_parameters():
        fisher[n] = fisher[n] / images.size(0)
        fisher[n] = torch.autograd.Variable(fisher[n], requires_grad=False)
    return fisher


########################################################################################################################
def get_running_acc_bwt(acc, current_task_id):
    acc = acc[:current_task_id+1,:current_task_id+1]
    acc = np.mean(acc[current_task_id,:])

    return acc


def print_log_acc_bwt(taskcla, acc, output_path, run_id):

    print('*'*100)
    print('Accuracies =')
    for i in range(acc.shape[0]):
        print('\t',end=',')
        for j in range(acc.shape[1]):
            print('{:5.4f}% '.format(acc[i,j]),end=',')
        # print()

    avg_acc = np.mean(acc[acc.shape[0]-1,:])
    print ('ACC: {:5.4f}%'.format(avg_acc))
    # print()


    # print()
    # BWT calculated based on GEM paper (https://arxiv.org/abs/1706.08840)
    gem_bwt = sum(acc[-1]-np.diag(acc))/ (len(acc[-1])-1)
    # BWT calculated based on UCB paper (https://arxiv.org/abs/1906.02425)
    ucb_bwt = (acc[-1] - np.diag(acc)).mean()
    print ('BWT: {:5.2f}%'.format(gem_bwt))
    # print ('BWT (UCB paper): {:5.2f}%'.format(ucb_bwt))

    print('*'*100)
    print('Done!')


    logs = {}
    # save results
    logs['name'] = output_path
    logs['taskcla'] = taskcla
    logs['acc'] = acc
    # logs['loss'] = lss
    logs['gem_bwt'] = gem_bwt
    logs['ucb_bwt'] = ucb_bwt
    logs['rii'] = np.diag(acc)
    logs['rij'] = acc[-1]

    # pickle
    path = os.path.join(output_path, 'logs_run_id_{}.p'.format(run_id))
    with open(path, 'wb') as output:
        pickle.dump(logs, output)

    print ("Log file saved in ", path)
    return avg_acc, gem_bwt


def pprint(acc, u):
    for i in range(u):
        print('\t',end=',')
        for j in range(u):
            print('{:5.4f}% '.format(acc[i,j]),end=',')
        print()


def make_directories(args):
     head = 'mh' if args.architecture.multi_head else 'sh'

     name = '{}_{}_{}_{}_budget{}'.format(args.experiment.dataset,
                                       args.experiment.approach,
                                       args.architecture.backbone,
                                       head,
                                       args.experiment.memory_budget)
     if args.experiment.xai_memory:
         name = '{}_{}_{}_{}_{}_budget{}'.format(args.experiment.dataset,
                                              args.experiment.approach,
                                              args.architecture.backbone,
                                              head,
                                              args.saliency.method,
                                              args.experiment.memory_budget)

     if args.wandb.notes == '':
         args.wandb.notes = name
     else:
        args.wandb.notes = args.wandb.notes + '_' + name


     if not os.path.exists(args.path.checkpoint):
         os.mkdir(args.path.checkpoint)

     args.path.checkpoint = os.path.join(args.path.checkpoint,args.wandb.notes)
     if not os.path.exists(args.path.checkpoint):
         os.mkdir(args.path.checkpoint)

     return args.path.checkpoint, args.wandb.notes


def dump(outfile, obj):

    with open(outfile, "w") as f:
        json.dump(str(obj), f)

def some_sanity_checks(args):
    datasets_tasks = {}
    datasets_tasks['miniimagenet']=[20]
    datasets_tasks['cub']=[20]

    if not args.experiment.ntasks in datasets_tasks[args.experiment.dataset]:
        raise Exception("Chosen number of tasks ({}) does not match with {} experiment".format(args.experiment.ntasks,args.experiment.dataset))

    # Making sure if memory usage is happenning:
    if args.experiment.raw_memory_only and not args.experiment.memory_budget > 0:
        raise Exception("Flags required to use_raw_memory: --use_raw_memory yes --samples n where n>0")

    if args.experiment.xai_memory and not args.experiment.memory_budget > 0:
        raise Exception("Flags required to experiment.xai_memory: --use_raw_memory yes --samples n where n>0")


def save_code(args):
    cwd = os.getcwd()
    des = os.path.join(args.path.checkpoint, 'code') + '/'
    if not os.path.exists(des):
        os.mkdir(des)


    def get_folder(folder):
        return os.path.join(cwd,folder)

    folders = [get_folder(item) for item in ['configs', 'dataloaders', 'networks', 'approaches', 'main.py', 'utils.py']]

    for folder in folders:
        call('cp -rf {} {}'.format(folder, des),shell=True)


# def dump(res, path):
#     with open(path, 'wb') as fp:
#         pickle.dump(res, fp)


def print_time(start=True):
    from datetime import datetime
    from pytz import timezone

    # datetime object containing current date and time
    now = datetime.now(timezone('US/Pacific'))

    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    if start:
        print("Experiment starting at =", dt_string)
    else:
        print ("Jon Finsihed at ", dt_string)


def pprint(t, u, lss, acc):
    print('>>> Test model {} on task {:2d}: loss={:.3f}, acc={:5.1f}% <<<'.format(t, u, lss[t, u], acc[t, u]))




class RAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.buffer = [[None, None, None] for ind in range(10)]
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = self.buffer[int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = group['lr'] * math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = group['lr'] / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
                else:
                    p_data_fp32.add_(-step_size, exp_avg)

                p.data.copy_(p_data_fp32)

        return loss



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        # print ("self.__dict__", self.__dict__)
        fmtstr = '{name} {val' + self.fmt + '}'
        # fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))


    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'



def accuracy(output, target, topk=(1,) ):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    example_images = []
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

    return res


def get_error(output, target):
    pred = output.argmax(dim=1)
    correct = pred.eq(target).float().sum()
    return float((1. - correct / output.size(0)) * 100.)




def get_device(model: torch.nn.Module):
    return next(model.parameters()).device