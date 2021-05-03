import os,argparse,time
import numpy as np
from omegaconf import OmegaConf

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import utils
import wandb

tstart=time.time()

# Arguments
parser = argparse.ArgumentParser(description='RRR')

parser.add_argument('--config',  type=str, default='./configs/config_cub_rrr.yaml')
parser.add_argument('--name',  type=str, default='')
parser.add_argument('overrides', nargs='*', help="Any key=svalue arguments to override config values "
                                                "(use dots for.nested=overrides)")
flags =  parser.parse_args()
overrides = OmegaConf.from_cli(flags.overrides)
cfg = OmegaConf.load(flags.config)
args = OmegaConf.merge(cfg, overrides)


########################################################################################################################
# Args -- Data generator
from dataloaders import datagenerator

# Args -- Aporoach
from approaches.rrr import RRR as approach

# Args -- Network
if args.experiment.dataset == 'cifar100':
    from networks import resnet_cifar as network
    # args.architecture.target_layer = "features.layer4.1.conv2"  # resnet_cifar
    args.architecture.target_layer = "m_8_0.3"  # resnet used in itaml
else:
    from networks import resnet as network

if args.architecture.backbone == 'resnet18':
    args.architecture.target_layer = "features.7.1.conv2"
elif args.architecture.backbone == 'densenet121':
    args.architecture.target_layer = "features.0.denseblock4.denselayer16.conv2"
elif args.architecture.backbone == 'alexnet':
    args.architecture.target_layer = "features.0.10"
elif args.architecture.backbone == 'vgg11':
    args.architecture.target_layer = "features.0.18"
elif args.architecture.backbone == 'squeezenet1_1':
    args.architecture.target_layer = "features.0.12.expand3x3"
elif args.architecture.backbone == 'googlenet':
    args.architecture.target_layer = 'features.15.branch4.1.conv'

########################################################################################################################


def run(args, run_id):

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Data loader
    print('Instantiate data generators and model...')
    dataset = datagenerator.DatasetGen(args)
    args.taskcla, args.inputsize = dataset.taskcla, dataset.inputsize
    args.num_classes = dataset.num_classes

    # Network
    net = network.Net(args)

    net.print_model_size()
    if args.device.multi:
        net = network._CustomDataParallel(net)
    net = net.to(device=args.device.name)

    for n,p in net.named_parameters():
        print (n, p.size())

    if args.device.multi:
        args.architecture.target_layer = 'module.'+ args.architecture.target_layer

    # Approach
    appr = approach(net, args, dataset=dataset, network=network)

    # Loop tasks
    perf =np.zeros((len(args.taskcla),len(args.taskcla)),dtype=np.float32)
    avg_rii = np.zeros((args.experiment.ntasks, 2))

    for t,ncla in args.taskcla:

        # Train and test
        appr.train(t, perf)



def main(args):
    utils.print_time(start=True)
    args.path.checkpoint, args.wandb.notes = utils.make_directories(args)

    if args.wandb.log:
        wandb.init(project=args.wandb.project,name=args.wandb.notes,
                   config=args.config,notes=args.wandb.notes,
                   allow_val_change=True)

    utils.save_code(args)

    print('=' * 100)
    print('Arguments =')
    for arg in vars(args):
        print('\t' + arg + ':', getattr(args, arg))
    print('=' * 100)

    for n in range(args.train.num_runs):
        args.seed = n+1

        args.experiment.memory_budget = int(args.experiment.memory_budget)
        args.path.output = 'Run_{}_{}.txt'.format(n+1, args.wandb.notes)

        if args.wandb.log:
            wandb.config.update(args, allow_val_change=True)

        print (">"*30, "Run #", n+1)
        run(args, n)


    print ("All Done! ")
    print('[Elapsed time = {:.1f} min - {:0.1f} hours]'.format((time.time()-tstart)/(60), (time.time()-tstart)/(3600)))
    utils.print_time(start=False)



#######################################################################################################################

if __name__ == '__main__':
    main(args)
