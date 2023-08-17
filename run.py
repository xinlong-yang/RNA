from ast import arg
import torch
import torch.nn as nn
import argparse
import os
import numpy as np
import random
from loguru import logger
from torch.cuda.amp import autocast
import rna
from model_loader import load_model
import warnings
from torch.cuda.amp import autocast
warnings.filterwarnings("ignore")

def seed_torch(seed=2022):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed) # numpy
    torch.manual_seed(seed) # torch
    torch.cuda.manual_seed(seed) # cuda
    torch.backends.cudnn.deterministic = True  

multi_labels_dataset = [
    'nus-wide-tc-10',
    'nus-wide-tc-21',
    'flickr25k',
    'coco'
]

num_features = {
    'alexnet': 4096,
    'vgg16': 4096,
}


def run():
    # Load configuration
    seed_torch()
    seed= 2022
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    args = load_config()
    logger.add(os.path.join('logs', '{}_{}_{}_{}_{}_{}.log'.format(
        args.method, args.tag, args.noisy_rate,args.source.split('/')[-1].split('.')[0], args.target.split('/')[-1].split('.')[0], args.code_length
    )), rotation="500 MB", level="INFO") 
    logger.info(args)

    if args.tag == 'OfficeHome':
        from officehome import load_data
        args.num_class = 65
    elif args.tag == 'OFFICE31':
        from office31 import load_data
        args.num_class = 31
    elif args.tag == 'm_u':
        from mnist2usps import load_data
        args.num_class = 10

    # Load dataset
    query_dataloader, train_s_dataloader, train_t_dataloader, retrieval_dataloader \
        = load_data(args.source, args.target,args.batch_size,args.num_workers, args.noisy_rate, args.noise_type)

    # if train
    if args.train:
        rna.train(
            train_s_dataloader,
            train_t_dataloader,
            query_dataloader,
            retrieval_dataloader,
            args.code_length,
            args.max_iter,
            args.arch,
            args.lr,
            args.device,
            args.verbose,
            args.topk,
            args.num_class,
            args.evaluate_interval,
            args.source,
            args.target,
            # args.method,
            args.tag,
            args.lamda,
            args.tau
        )
    elif args.evaluate:
        model = load_model(args.arch, args.code_length)
        #model = nn.DataParallel(model,device_ids=[0,1,2])
        model_checkpoint = torch.load('./checkpoints/resume_64.t')
        model.load_state_dict(model_checkpoint['model_state_dict'])
        mAP = rna.evaluate(
            model,
            query_dataloader,
            retrieval_dataloader,
            args.code_length,
            args.device,
            args.topk,
            )

    else:
        raise ValueError('Error configuration, please check your config, using "train", "resume" or "evaluate".')


def load_config():
    """
    Load configuration.

    Args
        None

    Returns
        args(argparse.ArgumentParser): Configuration.
    """
    parser = argparse.ArgumentParser(description='cdan_PyTorch')
    # 【office】
    parser.add_argument('--tag', type=str, default='OFFICE31', help="Tag")
    parser.add_argument('--source', type=str, default='/hdd/DataSet/OFFICE31/amazon_list.txt', help="The source dataset")
    parser.add_argument('--target', type=str, default='/hdd/DataSet/OFFICE31/dslr_list.txt', help="The target dataset")
    parser.add_argument('--num_class', default=31, type=int,
                        help='Number of clusters in Spectral Clusting(default:31)')

    # usps2mnist
    # parser.add_argument('--tag', type=str, default='m_u', help="Tag")
    # parser.add_argument('--target', type=str, default='/hdd/DataSet/mnist/mnist.txt', help="The source dataset")
    # parser.add_argument('--source', type=str, default='/hdd/DataSet/usps/usps.txt', help="The target dataset")
    # parser.add_argument('--num_class', default=10, type=int,
    #                     help='Number of clusters in Spectral Clusting(default:10)')


    # Office-Home
    # parser.add_argument('--tag', type=str, default='OfficeHome', help="Tag")
    # parser.add_argument('--source', type=str, default='/hdd/DataSet/OfficeHome/Art.txt', help="The source dataset")
    # parser.add_argument('--target', type=str, default='/hdd/DataSet/OfficeHome/Clipart.txt', help="The target dataset")
    # parser.add_argument('--num_class', default=65, type=int, help='Number of clusters in Spectral Clusting(default:70)')
    

    # Bit length
    parser.add_argument('-n', '--noisy_rate', default=0.6, type=float,
                        help='noisy rate.(default: 0.2)')

    parser.add_argument('-lm', '--lamda', default=1, type=float,
                        help='noisy rate.(default: 0.2)')

    parser.add_argument('-ta', '--tau', default=0.3, type=float,
                        help='noisy rate.(default: 0.2)')

    parser.add_argument('-nt', '--noise_type', default='symmetric', type=str,
                        help='noisy type.')

    parser.add_argument('-m', '--method', default='RNA', type=str,
                        help='method to train the model.(default: rna)')
                        
    parser.add_argument('-c', '--code_length', default=96, type=int,
                        help='Binary hash code length.(default: 64)')
    # R
    parser.add_argument('-k', '--topk', default=50000, type=int,
                        help='Calculate map of top k.(default: -1)')
    # max-itertion
    parser.add_argument('-T', '--max-iter', default=40, type=int,
                        help='Number of iterations.(default: 40)')

    parser.add_argument('-l', '--lr', default=1e-3, type=float,
                        help='Learning rate.(default: 1e-3)')

    parser.add_argument('-w', '--num-workers', default=os.cpu_count(), type=int,
                        help='Number of loading data threads.(default: 1)')

    parser.add_argument('-b', '--batch-size', default=18, type=int,
                        help='Batch size.(default: 24)')

    parser.add_argument('-a', '--arch', default='vgg16', type=str,
                        help='CNN architecture.(default: vgg16)')
    # about log
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print log.')
    parser.add_argument('--train', action='store_true',
                        help='Training mode.')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluation mode.')
    parser.add_argument('-g', '--gpu', default=1, type=int,
                        help='Using gpu.(default: False)')
    # 
    parser.add_argument('-e', '--evaluate-interval', default=2, type=int,
                        help='Interval of evaluation.(default: 500)')
    parser.add_argument('--temperature', default=0.5, type=float,
                        help='Hyper-parameter in SimCLR .(default:0.5)')
    parser.add_argument('--warmup_epoch', default=1, type=float,
                        help='warmup.(default:5)')
    

    args = parser.parse_args()

    # GPU
    if args.gpu is None:
        args.device = torch.device("cpu")
    else:
        args.device = torch.device("cuda:%d" % args.gpu)
        torch.cuda.set_device(args.gpu)
    return args


if __name__ == '__main__':
    run()
