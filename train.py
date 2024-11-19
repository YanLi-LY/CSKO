import torch
import numpy as np
import argparse
import random
import os
import logging
import sys
import time
import yaml

from utils.Trainer import Trainer
from set_seed import *


def main():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--data_dir', type=str, default='Data/', help='dataset dir')
    parser.add_argument('--dataset_name', type=str, default='CIFAR100', help='dataset name: TinyImageNet, CIFAR100')
    parser.add_argument('--dataset_type', type=str, default='cl', help='type of preprocessing dataset')
    parser.add_argument('--loader_type', type=str, default='normal', help='type of preprocessing dataloader')

    # dataset setting(class-division, way, shot)
    parser.add_argument('--base_class', type=int, default=50, help='number of the first phase class')
    parser.add_argument('--per_class', type=int, default=5, help='class number of per task')

    # model option
    parser.add_argument('--model_name', type=str, default='my', help='model name (default: my)')
    parser.add_argument('--model_path', type=str, default='./Pretrained_Model/model_CIFAR100_10.pth', help='model path: model_CIFAR100_10.pth')
    parser.add_argument('--pretrained', default=False, help='whether the backbone needs pre-training')
    parser.add_argument('--backbone_name', type=str, default='resnet18_no1', help='backbone name (default: resnet18)')
    parser.add_argument('--classifier', type=str, default='fc_IL_base', help='classifier name')

    # backbone
    parser.add_argument('--mode', type=str, default='normal', choices=['parallel_adapters', 'normal', 'film'], help='type of adapters (default: normal)')

    # gpu option
    parser.add_argument('--device', type=str, default='cuda:0', help='use which gpu to train')

    # loss option
    parser.add_argument('--loss_type', type=str, default='ce', help='type of loss (default: ce)')

    # hyper option
    parser.add_argument('--session', type=int, default=6, metavar='N', help='training session (default:9)')
    parser.add_argument('--base_epochs', type=int, default=100, metavar='N', help='base epochs')
    parser.add_argument('--new_epochs', type=int, default=60, metavar='N', help='base epochs')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='batch_size')

    # optimizer option
    parser.add_argument('--opt', type=str, default='opt1', choices=['opt1', 'opt2', 'opt3'], help='type of learnable para (default: opt1)')
    parser.add_argument('--optim', type=str, default='adam', choices=['sgd', 'adam'], help='type of optimizer (default: sgd)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
    parser.add_argument('--lr_new', type=float, default=0.0001, metavar='LR', help='learning rate ')
    parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='M', help='w-decay (default: 5e-4)')

    # evaluation options
    parser.add_argument('--val', default=True, help='val mode')

    # saver
    parser.add_argument('--result_save_dir', type=str, default='result')

    # my params
    parser.add_argument('--channel_sparsity', type=float, default=0.75)
    parser.add_argument('--thres', type=int, default=5)
    parser.add_argument('--train_layers_path', type=str, default='config.yaml')
    parser.add_argument('--adapter_layers_path', type=str, default='adapter.yaml')
    parser.add_argument('--GF_cfg_path', type=str, default='./utils/model/GF_cfg.yaml')

    args = parser.parse_args()
    args.data_path = os.path.join(args.data_dir, args.dataset_name)
    args.tasks = args.session - 1 
    args.all_class = args.base_class + args.per_class * args.tasks 


    if not os.path.exists(args.result_save_dir):
        os.makedirs(args.result_save_dir)
    args.result_save_name = f'{args.dataset_name}_{args.base_class}_{args.per_class}_{args.model_name}({args.backbone_name})'
    args.result_save_path = os.path.join(args.result_save_dir, args.result_save_name)
    if not os.path.exists(args.result_save_path):
        os.makedirs(args.result_save_path)
    now_time = str(time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()))
    logging.basicConfig(level=logging.INFO, filename=f'{args.result_save_path}/log_{args.dataset_name}_{args.base_class}_{args.per_class}_{args.model_name}({args.backbone_name}_{now_time}.log', format='%(message)s')
    logging.getLogger()

    for arg in vars(args):
        logging.info(f'{arg}: {getattr(args, arg)}')

    with open(args.train_layers_path, 'r') as f:
        config = yaml.safe_load(f)
    for key, value in config.items():
        setattr(args, key, value)

    with open(args.adapter_layers_path, 'r') as f:
        config = yaml.safe_load(f)
    for key, value in config.items():
        setattr(args, key, value)

    with open(args.GF_cfg_path, 'r') as f:
        config = yaml.safe_load(f)
    for key, value in config.items():
        setattr(args, key, value)

    trainer = Trainer(args)
    for session in range(args.session): 
        trainer.training(session)
        trainer.validation(session, end_flag=True)

if __name__ == "__main__":
    main()
