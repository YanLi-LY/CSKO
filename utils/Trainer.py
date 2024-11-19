import copy
import sys
import time
import numpy as np

import psutil
from tqdm import tqdm
import os
import logging

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torchvision import transforms

from utils.model import prepare_model
from utils.data import datatypes
from utils.data import loadertypes
from utils.model_para import filter_para
from utils.loss import MyLosses
from utils.evaluate import Evaluator
from torch.utils.data import DataLoader
from utils.model import backbones
from utils.data.data_manager_tiny import DataManager
from utils.adam_transform import Adam

from collections import defaultdict

from functools import reduce
from utils.model.gradient_filter import add_grad_filter

class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = args.device
        if self.args.dataset_name == 'CIFAR100':
            prepare_data = datatypes[args.dataset_type]
            self.dataset = prepare_data(args.dataset_name, args)
        self.dataloader = loadertypes[args.loader_type]

        self.label_per_task = []
        base_class_list = list(np.array(range(args.base_class)))
        self.label_per_task.append(base_class_list)
        for task_id in range(args.tasks):
            task_list = list(np.array(range(args.per_class)) + args.per_class * task_id + args.base_class)
            self.label_per_task.append(task_list)

        model = prepare_model(self.args)
        model = model.to(self.device)
        self.model = model

        self.criterion = MyLosses(weight=None, args=self.args).build_loss(mode=args.loss_type)

        self.evaluator = Evaluator(args.all_class, self.device)

        self.old_model = None

        self.best_pred = 0.0
        self.test_out = None
        self.ii = 0

        self.acc_history = []
        self.forget_history = []

        self.covar_matirx = defaultdict(dict)
        self.eigens = defaultdict(dict)
        self.transforms = defaultdict(dict)

        self.sparsed_channel_indices = defaultdict(dict)
        self.importance = defaultdict(dict)
        self.masked_covar_matrix = defaultdict(dict)

    def training(self, session):
        if session == 0:
            session_class_last = 0
        else:
            session_class_last = self.args.base_class + self.args.per_class * (session - 1)
        session_class = self.args.base_class + self.args.per_class * session
        if self.args.dataset_name == 'CIFAR100':
            classes = [session_class_last, session_class]
            train_dataset = self.dataset[0]
            train_dataset.getTrainData(classes)
        elif self.args.dataset_name == 'TinyImageNet':
            class_set = list(range(200))
            if session == 0:
                classes = class_set[:self.args.base_class]
            else:
                classes = class_set[session_class_last: session_class]
            train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            data_manager = DataManager(self.args)
            train_dataset = data_manager.get_dataset(train_transform, index=classes, train=True)
        train_loader = self.dataloader(train_dataset, self.args)

        if session == 0:
            epochs = self.args.base_epochs
            lr = self.args.lr
        else:
            epochs = self.args.new_epochs
            lr = self.args.lr_new

        if session == 0 and self.args.val:
            self.model.eval()
            para = torch.load(self.args.model_path)
            para_dict = para['state_dict']
            para_dict_re = self.structure_reorganization(para_dict)
            model_dict = self.model.state_dict()
            state_dict = {k: v for k, v in para_dict_re.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            self.model.load_state_dict(model_dict)
        else:
            if session > 0:
                if session == 1:
                    self.model.eval()
                    backbone = backbones[self.args.backbone_name](mode='parallel_adapters', args=self.args).to(self.device)
                    model_dict = backbone.state_dict()
                    para_dict = self.model.backbone.state_dict()
                    state_dict = {k: v for k, v in para_dict.items() if k in model_dict.keys()}
                    model_dict.update(state_dict)
                    backbone.load_state_dict(model_dict)
                    self.model.backbone = backbone

                    model_dict = self.model.state_dict()
                    for k, v in model_dict.items():
                        if 'adapter' in k:
                            k_conv3 = k.replace('adapter', 'conv')
                            model_dict[k] = model_dict[k_conv3][:, :, 1:2, 1:2].clone()
                            model_dict[k_conv3][:, :, 1:2, 1:2] = 0
                    self.model.load_state_dict(model_dict)

            self.model.classifier.Incremental_learning(session_class)
            self.model = self.model.to(self.device)

            if session == 1:
                if self.args.if_gradient_filter:
                    self.register_filter(self.args.filter_install_cfgs)
            self.model.fix_backbone_adapter()

            if session > 0 and session < self.args.session:
                self.model.eval()
                self.model.zero_grad()
                self.compute_channel_importance_mask(self.model, train_loader)
            
            optim_para_svd = filter_para(self.model, self.args, lr, svd=True)
            optim_para = filter_para(self.model, self.args, lr, svd=False)
            self.normal_optimizer = torch.optim.Adam(optim_para, weight_decay=self.args.weight_decay)
            self.optimizer = Adam(**optim_para_svd, weight_decay=self.args.weight_decay)
            self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=45, gamma=0.1)

            self.mask_covar_matrix()
            self.optimizer.get_eigens(self.masked_covar_matrix)
            self.optimizer.get_transforms()
            self.optimizer.get_sparsed_channel_indices(self.sparsed_channel_indices)

            self.model.train()
            
            for epoch in range(1, epochs+1):
                tbar = tqdm(train_loader)
                train_loss = 0.0
                for i, sample in enumerate(tbar):
                    query_image, query_target = sample
                    query_image, query_target = query_image.to(self.device), query_target.to(self.device)

                    self.optimizer.zero_grad()
                    if session > 0:
                        loss = self._compute_loss(query_image, query_target, session_class_last)
                        loss.backward()
                        self.optimizer.step()
                    else:
                        loss = self._compute_loss_pretrain(query_image, query_target, session_class_last)
                        loss.backward()
                        self.normal_optimizer.step()
                    

                    train_loss += loss.item()
                    tbar.set_description('Epoch: %d, Train loss: %.3f' % (epoch, train_loss / (i + 1)))
                self.scheduler.step()

                if epoch % 10 == 0:
                    self.validation(session, epoch=epoch)
                    self.model.train()

        if session >= 0 and session < self.args.session - 1:
            self.protoSave(self.model, train_loader, session)


        if session == self.args.session-1:
            self.model.eval()
            with torch.no_grad():
                self.afterTrain(session)

        if session >= 0 and session < self.args.session-1:
            self.model.eval()
            with torch.no_grad():
                self.compute_covariance(train_loader, session)

    def register_filter(self, filter_install_cfgs):
        for cfg in filter_install_cfgs: 
            assert 'filter_layer' in cfg.keys()
            layer_path = cfg['filter_layer'].split('.')
            target = reduce(getattr, layer_path, self.model.backbone)
            upd_layer = add_grad_filter(target, cfg)
            parent = reduce(getattr, layer_path[:-1], self.model.backbone)
            setattr(parent, layer_path[-1], upd_layer)

    def compute_channel_importance_mask(self, model, train_loader):
        for x, y in train_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            output = model.forward(x)
            loss_cls = nn.CrossEntropyLoss(reduce=False)(output / 0.1, y)
            loss_cls = torch.mean(loss_cls, dim=0)
            loss_cls.backward()

        grads = {}
        for name, param in model.named_parameters():
            if 'weight' in name and name.rsplit('.', 1)[0] in self.args.adapter_layers:
                conv3_name = name.replace('adapter', 'conv')
                if param.grad is not None:
                    grad = param.grad.data.clone()
                else:
                    grad = 0
                if conv3_name not in grads.keys():
                    grads[conv3_name] = 0
                grads[conv3_name] += grad.abs()

        torch.cuda.empty_cache()

        self.importance = grads.copy()
        grads.clear()

        for name, value in self.importance.items():
            importance_mean = value.mean(dim=0).squeeze() 
            num_sparsed_channel = int(self.args.channel_sparsity * importance_mean.size(0))
            top_values, top_indives = torch.topk(importance_mean, k=num_sparsed_channel)
            self.sparsed_channel_indices[name] = top_indives

    def standardize(self, grad):
        if torch.all(grad==0):
            pass
        else:
            grad_shape = grad.shape
            grad = grad.view(-1)
            ret = (grad - grad.mean()) / grad.std()
            ret = ret.view(*grad_shape)
            ret = torch.tanh(ret)
            return ret

    def compute_covariance(self, train_loader, session):
        self.module_names = []

        # TODO
        if session == 0:
            for name, module in self.model.named_modules():
                if name in self.args.train_layers:
                    self.module_names.append(name)
        else:
            for name, module in self.model.named_modules():
                if name in self.args.adapter_layers:
                    self.module_names.append(name)

        handles = []
        for name in self.module_names:
            module = dict(self.model.named_modules())[name]
            handles.append(module.register_forward_hook(hook=self.compute_conv))

        for i, sample in enumerate(train_loader):
            input, target = sample
            input = input.to(self.device)
            self.model.forward(input)

        for handle in handles:
            handle.remove()
        torch.cuda.empty_cache()

    def compute_conv(self, module, fea_in, fea_out):
        if isinstance(module, nn.Conv2d):
            kernel_size = module.kernel_size
            stride = module.stride
            padding = module.padding

            unfold = nn.Unfold(kernel_size=1, padding=0, stride=1)
            fea_in_ = unfold(torch.mean(fea_in[0], 0, True))

            fea_in_ = fea_in_.permute(0, 2, 1)
            fea_in_ = fea_in_.reshape(-1, fea_in_.shape[-1])

            module_name = next(name for name, mod in self.model.named_modules() if mod is module)

            if 'adapter' in module_name:
                module_name = module_name.replace('adapter', 'conv')
            self.update_conv(fea_in_, f'{module_name}.weight')

        torch.cuda.empty_cache()
        return None

    def update_conv(self, fea_in, module_name):
        covar = torch.mm(fea_in.transpose(0, 1), fea_in)
        if len(self.covar_matirx[module_name]) == 0:
            self.covar_matirx[module_name] = covar
        else:
            self.covar_matirx[module_name] = self.covar_matirx[module_name] + covar


    def mask_covar_matrix(self):
        for name, cov in self.covar_matirx.items():
            indice_sparsed_channel = self.sparsed_channel_indices[name]  
            fea_in_mask = torch.zeros(1, 512)  
            fea_in_mask[:, indice_sparsed_channel] = 1  
            covar_matrix_mask = torch.mm(fea_in_mask.transpose(0, 1), fea_in_mask).bool()

            true_indices = torch.where(covar_matrix_mask)
            unique_row_indices, _ = torch.unique(true_indices[0], sorted=True, return_inverse=True)
            unique_col_indices, _ = torch.unique(true_indices[1], sorted=True, return_inverse=True)
            select_covar_matrix = cov[unique_row_indices][:, unique_col_indices]
            self.masked_covar_matrix[name] = select_covar_matrix


    def afterTrain(self, current_task):
        if current_task == 0:
            torch.save({'state_dict': self.model.state_dict()}, os.path.join(self.args.result_save_path, f'model_{self.args.dataset_name}_{self.args.base_class}_{self.args.per_class}_{self.args.model_name}({self.args.backbone_name})_pretrained.pth'))
        torch.save({'state_dict': self.model.state_dict()}, os.path.join(self.args.result_save_path, f'model_{self.args.dataset_name}_{self.args.base_class}_{self.args.per_class}_{self.args.model_name}({self.args.backbone_name})_{current_task}.pth'))

        if current_task > 0:
            model_dict = self.model.state_dict()
            for k, v in model_dict.items():
                if 'adapter' in k:
                    k_conv3 = k.replace('adapter', 'conv')
                    model_dict[k_conv3] = model_dict[k_conv3] + F.pad(v, [1, 1, 1, 1], 'constant', 0)
                    model_dict[k] = torch.zeros_like(v)
            self.model.load_state_dict(model_dict)

    def _compute_loss_pretrain(self, imgs, target, old_class=0):
        output = self.model.forward(imgs)
        loss_cls = nn.CrossEntropyLoss(reduce=False)(output / 0.1, target)
        loss_cls = torch.mean(loss_cls, dim=0)
        return loss_cls

    def _compute_loss(self, imgs, target, old_class=0):
        output = self.model.forward(imgs)
        loss_cls = nn.CrossEntropyLoss(reduce=False)(output / 0.1, target)
        loss_cls = torch.mean(loss_cls, dim=0)


        proto_aug = []
        proto_aug_label = []
        index = list(range(old_class))
        for _ in range(self.args.batch_size):
            np.random.shuffle(index)
            temp = self.prototype[index[0]]
            proto_aug.append(temp)
            proto_aug_label.append(self.class_label[index[0]])

        proto_aug = torch.stack(proto_aug).float()
        proto_aug_label = torch.stack(proto_aug_label)
        soft_feat_aug = self.model.classifier(proto_aug, 0)
        loss_protoAug = nn.CrossEntropyLoss()(soft_feat_aug / 0.1, proto_aug_label)
        return loss_cls + 10 * loss_protoAug

    def structure_reorganization(self, para_dict):
        para_dict_re = copy.deepcopy(para_dict)
        for k, v in para_dict.items():
            if 'bn.weight' in k or 'bn1.weight' in k or 'downsample.1.weight' in k:
                if 'bn.weight' in k:
                    k_conv3 = k.replace('bn', 'conv')
                elif 'bn1.weight' in k:
                    k_conv3 = k.replace('bn1', 'conv1')
                elif 'downsample.1.weight' in k:
                    k_conv3 = k.replace('1', '0')
                k_conv3_bias = k_conv3.replace('weight', 'bias')
                k_bn_bias = k.replace('weight', 'bias')
                k_bn_mean = k.replace('weight', 'running_mean')
                k_bn_var = k.replace('weight', 'running_var')

                gamma = para_dict[k]
                beta = para_dict[k_bn_bias]
                running_mean = para_dict[k_bn_mean]
                running_var = para_dict[k_bn_var]
                eps = 1e-5
                std = (running_var + eps).sqrt()
                t = (gamma / std).reshape(-1, 1, 1, 1)
                para_dict_re[k_conv3] *= t
                para_dict_re[k_conv3_bias] = beta - running_mean * gamma / std
        return para_dict_re

    def protoSave(self, model, loader, current_task):
        features = []
        labels = []
        model.eval()
        with torch.no_grad():
            for i, (images, target) in enumerate(loader):
                feature = model.feature_extractor(images.to(self.device))
                if feature.shape[0] == self.args.batch_size:
                    labels.append(target.to(self.device))
                    features.append(feature)

        labels_set = torch.unique(torch.cat(labels))
        labels = torch.cat(labels).view(-1)
        features = torch.cat(features).view(-1, features[0].shape[-1])

        prototype = []
        class_label = []
        for item in labels_set:
            index = (item == labels).nonzero(as_tuple=True)[0]
            class_label.append(item)
            feature_classwise = features[index]
            prototype.append(torch.mean(feature_classwise, dim=0))

        prototype = torch.stack(prototype)
        class_label = torch.stack(class_label)
        if current_task == 0:
            self.prototype = prototype
            self.class_label = class_label
        else:

            self.prototype = torch.cat((prototype, self.prototype), axis=0)
            self.class_label = torch.cat((class_label, self.class_label), axis=0)

    def validation(self, session, epoch=None, end_flag=False):
        self.model.eval()
        self.evaluator.reset()
        forget_history = []

        session_class = self.args.base_class + self.args.per_class * session
        if self.args.dataset_name == 'CIFAR100':
            classes = [0, session_class]
            test_dataset = self.dataset[1] 
            test_dataset.getTestData_up2now(classes)
        elif self.args.dataset_name == 'TinyImageNet':
            classe_set = list(range(200))
            test_transform = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225])])
            data_manager = DataManager(self.args)
            test_dataset = data_manager.get_dataset(test_transform, index=classe_set[:session_class], train=False)
        test_loader = DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=4)

        for i, sample in enumerate(test_loader):
            image, target = sample
            image, target = image.to(self.device), target.to(self.device)
            with torch.no_grad():
                output = self.model(image)
            pred = output.data
            target = target
            pred = torch.argmax(pred, dim=1)
            self.evaluator.add_batch(target, pred)
        
        acc_class_list, mean_acc = self.evaluator.Acc(session_class)
        arr = torch.full((self.args.tasks, self.args.per_class), float('nan')).to(self.device)
        arr.view(-1)[:len(acc_class_list[:self.args.base_class])] = acc_class_list[:self.args.base_class]
        history_acc_matrix = arr.view(self.args.tasks, self.args.per_class)
        history_session_id = torch.full((self.args.tasks, 1), 0).to(self.device)
        history_acc_matrix = torch.cat((history_session_id, history_acc_matrix), dim=1)
        incremental_acc_matrix = torch.full((self.args.tasks, self.args.per_class), float('nan')).to(self.device)
        remaining_elements = acc_class_list[self.args.base_class:]
        rows = len(remaining_elements) // self.args.per_class
        cols = len(remaining_elements) % self.args.per_class
        incremental_acc_matrix[:rows, :] = remaining_elements[:rows * self.args.per_class].view(rows,
                                                                                                self.args.per_class)
        if cols != 0:
            incremental_acc_matrix[rows, :cols] = remaining_elements[rows * self.args.per_class:]
        incremental_session_id = torch.arange(1, self.args.session).view(self.args.tasks, 1).to(self.device)
        incremental_acc_matrix = torch.cat((incremental_session_id, incremental_acc_matrix), dim=1)
        total_matrix = torch.cat((history_acc_matrix, incremental_acc_matrix), dim=1)
        if end_flag:
            msg = f'\n[*][Session={session}, Num_Test_Image={i * self.args.batch_size + image.data.shape[0]}]\nThe final class acc matrix:'
        else:
            msg = f'\n[Session={session}, Epoch={epoch}, Num_Test_Image={i * self.args.batch_size + image.data.shape[0]}]\nThe total class acc matrix:'
        print(msg)
        logging.info(msg)
        for row in total_matrix:
            print(' '.join([f'[{int(item):02d}]' if i == 0 else f'{item:.4f}' for i, item in
                            enumerate(row[:self.args.per_class + 1])])
                  + ' | '
                  + ' '.join([f'[{int(item):02d}]' if i == 0 else f'{item:.4f}' for i, item in
                              enumerate(row[self.args.per_class + 1:])]))
        for row in total_matrix:
            logging.info(' '.join([f'[{int(item):02d}]' if i == 0 else f'{item:.4f}' for i, item in
                                   enumerate(row[:self.args.per_class + 1])])
                         + ' | '
                         + ' '.join([f'[{int(item):02d}]' if i == 0 else f'{item:.4f}' for i, item in
                                     enumerate(row[self.args.per_class + 1:])]))

        if end_flag:
            final_msg = f'The final mean acc: {round(mean_acc.item(), 4)}'
        else:
            final_msg = f'The mean acc: {round(mean_acc.item(), 4)}'
        print(final_msg + '\n')
        logging.info(final_msg)

        if end_flag:
            self.acc_history.append(round(mean_acc.item(), 4))
            for j in range(session + 1):  
                if j == 0:
                    forget_history.append(round(torch.mean(acc_class_list[:self.args.base_class]).item(), 4))
                else:  
                    forget_history.append(round(torch.mean(acc_class_list[self.args.base_class + self.args.per_class * (
                                j - 1): self.args.base_class + self.args.per_class * j]).item(), 4))
            self.forget_history.append(
                forget_history)  
            forget_list = []
            if session == (self.args.session - 1):  
                forget_avg = 0
                for ff in range(self.args.session - 1):  
                    forget_avg = forget_avg + self.forget_history[ff][ff] - self.forget_history[self.args.session - 1][
                        ff]
                    forget_list.append(
                        round(self.forget_history[ff][ff] - self.forget_history[self.args.session - 1][ff], 4))
                forget_avg /= (self.args.session - 1)
                print(f'Average forget: {round(forget_avg, 4)}')
                logging.info(f'\nAverage forget: {round(forget_avg, 4)}')
                print(f'Forget for every session: {forget_list}')
                logging.info(f'Forget for every session: {forget_list}')

                print(f'Last acc: {round(self.acc_history[-1], 4)}')
                logging.info(f'Last acc: {round(self.acc_history[-1], 4)}')
                print(f'Average acc: {round(torch.mean(torch.tensor(self.acc_history)).item(), 4)}')
                logging.info(f'Average acc: {round(torch.mean(torch.tensor(self.acc_history)).item(), 4)}')
                print(f'Acc for every session: {self.acc_history}')
                logging.info(f'Acc for every session: {self.acc_history}')

