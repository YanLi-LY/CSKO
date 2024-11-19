import sys
import numpy as np
import torch


class Evaluator(object):
    def __init__(self, num_class, device):
        self.device = device
        self.num_class = num_class 
        self.confusion_matrix = torch.zeros((self.num_class,) * 2).to(self.device) 

    def Acc(self, session_class):
        confusion_matrix = self.confusion_matrix[:session_class, :session_class] 
        return torch.diag(confusion_matrix) / torch.sum(confusion_matrix, dim=1), torch.sum(torch.diag(confusion_matrix)) / torch.sum(confusion_matrix)

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)

        label = self.num_class * gt_image[mask].int() + pre_image[mask]

        count = torch.bincount(label, minlength=self.num_class ** 2)

        confusion_matrix = count.view(self.num_class, self.num_class)

        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = torch.zeros((self.num_class,) * 2).to(self.device)
