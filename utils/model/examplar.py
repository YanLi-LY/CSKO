import abc
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import copy
import numpy as np


class ExemplarHandler(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self):
        super().__init__()

        self.exemplar_sets = []   
        self.exemplar_means = []
        self.compute_means = True

        self.memory_budget = 2000
        self.norm_exemplars = True
        self.herding = True

    @abc.abstractmethod
    def feature_extractor(self, images):
        pass

    def reduce_exemplar_sets(self, m):
        for y, P_y in enumerate(self.exemplar_sets):
            self.exemplar_sets[y] = P_y[:m]

    def construct_exemplar_set(self, dataset, n):
        self.eval()

        n_max = len(dataset)
        exemplar_set = []
        class_mean = None
        if self.herding:
            first_entry = True
            dataloader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=4, drop_last=False, pin_memory=True)
            for (image_batch, _) in dataloader:
                if self.args.cuda:
                    image_batch = image_batch.cuda()
                with torch.no_grad():
                    feature_batch = self.feature_extractor(image_batch)
                if first_entry:
                    features = feature_batch
                    first_entry = False
                else:
                    features = torch.cat([features, feature_batch], dim=0)
            if self.norm_exemplars:
                features = F.normalize(features, p=2, dim=1)

            class_mean = torch.mean(features, dim=0, keepdim=True)
            if self.norm_exemplars:
                class_mean = F.normalize(class_mean, p=2, dim=1)

            exemplar_features = torch.zeros_like(features[:min(n, n_max)])
            list_of_selected = []
            for k in range(min(n, n_max)):
                if k > 0:
                    exemplar_sum = torch.sum(exemplar_features[:k], dim=0).unsqueeze(0)
                    features_means = (features + exemplar_sum)/(k+1)
                    features_dists = features_means - class_mean
                else:
                    features_dists = features - class_mean
                index_selected = np.argmin(torch.norm(features_dists, p=2, dim=1).cpu())
                if index_selected in list_of_selected:
                    raise ValueError("Exemplars should not be repeated!!!!")
                list_of_selected.append(index_selected)

                exemplar_set.append(dataset[index_selected])
                exemplar_features[k] = copy.deepcopy(features[index_selected])

                features[index_selected] = features[index_selected] + 10000
        else:
            indeces_selected = np.random.choice(n_max, size=min(n, n_max), replace=False)
            for k in indeces_selected:
                exemplar_set.append(dataset[k])
        self.exemplar_sets.append(exemplar_set)

        return class_mean


class ExemplarDataset(Dataset):
    def __init__(self, exemplar_sets):
        super().__init__()
        self.exemplar_sets = exemplar_sets
        self.exemplar_datasets = []
        for class_id in range(len(self.exemplar_sets)):
            self.exemplar_datasets += self.exemplar_sets[class_id]

    def __len__(self):
        return len(self.exemplar_datasets)

    def __getitem__(self, index):
        return self.exemplar_datasets[index]