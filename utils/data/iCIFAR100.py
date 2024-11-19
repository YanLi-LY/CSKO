from torchvision.datasets import CIFAR100
import numpy as np
from PIL import Image


class iCIFAR100(CIFAR100):
    def __init__(self, root, 
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=False):
        super(iCIFAR100, self).__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.TrainData = []
        self.TrainLabels = []
        self.TestData = []
        self.TestLabels = []

    def _map_new_class_index(self, y, order):
        """Transforms targets for new class order."""
        return list(map(lambda x: order.index(x), y))

    def concatenate(self, datas, labels):
        con_data = datas[0]
        con_label = labels[0]
        for i in range(1, len(datas)):
            con_data = np.concatenate((con_data, datas[i]), axis=0)
            con_label = np.concatenate((con_label, labels[i]), axis=0)
        return con_data, con_label

    def getTestData(self, classes):
        datas, labels = [], []
        for label in range(classes[0], classes[1]):
            data = self.data[np.array(self.targets) == label]
            datas.append(data)
            labels.append(np.full((data.shape[0]), label))
        datas, labels = self.concatenate(datas, labels)
        self.TestData = datas if self.TestData == [] else np.concatenate((self.TestData, datas), axis=0)
        self.TestLabels = labels if self.TestLabels == [] else np.concatenate((self.TestLabels, labels), axis=0)

    def getTestData_up2now(self, classes):
        datas, labels = [], []
        for label in range(classes[0], classes[1]):
            data = self.data[np.array(self.targets) == label]
            datas.append(data)
            labels.append(np.full((data.shape[0]), label))
        datas, labels = self.concatenate(datas, labels)
        self.TestData = datas
        self.TestLabels = labels

    def getTrainData(self, classes): 
        datas, labels = [], []
        for label in range(classes[0], classes[1]):
            data = self.data[np.array(self.targets) == label] 
            datas.append(data)
            labels.append(np.full((data.shape[0]), label)) 
        self.TrainData, self.TrainLabels = self.concatenate(datas, labels)

    def getTrainData_sub(self, classes):
        datas, labels = [], []
        for label in range(classes[0], classes[1]):
            data = self.data[np.array(self.targets) == label][:20]
            datas.append(data)
            labels.append(np.full((data.shape[0]), label))
        self.TrainData, self.TrainLabels = self.concatenate(datas, labels)
      
    def getTrainItem(self, index):
        img, target = Image.fromarray(self.TrainData[index]), self.TrainLabels[index]
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)
        return img, target

    def getTestItem(self, index):
        img, target = Image.fromarray(self.TestData[index]), self.TestLabels[index]
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)
        return img, target

    def __getitem__(self, index):
        if self.TrainData != []:
            return self.getTrainItem(index)
        elif self.TestData != []:
            return self.getTestItem(index)

    def __len__(self):
        if self.TrainData != []:
            return len(self.TrainData)
        elif self.TestData != []:
            return len(self.TestData)

    def get_image_class(self, label):
        return self.data[np.array(self.targets) == label]

