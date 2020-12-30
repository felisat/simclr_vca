from PIL import Image
from torchvision import transforms
from torchvision.datasets import CIFAR10, STL10, SVHN, MNIST, EMNIST, CelebA
import numpy as np
import torch
import os, pickle


class MNISTPair(MNIST):

    def __init__(self, **kw_args):
        super().__init__(**kw_args)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(np.stack([img]*3, axis=-1))#Image.fromarray(np.transpose(img, (1,2,0)))

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target


class EMNISTPair(EMNIST):

    def __init__(self, **kw_args):
        super().__init__(**kw_args)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(np.stack([img]*3, axis=-1))#Image.fromarray(np.transpose(img, (1,2,0)))

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target


class CIFAR10Pair(CIFAR10):

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target




class SVHNPair(SVHN):

    def __init__(self, **kw_args):
        super().__init__(**kw_args)
        self.targets = self.labels


    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(np.transpose(img, (1,2,0)))

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target



class STL10Pair(STL10):

    def __init__(self, **kw_args):
        super().__init__(**kw_args)
        self.targets = self.labels

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]

        img = Image.fromarray(np.transpose(img, (1,2,0)))

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target


class CelebAPair(CelebA):

    def __init__(self, **kw_args):
        super().__init__(**kw_args)

        print(dir(self))

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(np.transpose(img, (1,2,0)))

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target



class Imagenet32(torch.utils.data.Dataset):

    def __init__(self, root, subset, transform):
        super().__init__()
        data = load_imagenet_data(data_folder=root, 
                       subset=subset, verbose=False, max_data=160000)
        self.data = data["X_train"]
        self.labels = self.targets = data["Y_train"]
        self.transform = transform 
        self.target_transform = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]

        if self.transform is not None:
            img = self.transform(img)


        return img, target



class Imagenet32Pair(Imagenet32):

    def __init__(self, **kw_args):
        super().__init__(**kw_args)
        self.targets = self.labels

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]

        img = Image.fromarray(np.transpose(img, (1,2,0)))

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target





def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict


def get_ids(filename, data_folder="/home/sattler/Data/PyTorch/Imagenet32"):
    tags = np.loadtxt(os.path.join(data_folder, "classes.txt"), dtype=str)[:,0]
    class_tags = np.loadtxt(os.path.join(data_folder, filename), dtype=str)
    
    class_ids = np.array([np.argwhere(x==tags).flatten() for x in class_tags if x in tags]).flatten()
    
    return class_ids


def load_imagenet_data(data_folder="/home/sattler/Data/PyTorch/Imagenet32", 
                       subset=None, verbose=False, max_data=None):
    
    if subset:
        classes = get_ids(filename=subset, data_folder=data_folder)
    else:
        classes = range(1000)
    
    X_train = []
    Y_train = []
    for i in range(1,11):
        if verbose:
            print('Loading.. train_data_batch_{}'.format(i))
        data_file = os.path.join(data_folder, 'train_data_batch_{}'.format(i))

        img_size = 32

        d = unpickle(data_file)
        x = d['data']
        y = np.array(d['labels'])-1

        idcs = np.argwhere([yy in classes for yy in y]).flatten()

        x, y = x[idcs], y[idcs]


        data_size = x.shape[0]

        img_size2 = img_size * img_size

        x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
        x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2)

        X_train += [x]
        Y_train += [y.astype('int32')]
        
        
    return dict(
        X_train=np.concatenate(X_train)[:max_data],
        Y_train=np.concatenate(Y_train)[:max_data])



train_transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])



train_transform_mnist = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor()])

test_transform_mnist = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor()])