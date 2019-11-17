import torch
import torchvision
import torchvision.transforms as transforms

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.utils import shuffle

from .sorting_utils import sort_one_class, sort_all_classes

class DataLoader():
    
    def __init__(self, root='./data'):
        self.root = root        
        self.trainset = None
        self.testset = None
        
    def download_cifar(self, batch_size=64):
        #print('==> Preparing data..')
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        # data needs to be loaded through dataloader to get into the correct format
        trainset = torchvision.datasets.CIFAR10(root=self.root, train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2)
        self.X_train, self.Y_train = [], []
        for x, y in trainloader:
            self.X_train.extend(x.numpy()) #numpy needed for a casting workaround
            self.Y_train.extend(y.numpy())
        self.X_train = np.array(self.X_train)
        self.Y_train = np.array(self.Y_train)
        
        testset = torchvision.datasets.CIFAR10(root=self.root, train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
        self.X_test, self.Y_test = [], []
        for x, y in testloader:
            self.X_test.extend(x.numpy())
            self.Y_test.extend(y.numpy())
        self.X_test = np.array(self.X_test)
        self.Y_test = np.array(self.Y_test)
                
        self.X_batches_train, self.Y_batches_train = None, None
        self.X_batches_test, self.Y_batches_test = None, None
        
    def prepare_cifar(self, strategy, batch_size=64, random_state=None):
        self.batch_size = batch_size
        assert strategy in ['freeze', 'shuffle', 'homogeneous', 'heterogeneous'], 'Unknown action'
        if strategy == 'freeze':
            self.X_batches_train, self.Y_batches_train = self.X_train, self.Y_train
            self.X_batches_test, self.Y_batches_test = self.X_test, self.Y_test
        elif strategy == 'shuffle':
            self.X_batches_train, self.Y_batches_train = shuffle(self.X_train, self.Y_train, random_state=random_state)
            self.X_batches_test, self.Y_batches_test = shuffle(self.X_test, self.Y_test, random_state=random_state)
        elif strategy == 'homogeneous':
            self.X_batches_train, self.Y_batches_train = sort_one_class(self.X_train, self.Y_train, self.batch_size, \
                                                            use_shuffle=True, random_state=random_state)
            self.X_batches_test, self.Y_batches_test = sort_one_class(self.X_test, self.Y_test, self.batch_size, \
                                                            use_shuffle=True, random_state=random_state)
        elif strategy == 'heterogeneous':
            self.X_batches_train, self.Y_batches_train = sort_all_classes(self.X_train, self.Y_train, self.batch_size, \
                                                              use_shuffle=True, random_state=random_state)
            self.X_batches_test, self.Y_batches_test = sort_all_classes(self.X_test, self.Y_test, self.batch_size, \
                                                              use_shuffle=True, random_state=random_state)
    
    def yield_batches(self, use_train=True):
        batch_idx = 0
        X = self.X_batches_train if use_train else self.X_batches_test
        Y = self.Y_batches_train if use_train else self.Y_batches_test
        X, Y = torch.from_numpy(X), torch.from_numpy(Y)
        while batch_idx < len(X):
            yield X[batch_idx: batch_idx+self.batch_size], Y[batch_idx: batch_idx+self.batch_size]
            batch_idx += self.batch_size
            
###### OLD VERSION
##class DataLoader():
##    
##    def __init__(self, root='./data', batch_sizes = {'train': 128, 'test': 128}, shuffle = {'train': True, 'test': False}):
##        assert 'train' in batch_sizes and 'test' in batch_sizes, \
##        'Parameter @batch_sizes contains wrong arguments. Please specify @train and @test'
##        assert 'train' in shuffle and 'test' in shuffle, \
##        'Parameter @shuffle contains wrong arguments. Please specify @train and @test'
##        self.root = root
##        self.batch_sizes = batch_sizes
##        self.shuffle = shuffle
##        
##        self.trainset = None
##        self.testset = None
##        
##    def download_cifar(self):
##        print('==> Preparing data..')
##        transform_train = transforms.Compose([
##            transforms.RandomCrop(32, padding=4),
##            transforms.RandomHorizontalFlip(),
##            transforms.ToTensor(),
##            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
##        ])
##
##        transform_test = transforms.Compose([
##            transforms.ToTensor(),
##            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
##        ])
##
##        self.trainset = torchvision.datasets.CIFAR10(root=self.root, train=True, download=True, transform=transform_train)
##        self.testset = torchvision.datasets.CIFAR10(root=self.root, train=False, download=True, transform=transform_test)
##    
##    def get_loaders(self, verbose=True):
##        trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_sizes['train'], 
##                                                  shuffle=self.shuffle['train'], num_workers=2)
##        testloader = torch.utils.data.DataLoader(self.testset, batch_size=self.batch_sizes['test'], 
##                                                 shuffle=self.shuffle['test'], num_workers=2)
##        return trainloader, testloader