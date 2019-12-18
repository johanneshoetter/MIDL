import torch
import torchvision
import torchvision.transforms as transforms

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.utils import shuffle

from .sorting_utils import sort_one_class, sort_all_classes, weighted_highest_sampling

class DataLoader():
    
    def __init__(self, root='./data', batch_size=64):
        self.root = root        
        self.trainset = None
        self.testset = None
        
        self.seed_incrementer = {
            'freeze': 0,
            'shuffle': 0,
            'homogeneous': 0,
            'heterogeneous': 0
        }
        
        self.batch_size = batch_size
        
    def download_cifar(self):
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
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=False, num_workers=2)
        self.X_train, self.Y_train = [], []
        for x, y in trainloader:
            self.X_train.extend(x.numpy()) #numpy needed for a casting workaround
            self.Y_train.extend(y.numpy())
        self.X_train = np.array(self.X_train)
        self.Y_train = np.array(self.Y_train)
        
        testset = torchvision.datasets.CIFAR10(root=self.root, train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size, shuffle=False, num_workers=2)
        self.X_test, self.Y_test = [], []
        for x, y in testloader:
            self.X_test.extend(x.numpy())
            self.Y_test.extend(y.numpy())
        self.X_test = np.array(self.X_test)
        self.Y_test = np.array(self.Y_test)
                
        self.X_batches_train, self.Y_batches_train = None, None
        self.X_batches_test, self.Y_batches_test = None, None
            
    def yield_batches(self, strategy, use_train=True, random_state=None, model=None):
        
        assert strategy in ['freeze', 'shuffle', 'homogeneous', 'heterogeneous'], 'Unknown action'
        
        # seeds will be incremented at each epoch, to give the shuffling methods a new random seed
        # this will still be deterministic!
        current_seed = random_state + self.seed_incrementer[strategy]
        self.seed_incrementer[strategy] += 1
        
        # prepare the data
        if strategy == 'freeze':
            self.X_batches_train, self.Y_batches_train = self.X_train, self.Y_train
            self.X_batches_test, self.Y_batches_test = self.X_test, self.Y_test
        elif strategy == 'shuffle':
            self.X_batches_train, self.Y_batches_train = shuffle(self.X_train, self.Y_train, random_state=current_seed)
            self.X_batches_test, self.Y_batches_test = shuffle(self.X_test, self.Y_test, random_state=current_seed)
        elif strategy == 'homogeneous':
            self.X_batches_train, self.Y_batches_train = sort_one_class(self.X_train, self.Y_train, self.batch_size, \
                                                            use_shuffle=True, random_state=current_seed)
            self.X_batches_test, self.Y_batches_test = sort_one_class(self.X_test, self.Y_test, self.batch_size, \
                                                            use_shuffle=True, random_state=current_seed)
        elif strategy == 'heterogeneous':
            self.X_batches_train, self.Y_batches_train = sort_all_classes(self.X_train, self.Y_train, self.batch_size, \
                                                              use_shuffle=True, random_state=current_seed)
            self.X_batches_test, self.Y_batches_test = sort_all_classes(self.X_test, self.Y_test, self.batch_size, \
                                                              use_shuffle=True, random_state=current_seed)
        elif strategy == 'max_k_loss':
            pass
        elif strategy == 'min_k_loss':
            pass
        
        # yield it in batches
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