import torch
import torchvision
import torchvision.transforms as transforms

import torch
import torchvision
import torchvision.transforms as transforms

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.utils import shuffle

from utils.sorting_utils import sort_one_class, sort_all_classes, weighted_highest_sampling, weighted_highest_sampling_per_class, weighted_random_sampling

from collections import defaultdict

class DataLoader():
    
    def __init__(self, root='./data', batch_size=64):
        self.root = root        
        self.trainset = None
        self.testset = None
        self.known_strategies = ['freeze', 'shuffle', 'homogeneous', 'heterogeneous', 'max_k_loss', 'min_k_loss', 'heterogeneous_max_k_loss', 'heterogeneous_min_k_loss', 'weighted_random_sampling']
        self.seed_incrementer = {strategy: 0 for strategy in self.known_strategies}
        
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
            
    def yield_batches(self, strategy, use_train=True, random_state=None, criterion=None, device=None, num_iterations=100, update_every_iteration=False):
        assert strategy in self.known_strategies, 'Unknown action'
        
        # seeds will be incremented at each epoch, to give the shuffling methods a new random seed
        # this will still be deterministic!
        current_seed = random_state + self.seed_incrementer[strategy]
        self.seed_incrementer[strategy] += 1
        
        # prepare the data
        yield_batchwise = True
        self.X_batches_test, self.Y_batches_test = self.X_test, self.Y_test
        if strategy == 'freeze':
            self.X_batches_train, self.Y_batches_train = self.X_train, self.Y_train
        elif strategy == 'shuffle':
            self.X_batches_train, self.Y_batches_train = shuffle(self.X_train, self.Y_train, random_state=current_seed)
        elif strategy == 'homogeneous':
            self.X_batches_train, self.Y_batches_train = sort_one_class(self.X_train, self.Y_train, self.batch_size, \
                                                            use_shuffle=True, random_state=current_seed)
        elif strategy == 'heterogeneous':
            self.X_batches_train, self.Y_batches_train = sort_all_classes(self.X_train, self.Y_train, self.batch_size, \
                                                              use_shuffle=True, random_state=current_seed)
        elif strategy == 'max_k_loss' or strategy == 'min_k_loss':
            yield_batchwise = False
            top_fn = max if strategy == 'max_k_loss' else min
            for _ in range(num_iterations):
                pulled_idxs = weighted_highest_sampling(self.weighted_indices, batch_size=self.batch_size, top_fn=top_fn)
                if update_every_iteration:
                    self.initialize_weights(criterion, device, seed=random_state)
                else:
                    self._update_weights(pulled_idxs, criterion, device)
                X, Y = self.get_from_idxs(pulled_idxs)
                yield X, Y
                
        elif strategy == 'weighted_random_sampling':
            yield_batchwise = False
            for _ in range(num_iterations):
                pulled_idxs = weighted_random_sampling(self.weighted_indices, batch_size=self.batch_size, random_state=random_state)
                if update_every_iteration:
                    self.initialize_weights(criterion, device, seed=random_state)
                else:
                    self._update_weights(pulled_idxs, criterion, device)
                X, Y = self.get_from_idxs(pulled_idxs)
                yield X, Y
                
        elif strategy == 'heterogeneous_max_k_loss' or strategy == 'heterogeneous_min_k_loss':
            yield_batchwise = False
            top_fn = max if strategy == 'heterogeneous_max_k_loss' else min
            for _ in range(num_iterations):
                pulled_idxs = weighted_highest_sampling_per_class(self.weighted_indices_per_class, batch_size=self.batch_size, top_fn=top_fn)
                if update_every_iteration:
                    self.initialize_weights_per_class(criterion, device, seed=random_state)
                else:
                    self._update_weights_per_class(pulled_idxs, criterion, device)
                X, Y = self.get_from_idxs(pulled_idxs)
                yield X, Y
            
        if yield_batchwise:
            # yield it in batches
            batch_idx = 0
            X = self.X_batches_train if use_train else self.X_batches_test
            Y = self.Y_batches_train if use_train else self.Y_batches_test
            X, Y = torch.from_numpy(X), torch.from_numpy(Y)
            while batch_idx < len(X):
                yield X[batch_idx: batch_idx+self.batch_size], Y[batch_idx: batch_idx+self.batch_size]
                batch_idx += self.batch_size
            
    def get_from_idxs(self, idxs, use_train=True):
        if use_train:
            X, Y = self.X_train[idxs], self.Y_train[idxs]
        else:
            X, Y = self.X_test[idxs], self.Y_test[idxs]
        return torch.from_numpy(X), torch.from_numpy(Y)
    
    def set_model(self, model):
        self.model = model
    
    def initialize_weights(self, criterion, device, seed=None, dump=None):
        assert self.model != None, 'Model needs to be set first!'
        
        self.weighted_indices = {}
        sample_idx = 0
        
        # using freeze as this is the simplest way of getting data in batches fast
        dump_losses = []
        for inputs, targets in self.yield_batches('freeze', random_state=seed, use_train=True):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = self.model(inputs)
            losses = criterion(outputs, targets)
            for loss in losses:
                loss = float(loss.cpu().detach().numpy())
                dump_losses.append(loss)
                self.weighted_indices[sample_idx] = loss
                sample_idx += 1
        if dump:
            with open(dump, 'w') as file_handler:
                for loss in dump_losses:
                    file_handler.write("{}\n".format(loss))
                    
    def initialize_weights_per_class(self, criterion, device, seed=None):
        assert self.model != None, 'Model needs to be set first!'
        
        self.weighted_indices_per_class = defaultdict(dict)
        self.sample2class = {}
        sample_idx = 0
        
        # using freeze as this is the simplest way of getting data in batches fast
        for inputs, targets in self.yield_batches('freeze', random_state=seed, use_train=True):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = self.model(inputs)
            losses = criterion(outputs, targets)
            for target, loss in zip(targets, losses):
                loss = float(loss.cpu().detach().numpy())
                self.weighted_indices_per_class[target][sample_idx] = loss
                self.sample2class[sample_idx] = target
                sample_idx += 1
            
                
    def _update_weights(self, idxs, criterion, device, seed=None):
        inputs, targets = self.get_from_idxs(idxs)
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = self.model(inputs)
        losses = criterion(outputs, targets)
        for idx, loss in zip(idxs, losses):
            loss = float(loss.cpu().detach().numpy())
            self.weighted_indices[idx] = loss
            
            
    def _update_weights_per_class(self, idxs, criterion, device, seed=None):
        inputs, targets = self.get_from_idxs(idxs)
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = self.model(inputs)
        losses = criterion(outputs, targets)
        for idx, loss in zip(idxs, losses):
            target = self.sample2class[idx]
            loss = float(loss.cpu().detach().numpy())
            self.weighted_indices_per_class[target][idx] = loss
            