import torch
import torchvision
import torchvision.transforms as transforms

class DataLoader():
    
    def __init__(self, root='./data', batch_sizes = {'train': 128, 'test': 128}, shuffle = {'train': True, 'test': False}):
        assert 'train' in batch_sizes and 'test' in batch_sizes, \
        'Parameter @batch_sizes contains wrong arguments. Please specify @train and @test'
        assert 'train' in shuffle and 'test' in shuffle, \
        'Parameter @shuffle contains wrong arguments. Please specify @train and @test'
        self.root = root
        self.batch_sizes = batch_sizes
        self.shuffle = shuffle
        
        self.trainset = None
        self.testset = None
        
    def download_cifar(self):
        print('==> Preparing data..')
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

        self.trainset = torchvision.datasets.CIFAR10(root=self.root, train=True, download=True, transform=transform_train)
        self.testset = torchvision.datasets.CIFAR10(root=self.root, train=False, download=True, transform=transform_test)
    
    def get_loaders(self, verbose=True):
        trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_sizes['train'], 
                                                  shuffle=self.shuffle['train'], num_workers=2)
        testloader = torch.utils.data.DataLoader(self.testset, batch_size=self.batch_sizes['test'], 
                                                 shuffle=self.shuffle['test'], num_workers=2)
        return trainloader, testloader