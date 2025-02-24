# Machine Intelligence with Deep Learning
## Importance batching for improved training of neural networks
---
**Course description**: Current day neural networks are trained using stochastic learning, which consists of splitting the (usually large) training data into multiple batches, called mini-batches. This is desirable since it speeds up training and helps the network's convergence due to the added noise. However, the samples that form such mini-batches are usually chosen randomly throughout training, which might not be ideal for optimal learning. The goal of this topic is to study the effects of constructing each mini-batch using importance sampling techniques based on the network's loss.

--- 
This repository holds the code for the **MIDL** seminar by [**Hasso-Plattner-Institut**](https://hpi.de).  
In this seminar, we're trying to find out whether there are better approaches for creating mini batches than random-sampling. Current ideas are:
- Sorting batches by overall loss 
- Sorting batches by loss per category
- Sorting by class distribution
- Batching dynamic sizes
- ...

---
**Dataset**: [**CIFAR-10**](https://www.cs.toronto.edu/~kriz/cifar.html)

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.
The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.
![CIFAR-10](images/cifar10.jpg "CIFAR-10")

**Model**: [**ResNet**](https://arxiv.org/abs/1512.03385)

The neural network used during the seminar is the ResNet model with 18 layers, which is a model published in late 2015. Back then, it was the state-of-the-art for image recognition. 
![ResNet](images/resnet_visualization.png "ResNet")

**Metrics**: The following table shows the best metrics (i.e. highest accuracy and lowest loss) which the model reached in the given mode after 150 epochs. The modes describe the way the dataset was sorted before being batched:
- Freeze: the data was not sorted at all
- Shuffle: the data is randomly sorted before each epoch
- Homogeneous: for as many batches as possible, the inputs consist of one class only
- Heterogeneous: for as many batches as possible, the inputs consist of all classes

| Mode          | Training | Accuracy | Loss  |
|---------------|----------|----------|-------|
|Freeze         | X        |100.00    |0.030  |
|Shuffle        | X        |100.00    |0.030  |
|Homogeneous    | X        |085.95    |0.045  |
|Heterogeneous  | X        |100.00    |0.028  |
|Freeze         |          |086.65    |0.026  |
|Shuffle        |          |089.57    |0.026  |
|Homogeneous    |          |010.00    |0.209  |
|Heterogeneous  |          |089.26    |0.024  |
(must be updated)

**Graphs**: The complete results can be analyzed given the following eight graphs.
![Train Accuracy graph 1](figures/20200129_1_accuracy_train.jpg "Train Accuracy 1")
![Test Accuracy graph 1](figures/20200129_1_accuracy_test.jpg "Test Accuracy 1")
![Train Accuracy graph 2](figures/20200129_2_accuracy_train.jpg "Train Accuracy 2")
![Test Accuracy graph 2](figures/20200129_2_accuracy_test.jpg "Test Accuracy 2")

![Train Loss graph 1](figures/20200129_1_loss_train.jpg "Train Loss 1")
![Test Loss graph 1](figures/20200129_1_loss_test.jpg "Test Loss 1")
![Train Loss graph 2](figures/20200129_2_loss_train.jpg "Train Loss 2")
![Test Loss graph 2](figures/20200129_2_loss_test.jpg "Test Loss 2")
known error for the loss graph: the y-axis has a wrong label (should be cross entropy loss instead of accuracy, will be updated soon)
