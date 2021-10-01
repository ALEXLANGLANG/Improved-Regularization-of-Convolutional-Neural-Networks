# Improved Regularization of Convolutional Neural Networks

Overfitting is a common problem for Convolutional neural networks (CNN).
This project is mainly exploring the combination of two regularization techniques: 
mixup and cutout in order to improve the generalization of CNN models. We explore performance of them on CIFAR10-Resnet 
with different depth and try to find the best combination of them on CIFAR10-ResNet18. 
And then we transfer this combination on other datasets like SVHN and Fashion-MNIS. 
In general, the experiments show that both of mixup and cutout works well on addressing overfitting, 
but it is harder for mixup to handle overfitting problems if the model goes deeper. 
And the best combination of them also works well for these two other datasets.

## Poster
![poster](https://user-images.githubusercontent.com/49976598/135658720-73fd6bba-cbd5-462c-a86a-0ea101dce97d.jpg)
[downLoad_poster](https://github.com/ALEXLANGLANG/Improved-Regularization-of-Convolutional-Neural-Networks/files/7268843/Poster.pdf)
