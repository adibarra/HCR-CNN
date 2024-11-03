# CS 4375 - Machine Learning Final Project

## Overview

This project is a machine learning project that uses convolutional neural networks (CNNs) to classify handwritten digits and letters.

## Datasets

- The [MNIST](https://www.tensorflow.org/datasets/catalog/mnist) dataset is a collection of 28x28 pixel grayscale images of handwritten digits (0-9). The dataset contains 60,000 training images and 10,000 test images.

- The [EMNIST Digits](https://www.tensorflow.org/datasets/catalog/emnist) dataset is an extension of the MNIST dataset that contains 28x28 pixel grayscale images of handwritten digits (0-9). The dataset contains 240,000 training images and 40,000 test images.

- The [EMNIST Letters](https://www.tensorflow.org/datasets/catalog/emnist) dataset is an extension of the MNIST dataset that contains 28x28 pixel grayscale images of handwritten letters (A-Z). The dataset contains 88,800 training images and 14,800 test images.

## Methodology

The project uses the [TensorFlow](https://www.tensorflow.org/) library to train a convolutional neural network (CNN) model. The model has 2 convolutional layers, 2 max pooling layers, and 2 dense layers. The model is trained using the Adam optimizer with the categorical crossentropy loss function. Some of the hyperparameters that were tuned include the learning rate, convolutional layer units, dense layer units, and dropout rate.

Tuning of the hyperparameters was done using the [Keras Tuner](https://keras-team.github.io/keras-tuner/) library's Hyperband tuner. This tuner uses a novel approach of tournament bracket-style resource allocation known as successive halving. Hyperband optimizes hyperparameter tuning by arranging models in several brackets, each containing multiple trials with unique hyperparameter configurations. Initially, a small amount of computational resources (such as training epochs) is allocated to each model in a bracket. Models that perform best after these early training rounds advance to receive additional resources, while weaker models are eliminated from further training. This approach allows Hyperband to explore a large range of hyperparameters quickly and focus resources on the most promising configurations.

## Results

There are three iterations of the model. The first of which was trained on the MNIST digits dataset, the second on the EMNIST digits dataset, and the third on a combination of the EMNIST digits and letters datasets.

### Model 1: MNIST Digits

Below are the results for the 3 best performing models trained on the MNIST digits dataset. Each of these models was trained with a different batch size. All models were trained for 30 epochs.

Total trained models: ~ 7000
Total training time: ~ 72 hours

##### Best Models

> mean_val_accuracy: 0.994499, conv1: 256, conv2: 160, dense_units: 256, dropout_rate: 0.3, learning_rate: 0.000804, batch_size: 208

![](./results/mnist/208-best.png)

> mean_val_accuracy: 0.994083, conv1: 256, conv2: 128, dense_units: 224, dropout_rate: 0.6, learning_rate: 0.000590, batch_size: 96

![](./results/mnist/96-best.png)

> mean_val_accuracy: 0.994000, conv1: 128, conv2: 192, dense_units: 160, dropout_rate: 0.5, learning_rate: 0.001392, batch_size: 192

![](./results/mnist/192-best.png)


### Model 2: EMNIST Digits

Below are the results for each of the best models trained on the EMNIST digits dataset. Each of these models was trained with a different batch size. All models were trained for 30 epochs.

Total training time: ~ 
