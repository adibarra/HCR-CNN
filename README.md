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

Below are the results for each of the best models trained on the MNIST digits dataset. Each of these models was trained with a different batch size. All models were trained for 30 epochs.

Total training time: ~ 72 hours

##### Best Model

> mean_val_accuracy: 0.994499, conv1: 256, conv2: 160, dense_units: 256, dropout_rate: 0.3, learning_rate: 0.000804, batch_size: 208

![](./results/mnist/208-best.png)

<details>
<summary>Click to expand</summary>

###### Batch Size: 16

> mean_val_accuracy: 0.993499, conv1: 160, conv2: 256, dense_units: 128, dropout_rate: 0.4, learning_rate: 0.000263, batch_size: 16

![](./results/mnist/16-best.png)

###### Batch Size: 32

> mean_val_accuracy: 0.993749, conv1: 192, conv2: 224, dense_units: 192, dropout_rate: 0.6, learning_rate: 0.000844, batch_size: 32

![](./results/mnist/32-best.png) 

###### Batch Size: 48

> mean_val_accuracy: 0.994000, conv1: 160, conv2: 256, dense_units: 256, dropout_rate: 0.4, learning_rate: 0.000396, batch_size: 48

![](./results/mnist/64-best.png)

###### Batch Size: 64

> mean_val_accuracy: 0.993833, conv1: 160, conv2: 160, dense_units: 192, dropout_rate: 0.4, learning_rate: 0.000698, batch_size: 64

![](./results/mnist/64-best.png)

###### Batch Size: 80

> mean_val_accuracy: 0.993166, conv1: 160, conv2: 128, dense_units: 160, dropout_rate: 0.7, learning_rate: 0.000615, batch_size: 80

![](./results/mnist/80-best.png)

###### Batch Size: 96

> mean_val_accuracy: 0.994083, conv1: 256, conv2: 128, dense_units: 224, dropout_rate: 0.6, learning_rate: 0.000590, batch_size: 96

![](./results/mnist/96-best.png)

###### Batch Size: 112

> mean_val_accuracy: 0.993499, conv1: 64, conv2: 192, dense_units: 224, dropout_rate: 0.2, learning_rate: 0.000627, batch_size: 112

![](./results/mnist/112-best.png)

###### Batch Size: 128

> mean_val_accuracy: 0.993833, conv1: 256, conv2: 128, dense_units: 224, dropout_rate: 0.6, learning_rate: 0.000449, batch_size: 128

![](./results/mnist/128-best.png)

###### Batch Size: 144

> mean_val_accuracy: 0.993416, conv1: 128, conv2: 64, dense_units: 224, dropout_rate: 0.6, learning_rate: 0.000989, batch_size: 144

![](./results/mnist/144-best.png)

###### Batch Size: 160

> mean_val_accuracy: 0.993833, conv1: 192, conv2: 224, dense_units: 224, dropout_rate: 0.8, learning_rate: 0.001038, batch_size: 160

![](./results/mnist/160-best.png)

###### Batch Size: 176

> mean_val_accuracy: 0.993916, conv1: 64, conv2: 224, dense_units: 96, dropout_rate: 0.6, learning_rate: 0.001665, batch_size: 176

![](./results/mnist/176-best.png)

###### Batch Size: 192

> mean_val_accuracy: 0.994000, conv1: 128, conv2: 192, dense_units: 160, dropout_rate: 0.5, learning_rate: 0.001392, batch_size: 192

![](./results/mnist/192-best.png)


###### Batch Size: 208

> mean_val_accuracy: 0.994499, conv1: 256, conv2: 160, dense_units: 256, dropout_rate: 0.3, learning_rate: 0.000804, batch_size: 208

![](./results/mnist/208-best.png)

###### Batch Size: 224

> mean_val_accuracy: 0.993749, conv1: 192, conv2: 224, dense_units: 224, dropout_rate: 0.6, learning_rate: 0.000453, batch_size: 224

![](./results/mnist/224-best.png)

###### Batch Size: 240

> mean_val_accuracy: 0.993583, conv1: 96, conv2: 224, dense_units: 192, dropout_rate: 0.7, learning_rate: 0.000664, batch_size: 240

![](./results/mnist/240-best.png)

###### Batch Size: 256

> mean_val_accuracy: 0.993666, conv1: 192, conv2: 192, dense_units: 256, dropout_rate: 0.5, learning_rate: 0.000676, batch_size: 256

![](./results/mnist/256-best.png)

</details>

### Model 2: EMNIST Digits

Below are the results for each of the best models trained on the EMNIST digits dataset. Each of these models was trained with a different batch size. All models were trained for 30 epochs.

Total training time: ~ 
