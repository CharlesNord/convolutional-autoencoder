# onvolutional-autoencoder

## cae
Re-implementation of code in this link: https://github.com/XifengGuo/DCEC/blob/master/ConvAE.py

With little modification, sigmoid activation function is added on the top of model, thus we can output images with pixel vaule between 0 and 1, and we can apply the binary crossentropy loss:

![crossentropy](https://github.com/CharlesNord/convolutional-autoencoder/blob/master/images/gif.gif)

Here the crossentropy is averaged by the number of pixels, just like mse loss.

## cae2
Use fully convolutional layer instead of fully connected layer.(i.e. removed the steps of reshaping feature maps). 

The bottle neck features are in 2D space, so that we can directly visualize them, but the performance is not satisfying, after 20 epochs of iteration, we got the following clustering result and reconstruction result.
### 2D-CAE reconstruction
![reconstruction](https://github.com/CharlesNord/convolutional-autoencoder/blob/master/images/reconstruction_20.png)
### 2D-CAE visualization
![clustering](https://github.com/CharlesNord/convolutional-autoencoder/blob/master/images/scatter_20.png)


