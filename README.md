# convolutional-autoencoder

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


## cae tsne
Encode the image to 10D vector, and use tsne to visualize it
### 10D-CAE reconstruction
![10D-reconstruction](https://github.com/CharlesNord/convolutional-autoencoder/blob/master/images/reconstruction_10D.png)
### 10D-CAE tsne visualization
![10D-tsne](https://github.com/CharlesNord/convolutional-autoencoder/blob/master/images/10D_tsne.png)


## cvae
Convolutional variational autoencoder
Encode the image to 10D vector, and visualize it by tsne
### 10D-CVAE reconstruction
![10D-cvae-reconstruction](https://github.com/CharlesNord/convolutional-autoencoder/blob/master/images/10D_CVAE_recon.png)
### 10D-CVAE tsne visualization
![10D-cvae-tsne](https://github.com/CharlesNord/convolutional-autoencoder/blob/master/images/10D_CVAE.png)

## conditional_conv_vae
Modified based on convolutional variational autoencoder, I tested only input label in decoder and input label both in encoder and decoder, here is the result.
### input label only in decoder
![only in decoder](https://github.com/CharlesNord/convolutional-autoencoder/blob/master/images/conditional_conv_vae_sample_20.png)
### input label in both encoder and decoder
![both in decoder and encoder](https://github.com/CharlesNord/convolutional-autoencoder/blob/master/images/2_conditional_conv_vae_sample_3.png)

We could easily find that the second is better, and this is what most conditional vae algorithms do.
