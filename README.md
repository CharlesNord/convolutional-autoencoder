# convolutional-autoencoder

Re-implementation of code in this link: https://github.com/XifengGuo/DCEC/blob/master/ConvAE.py
With little modification, sigmoid activation function is added on the top of model, thus we can output images with pixel vaule between 0 and 1, and we can apply the binary crossentropy loss:
[[https://github.com/CharlesNord/convolutional-autoencoder/blob/master/gif.gif]]

Here the crossentropy is averaged by the number of pixels, just like mse loss.
