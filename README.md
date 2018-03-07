# convolutional-autoencoder

Re-implementation of code in this link: https://github.com/XifengGuo/DCEC/blob/master/ConvAE.py
With little modification, sigmoid activation function is added on the top of model, thus we can output images with pixel vaule between 0 and 1, and we can apply the binary crossentropy loss:

$$
binaryCrossentropy=\frac{1}{m}\sum_{i=1}^m(y_{label}\mathrm{log}(y_{pred})+(1-y_{label}\mathrm{log}(1-y_pred)))
$$
Here the crossentropy is averaged by the number of pixels, just like mse loss.
