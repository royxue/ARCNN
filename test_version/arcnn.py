import cPickle
import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer

class ARCNNConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(numpy.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
            dtype=theano.config.floatX),
                               borrow=True)

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=input, filters=self.W,
                filter_shape=filter_shape, image_shape=image_shape)

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(input=conv_out,
                                            ds=poolsize, ignore_border=True)

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]


def train_ARCNN(learning_rate, n_epochs, nkerns, dataset, batch_size):
    
    rng = numpy.random.RandomState(23455)

    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    # test_set_x, test_set_y = datasets[2]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches /= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ishape = (640, 480)  # this is the size of  images

    print '... building the model'

    # Reshape matrix of rasterized images of shape (batch_size,160*120)
    # to a 4D tensor, compatible with our ARCNNConvPoolLayer
    layer0_input = x.reshape((batch_size, 1, 160, 120))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (160-9+1,120-9+1)=(152,112)
    # maxpooling reduces this further to (152/2,112/2) = (76,56)
    # 4D output tensor is thus of shape (batch_size,nkerns[0],76,56)
    layer0 = ARCNNConvPoolLayer(rng, input=layer0_input,
            image_shape=(batch_size, 1, 160, 120),
            filter_shape=(nkerns[0], 1, 9, 9), poolsize=(2, 2))

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (76-7+1,56-7+1)=(70,50)
    # maxpooling reduces this further to (70/2,50/2) = (35,25)
    # 4D output tensor is thus of shape (nkerns[0],nkerns[1],35,25)
    layer1 = LeNetConvPoolLayer(rng, input=layer0.output,
            image_shape=(batch_size, nkerns[0], 76, 56),
            filter_shape=(nkerns[1], nkerns[0], 7, 7), poolsize=(2, 2))

    # filtering reduces the image size to (35-4+1,25-4+1)=(32,22)
    # maxpooling reduces this further to (32/2,22/2) = (16,11)
    # 4D output tensor is thus of shape (nkerns[1],nkerns[2],16,11)
    layer2 = LeNetConvPoolLayer(rng, input=layer1.output,
            image_shape=(batch_size, nkerns[1], 35, 25),
            filter_shape=(nkerns[2], nkerns[1], 4, 4), poolsize=(2, 2))

    # filtering reduces the image size to (16-2+1,11-2+1)=(15,10)
    # maxpooling reduces this further to (70/2,/2) = (7,5)
    # 4D output tensor is thus of shape (nkerns[2],nkerns[3],35,25)
    layer3 = LeNetConvPoolLayer(rng, input=layer2.output,
            image_shape=(batch_size, nkerns[2], 16, 11),
            filter_shape=(nkerns[3], nkerns[2], 3, 3), poolsize=(2, 2))

    # filtering reduces the image size to (7-2+1,5-2+1)=(6,4)
    # maxpooling reduces this further to (7/2,5/2) = (3,2)
    # 4D output tensor is thus of shape (nkerns[3],nkerns[4],3,2)
    layer4 = LeNetConvPoolLayer(rng, input=layer3.output,
            image_shape=(batch_size, nkerns[3], 7, 5),
            filter_shape=(nkerns[4], nkerns[3], 2, 2), poolsize=(2, 2))

    # filtering reduces the image size to (3-1+1,2-1+1)=(3,2)
    # maxpooling reduces this further to (3/2,2/2) = (1,1)
    # 4D output tensor is thus of shape (nkerns[4],nkerns[5],1,1)
    layer5 = LeNetConvPoolLayer(rng, input=layer4.output,
            image_shape=(batch_size, nkerns[4], 3, 2),
            filter_shape=(nkerns[5], nkerns[4], 1, 1), poolsize=(2, 2))

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size,num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (20,32*4*4) = (20,512)
    layer6_input = layer5.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer6 = HiddenLayer(rng, input=layer5_input, n_in=nkerns[1] * 4 * 4,
                         n_out=500, activation=T.tanh)

    # classify the values of the fully-connected sigmoidal layer
    layer7 = LogisticRegression(input=layer5.output, n_in=500, n_out=10)

    # the cost we minimize during training is the NLL of the model
    cost = layer7.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function([index], layer7.errors(y),
             givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size],
                y: test_set_y[index * batch_size: (index + 1) * batch_size]})

    validate_model = theano.function([index], layer7.errors(y),
            givens={
                x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                y: valid_set_y[index * batch_size: (index + 1) * batch_size]})

    # create a list of all model parameters to be fit by gradient descent
    params = layer7.params + layer6.params + layer5.params + layer4.params + 
            layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)
