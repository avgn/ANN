clear; close all; clc

% define the size of the network s.t.
% number of layers = length(sizes), dim of input layer = sizes(1),
% dim of first hidden layer = sizes(2), dim of output layes = sizes(end)
sizes = [784, 30, 10];
% create the network
[num_layers, biases, weights] = network (sizes);

% load training data
images = loadMNISTImages('train-images-idx3-ubyte');
labels = loadMNISTLabels('train-labels-idx1-ubyte');
% set 0 labels to 10 to avoid numbering confusion
labels(labels==0) = 10;
training_data = [labels, images'];

clear labels images;

%load test data
images_test = loadMNISTImages('t10k-images-idx3-ubyte');
labels_test = loadMNISTLabels('t10k-labels-idx1-ubyte');
% set 0 labels to 10 to avoid numbering confusion
labels_test(labels_test==0) = 10;
test_data = [labels_test, images_test'];

clear labels_test images_test;

% definine the parameters for the stochastic gradient descent algorithm
epochs = 10;
mini_batch_size = 10;
learn_rate = 3.0;

% run the stochastic gradient descent algorithm to train and evaluate the network
SGD (num_layers, sizes, biases, weights, training_data, epochs, mini_batch_size, learn_rate, test_data)
