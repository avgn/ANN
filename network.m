function [num_layers, biases, weights] = network (sizes)
  % Input:
  % - sizes: vector containing the dimension of each layer s.t.
  %   number of layers = length(sizes), dim of input layer = sizes(1),
  %   dim of first hidden layer = sizes(2), dim of output layes = sizes(end)

  % define the number of layers of the network
  num_layers = length(sizes);
  % initialize biases and weights vectors
  biases = [];
  weights = [];
  % initialize weights and biases with values from a standard normal distribution
  for i = 1:num_layers-1
    y = sizes(i+1);
    x = sizes(i);
    % the (:) after the stdnormal generation is used to unroll the matrix into a vector
    biases = [biases; stdnormal_rnd(y, 1)(:)];
    weights = [weights; stdnormal_rnd(y, x)(:)];
  end
end
