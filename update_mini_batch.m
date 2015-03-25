function [biases, weights] = update_mini_batch (mini_batch, mini_batch_size, learn_rate, num_layers, sizes, biases, weights)
  % Inputs:
  % - mini_batch: mini batch to use for the update
  % - mini_batch_size: size of the mini_batches to use when sampling
  % - learn_rate: learning rate
  % - num_layers, sizes: dimensions of the network
  % - biases, weights: biases and weights to update

  % Outputs:
  % - biases: updated biases
  % - weights: updated weights

  % initialize the gradients of biases and weights
  nabla_b = zeros(size(biases));
  nabla_w = zeros(size(weights));

  % loop through each row of mini_batch
  for j = 1:mini_batch_size
    % use the backpropagation algorithm to compute the gradients
    [delta_nabla_b, delta_nabla_w] = backprop (mini_batch(j, 2:end), mini_batch(j, 1), num_layers, sizes, biases, weights);
    nabla_b = nabla_b + delta_nabla_b;
    nabla_w = nabla_w + delta_nabla_w;
  end
  % update biases and weights
  biases = biases - (learn_rate/mini_batch_size).*nabla_b;
  weights = weights - (learn_rate/mini_batch_size).*nabla_w;
end
