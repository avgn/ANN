function SGD (num_layers, sizes, biases, weights, training_data, epochs, mini_batch_size, learn_rate, test_data)
  % Inputs:
  % - num_layers, sizes: dimensions of the network
  % - biases, weights: initialized biases and weights
  % - training_data: matrix with labels y in the first column
  % - epochs: number of epochs to train for
  % - mini_batch_size: size of the mini_batches to use when sampling
  % - learn_rate: learning rate
  % - test_data: data for evaluation

  % Output:
  % If test_data are provided, it prints out the number of correct digit
  % classification over the the dimension of test_data.
  % If test_data not provided, it prints out the state of the training process.

  % if test_data is present
  if nargin > 8
    n_test = size(test_data)(1);
  end
  % define the dimension of training data
  n = size(training_data)(1);

  % loop through each epoch
  for i = 1:epochs
    % shuffle training data
    perm = randperm(n);
    training_data = training_data(perm', :);
    % loop through each mini_batch
    for j = 1:mini_batch_size:n
      [biases, weights] = update_mini_batch (training_data(j:j+mini_batch_size-1, :), mini_batch_size, learn_rate, num_layers, sizes, biases, weights);
    end

    % if test_data is present
    if nargin > 8
      % print the ratio of correct digit classification
      fprintf("Epoch %d: %d / %d\n", i, evaluate(test_data, num_layers, sizes, biases, weights), n_test);
    else
      fprintf("Epoch %d complete\n", i);
    end
  end
end
