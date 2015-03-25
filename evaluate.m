function test = evaluate (test_data, num_layers, sizes, biases, weights)
  % Inputs:
  % - test_data: data for evaluation
  % - num_layers, sizes: dimensions of the network
  % - biases, weights: biases and weights to feedforward

  % Output:
  % - test: number of correct evaluations

  % initialize test values
  X = test_data(:, 2:end);
  y = test_data(:, 1);
  % loop through test data
  for i = 1:size(test_data)(1)
    % store in test_results the guessed output
    [res(i), test_results(i)] = max(feedforward(X(i, :), num_layers, sizes, biases, weights));
  end
  % transpose to a column vector to confront with actual y
  test_results = test_results';
  % confront evaluations and correct outputs, and sum the number of correct evaluations
  test = sum(test_results == y);
end
