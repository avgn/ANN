function a = feedforward (a, num_layers, sizes, biases, weights)
  % Inputs:
  % - a: values of the input layer
  % - num_layers, sizes: dimensions of the network
  % - biases, weights: biases and weights to feedforward

  % Output:
  % - a = activations of the output layer

  % define the sigmoid (logistic) function
  sigmoid = inline ("1.0/(1.0+exp(-z))");
  sigmoid_vec = vectorize(sigmoid);

  % compute the output of the network
  % initialize the input layer
  a = a';
  % initialize parameters to use to reshape the biases and weights vectors
  w_current = 1;
  w_next = sizes(1)*sizes(2);
  b_current = 1;
  b_next = sizes(2);
  % feedforward loop
  for i = 1:num_layers-1
    % reshape weight and biases
    w = reshape(weights(w_current:w_next), [sizes(i), sizes(i+1)])';
    b = biases(b_current:b_next);
    % update reshape parameters
    if i < num_layers - 1
      w_current = w_current + sizes(i)*sizes(i+1);
      w_next = w_current - 1 + sizes(i+1)*sizes(i+2);
      b_current = b_current + sizes(i+1);
      b_next = b_current - 1 + sizes(i+2);
    end
    % compute the activation function
    a = sigmoid_vec(w*a + b);
  end
end
