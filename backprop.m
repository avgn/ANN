function [nabla_b, nabla_w] = backprop (x, y, num_layers, sizes, biases, weights)
  % Inputs:
  % - x: values of the input layer
  % - y: correct output
  % - num_layers, sizes: dimensions of the network
  % - biases, weights: biases and weights to backpropagate the errors

  % Outputs:
  % - nabla_b: update for the gradient of biases
  % - nabla_w: update for the gradient of weights

  % define the sigmoid (logistic) function
  sigmoid = inline ("1.0/(1.0+exp(-z))");
  sigmoid_vec = vectorize(sigmoid);
  % define the first derivative of the sigmoid function
  sigmoid_prime = inline ("sigmoid(z)*(1-sigmoid(z))");
  sigmoid_prime_vec = vectorize(sigmoid);
  % define the first derivative of the cost function
  cost_derivative = inline ("output_activations - y", "output_activations", "y");

  % initialize the gradients for weights and biases
  nabla_b = zeros(size(biases));
  nabla_w = zeros(size(weights));

  % feedforward part
  % set the activations for the input layer to the input values
  activation = x';
  % define the vector containing all activations layer by layer
  activations = x';
  % define the vector containing all z = w*x + b layer by layer
  zs = [];

  % initialize parameters to use to reshape the biases and weights vectors
  w_current = 1;
  w_next = sizes(1)*sizes(2);
  b_current = 1;
  b_next = sizes(2);

  % feedforward loop
  for i = 1:num_layers-1
    % reshape weights from vector to a matrix
    w = reshape(weights(w_current:w_next), [sizes(i), sizes(i+1)])';
    % set the biases to use
    b = biases(b_current:b_next);
    % update reshape parameters
    if i < num_layers - 1
      w_current = w_current + sizes(i)*sizes(i+1);
      w_next = w_current - 1 + sizes(i+1)*sizes(i+2);
      b_current = b_current + sizes(i+1);
      b_next = b_current - 1 + sizes(i+2);
    end
    % compute z and the activation function, and update the vectors to track them
    z = w*activation + b;
    zs = [zs; z];
    activation = sigmoid_vec(z);
    activations = [activations; activation];
  end

  % backward pass
  % initialize parameters to use to reshape the vectors
  w_current = length(weights);
  w_next = length(weights) - sizes(end)*sizes(end-1) + 1;
  b_current = length(biases);
  b_next = length(biases) - sizes(end) + 1;
  a_current = length(activations);
  a_next = length(activations) - sizes(end) + 1;
  % define the vector with output size to transform each output y in a logical vector
  c = (1:sizes(end))';
  % compute the errors delta in the output layer
  delta = cost_derivative(activations(a_next:a_current), c==y) .* sigmoid_prime_vec(zs(b_next:b_current));
  % update nabla_b and nabla_w with the errors
  nabla_b(b_next:b_current) = delta;
  nabla_w(w_next:w_current) = (delta*activations((a_next-sizes(end-1)):(a_next-1))')'(:); % outer product
  % backpropagate the errors by looping through layers backward
  for j = (num_layers-1):-1:2
    % update reshape parameters
    if j > 1
      b_current = b_next - 1;
      b_next = b_next - sizes(j);
      a_current = a_next - 1;
      a_next = a_next - sizes(j);
    end
    % backpropagate the errors
    z = zs(b_next:b_current);
    spv = sigmoid_prime_vec(z);
    w = reshape(weights(w_next:w_current), [sizes(j), sizes(j+1)])';
    delta = w'*delta .* spv;
    % update nabla_b and nabla_w with the backrpopagated errors
    nabla_b(b_next:b_current) = delta;
    % update reshape parameters
    if j > 1
      w_current = w_next - 1;
      w_next = w_next - sizes(j)*sizes(j-1);
    end
    nabla_w(w_next:w_current) = (delta*activations((a_next-sizes(j-1)):(a_next-1))')'(:);
  end
end
