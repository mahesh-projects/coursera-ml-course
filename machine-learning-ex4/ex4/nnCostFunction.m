function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
% ============================================
% Part 1: Vectorized Feedforward implementation to get predictions for all training examples
% ============================================
% Reference: https://www.coursera.org/learn/machine-learning/programming/AiHgN/neural-network-learning/discussions/threads/QFnrpQckEeWv5yIAC00Eog

% Expand the 'y' output values into a matrix of single values (see ex4.pdf Page 5)
y_matrix = eye(num_labels)(y,:);

% Add column of 1's to the X matrix i.e. Bias Unit 
X = [ones(m, 1) X];

% -------------------------------------------------------------
% Perform Foward Propogation for 3 layered neural network:
% -------------------------------------------------------------
% Input layer a1 equals X input matrix with a column of 1's added (bias units) as the first column.
a1 = X;
% z2​ equals the product of a1 and Theta_1
% Hidden layer a2 = sigmoid activation function. a2​ is the result of passing z2 through g() 
z2 = a1 * transpose(Theta1);
a2 = sigmoid(z2); 

% add a column of bias units to a2 (as the first column).
a2 = [ones(m, 1) a2];

% Output layer a3 = sigmoid activation function. a2​ is the result of passing z3 through g() 
z3 = a2 * transpose(Theta2); 
a3 = sigmoid(z3); % predicted values stored in predict

% Calculated Unregularized Cost.
% Remember to use element-wise multiplication with the log() function. For a discussion of why you can't (easily) use matrix multiplication here, see this thread:
% https://www.coursera.org/learn/machine-learning/discussions/weeks/5/threads/ag_zHUGDEeaXnBKVQldqyw
% https://www.coursera.org/learn/machine-learning/discussions/all/threads/AzIrrO7wEeaV3gonaJwAFA 
% Double summation in Octave - Reference http://sachinashanbhag.blogspot.com/2010/02/double-summation-in-gnu-octave-or.html 
J_unregularized = (1/m) * sum(sum([ (-1 .* y_matrix) .* (log(a3)) - (1 - y_matrix) .* (log(1 - a3)) ]));

% -------------------------------------------------------------
% Calculate regularization cost
% -------------------------------------------------------------

% Drop the bias units from Theta1 and Theta2 as we do not regularize bias units
Theta1_minus_bias = Theta1(:,2:end);
Theta2_minus_bias = Theta2(:,2:end);

% Use element wise power operator to square Theta1_minus_bias and Theta2_minus_bias
% Use double summation to sum over 
J_regularized = (lambda / (2 * m)) * [ sum(sum(Theta1_minus_bias .^ 2)) + sum(sum(Theta2_minus_bias .^ 2)) ];

% Cost J is sum of unregularized and regularized components
J = J_unregularized + J_regularized;

% =========================================================================
%Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad% 
% Let:

% m = the number of training examples

% n = the number of training features, including the initial bias unit.

% h = the number of units in the hidden layer - NOT including the bias unit

% r = the number of output classifications 
% =========================================================================



% Error term (d3) for 3rd layer i.e. output layer is difference between predicted values and actual values
d3 = a3 - y_matrix;


% Error term (d2) for 2nd layer i.e. hidden layer is weighted average of the error terms of the nodes in layer (l + 1)
% (m x r) ⋅ (r x h) --> (m x h)
d2 = (d3 * Theta2_minus_bias) .* sigmoidGradient(z2);

% Delta1 is the product of d2 and a1. The size is (h x m) ⋅ (m x n) --> (h x n)
Delta1 = transpose(d2) * a1;

% Delta2 is the product of d3 and a2. The size is (r x m) ⋅ (m x [h+1]) --> (r x [h+1])
Delta2 = transpose(d3) * a2;

% Theta1_grad and Theta2_grad are the same size as their respective Deltas, just scaled by 1/m
Theta1_grad = (1/m) * Delta1;
Theta2_grad = (1/m) * Delta2;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
