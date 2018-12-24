function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%


g = sigmoid(X * theta); % Use the sigmoid function. Z = X * theta

% Subsitute g in place of h(x)
% Transpose -y and (1-y) for vectorization
% sizes are as follows size(-y) = 100 * 1, size(log(g)) is 100 * 1, size of log(1-g) is 100 *1
% J = (1/100) * [ [1 * 100] * [100 * 1] -  [1 * 100] * [100 * 1] ]
% J is a scalar
J = (1/m) * [ transpose(-1 .* y) * log(g) - transpose(1 - y) * log(1 - g) ];

% grad = (1/100) * [3 * 100] * ([100 * 1] - [100 * 1])
% size(grad) = [3, 1]
grad = (1/m) * transpose(X) * (g - y) ;








% =============================================================

end
