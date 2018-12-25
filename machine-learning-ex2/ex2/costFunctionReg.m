function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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



% Vectorization of J = (1/118) * [ [1 , 100] * [100 , 1] -  [1 , 100] * [100 , 1] ] + (1/2*118) * [ [1 , 27] * [27 , 1]]
%J = (1/m) * [transpose(-1 * y) * log(g) - transpose(1 - y) * log(1 - g)] + (lambda ./ 2 * m) * [transpose(theta) * theta] ;
% Separate theta_1 and remaining theta in order to avoid applying regularization to theta_1
theta_1 = theta(1); % single value
X_1 = X(:,1);

theta_rem = theta(2:size(theta,1),:); % vector with size one less than theta vector i.e.  [27, 1]
X_rem = X(:,2:size(X,2));

g = sigmoid(X * theta); 
g_1 = sigmoid(X_1 * theta_1);
%J_1 = (1/m) * [(transpose(-1 * y) * log(g_1)) - (transpose(1 - y) * log(1 - g_1))];

g_rem = sigmoid(X_rem * theta_rem);
%J_rem =  (1/m) * [(transpose(-1 * y) * log(g_rem)) - (transpose(1 - y) * log(1 - g_rem))] + [(lambda / (2 * m)) * [transpose(theta_rem) * theta_rem]] ;

%J = J_1 + J_rem;

J =  [(1/m) * [(transpose(-1 * y) * log(g)) - (transpose(1 - y) * log(1 - g))]] + [(lambda / (2 * m)) * (transpose(theta_rem) * theta_rem)] ;


% Calculate gradient for theta_1
% Note that the sigmoid function is called with theta_1
% Vectorization of grad_1 = (1/118) * [1, 118] * ([118, 1] * [1, 1] - [118, 1])
grad_1 = (1/m) * (transpose(X_1) * (g_1 - y));


% Calculate gradient for remaining theta values
% Note that the sigmoid function is called with theta_rem
% Note the regularization is applied
% Vectorization of grad_1 = (1/118) * [27, 118] * ([118, 27] * [27, 1] - [118, 1])
grad_rem = [ (1/m) * [transpose(X_rem) * (g_rem - y)] ] + [(lambda / m) * theta_rem];

% Combine the gradients to arrive at the final gradient
grad = [grad_1; grad_rem];




% =============================================================

end
