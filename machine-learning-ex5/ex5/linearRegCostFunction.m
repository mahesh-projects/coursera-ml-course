function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% Compute un-regularized cost and gradient
J_unregularized = (1/(2 * m)) * sum(((X * theta) - y ) .^ 2);

grad_unregularized = (1 / m) * (transpose (X) * (X * theta - y));

% Compute regularization term for cost and grad. 
% Set theta(1) to 0 in order to NOT regularize theta(1)

theta(1) = 0;

J_regularization_term = (lambda / (2 * m)) * sum(theta.^2);

grad_regularization_term = (lambda / m) .* (theta);

% Sum unregularized and regularized terms
J = J_unregularized + J_regularization_term;

grad = grad_unregularized + grad_regularization_term;

% =========================================================================

grad = grad(:);

end


% Unit Test - just run with "test linearRegCostFunction"
%!test
%! X = [[1 1 1]' magic(3)];
%! y = [7 6 5]';
%! theta = [0.1 0.2 0.3 0.4]';
%! lambda = 0;
%! [J0 g0] = linearRegCostFunction(X, y, theta, lambda);
%! [J7 g7] = linearRegCostFunction(X, y, theta, 7);
%! Jexpected_lambda0 = 1.3533;
%! Gexpected_lambda0 = [-1.4000; -8.7333; -4.3333; -7.9333;];
%! Jexpected_lambda7 = 1.6917;
%! Gexpected_lambda7 = [-1.4000; -8.2667; -3.6333; -7.0000;];
%! assert(J0, Jexpected_lambda0, .0001);
%! assert(g0, Gexpected_lambda0, .0001);
%! assert(J7, Jexpected_lambda7, .0001);
%! assert(g7, Gexpected_lambda7, .0001);