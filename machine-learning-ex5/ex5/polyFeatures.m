function [X_poly] = polyFeatures(X, p)
%POLYFEATURES Maps X (1D vector) into the p-th power
%   [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
%   maps each example into its polynomial features where
%   X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];
%


% You need to return the following variables correctly.
X_poly = zeros(numel(X), p);

% ====================== YOUR CODE HERE ======================
% Instructions: Given a vector X, return a matrix X_poly where the p-th 
%               column of X contains the values of X to the p-th power.
%
% 

for col = 1:p
    X_poly(:,col) = X(:,1) .^ col;
end


% =========================================================================

end

% Unit Test - just run with "test linearRegCostFunction"
%!test
%! A = [2;3;4];
%! p = 3;
%! A_poly = polyFeatures(A,p);
%! A_poly_expected = [2, 4, 8; 3, 9, 27; 4, 16, 64];
%! assert(A_poly, A_poly_expected);