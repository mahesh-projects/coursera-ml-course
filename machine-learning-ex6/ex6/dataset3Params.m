function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
C_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';
sigma_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';

%C_vec = [0.01 0.03]';
%sigma_vec = [0.01 0.03]';
results = [];

%outer loop - iterate through C values
for i = 1:length(C_vec)
    C_test = C_vec(i);
    %inner loop - iterate through sigma values
    for j = 1:length(sigma_vec)
        sigma_test = sigma_vec(j);
        model = svmTrain(X, y, C_test, @(x1, x2) gaussianKernel(x1, x2, sigma_test)); 
        %Compute the predictions for the validation set using svmPredict() with model and Xval.
        predictions = svmPredict(model, Xval);
        err_value = mean(double(predictions ~= yval));
        results = [results; C_test sigma_test err_value];
    end
end

[min_values, index_min_values] = min(results);

%Pick C and sigma that corresponds to the minimum err_value for the validation set
C = results(index_min_values(3), 1)
sigma = results(index_min_values(3),2)






% =========================================================================

end
