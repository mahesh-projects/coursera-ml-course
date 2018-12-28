function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Add ones to the X data matrix i.e. Bias Unit 
X = [ones(m, 1) X];

% Input layer a1 is the training examples X
a1 = X;
% Hidden layer a2 = sigmoid activation function with Theta1 and inputs from Input layer 
z2 = a1 * transpose(Theta1);
a2 = sigmoid(z2); 

% Add bias unit to hidden layer  
a2 = [ones(m, 1) a2];

% Output layer unit a3 = sigmoid activation function with Theta2 and inputs from Hidden layer 
z3 = a2 * transpose(Theta2); 
predict = sigmoid(z3);

% Use max function to return the max element and index of max element
[predict_max, index_max] = max(predict, [], 2);

% Set vector to index_max vector
p = index_max;


% =========================================================================


end
