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

fprintf('Theta1: %f \n', size(Theta1)); %hidden_layer_size x input_layer_size + 1
fprintf('Theta2: %f \n', size(Theta2)); %num_labels x hidden_layer_size + 1

%calc hidden layer (layer 2) units
%add bias unit column of 1's to inputs
X = [ones(m, 1) X];
z2 = X*Theta1'; %m x (input_layer_size + 1) * (input_layer_size + 1) x hidden_layer_size
a2 = sigmoid(z2); %m x hidden_layer_size

%calc output layer (layer 3) units
%add bias unit column of 1's to layer 2 units
a2 = [ones(m, 1) a2];
z3 = a2*Theta2'; %m x (hidden_layer_size + 1) * (hidden_layer_size + 1) x num_labels
a3 = sigmoid(z3); %m x num_labels

%take max of each row and set as the prediction for each data point
[max_value, p] = max(a3, [], 2);


% =========================================================================


end
