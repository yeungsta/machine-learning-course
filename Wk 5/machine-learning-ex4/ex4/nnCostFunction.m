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

%calc hidden layer (layer 2) units
%add bias unit column of 1's to inputs
X = [ones(m, 1) X];
a1=X;

z2 = X*Theta1'; %m x (input_layer_size + 1) * (input_layer_size + 1) x hidden_layer_size
a2 = sigmoid(z2); %m x hidden_layer_size

%calc output layer (layer 3) units
%add bias unit column of 1's to layer 2 units
a2 = [ones(m, 1) a2];
z3 = a2*Theta2'; %m x (hidden_layer_size + 1) * (hidden_layer_size + 1) x num_labels
a3 = sigmoid(z3); %m x num_labels

H=a3; %m x num_labels
%fprintf('H: %f \n', size(H));

%for each training value of the set
for i = 1:m
  %transform y label to bit vector
  yvec = zeros(1,num_labels);
  yvec(y(i))=1;
  
  y1=-1*yvec*log(H(i,:)'); %1 x num_labels * num_labels x 1 = 1x1; 
  y2=(-1*(1-yvec))*log(1-H(i,:)'); %1 x num_labels * num_labels x 1 = 1x1; 
  
  J=J+((1/m)*(y1+y2)); %scalars
  
  %now back-propagate to calculate deltas for layer 2 and 3:
  %calc layer 3 deltas
  d3 = a3(i,:)-yvec; %1 x num_labels - 1 x num_labels = 1 x num_labels

  %calc layer 2 deltas
  d2 = Theta2'*d3'; %(hidden_layer_size + 1) x num_labels * num_labels x 1 = (hidden_layer_size + 1) x 1
  %drop bias term
  d2=d2(2:end);
  
  %calculating g' using a2 prevents having to re-run sigmoid() function already done at cost step
  gDeriv = a2(i,2:end).*(1-a2(i,2:end)); %1 x hidden_layer_size (skip first bias column)
  %gDeriv = sigmoidGradient(z2(i,:)); %1 x hidden_layer_size
  
  %element-wise multiplication
  d2=d2.*gDeriv';
  
  %for each layer, multiply+sum activation unit and delta of next layer. Accumulate gradients.
  %layer 1
  Theta1_grad = Theta1_grad + d2*a1(i,:); %hidden_layer_size x 1 * 1 x (input_layer_size + 1) = hidden_layer_size x (input_layer_size + 1)
  %layer 2
  Theta2_grad = Theta2_grad + d3'*a2(i,:); %num_labels x 1 * 1 x (hidden_layer_size + 1) = num_labels x (hidden_layer_size + 1)
end

%Obtain the (unregularized) gradient for the neural network cost function by dividing the accumulated gradients by m:
Theta1_grad = Theta1_grad./m; %hidden_layer_size x (input_layer_size + 1) / scalar
Theta2_grad = Theta2_grad./m; %num_labels x (hidden_layer_size + 1) / scalar

%Compute regularization components for cost J; skip theta indices 0; used for bias term
Theta1(:,1)=0;
for i = 1:size(Theta1,1)
  J=J+(lambda*(1/(2*m)))*(Theta1(i,:)*Theta1(i,:)'); %scalar * ((1 x n) * (n x 1) = scalar)
end

Theta2(:,1)=0;
for i = 1:size(Theta2,1)
  J=J+(lambda*(1/(2*m)))*(Theta2(i,:)*Theta2(i,:)'); %scalar * ((1 x n) * (n x 1) = scalar)
end

%Compute regularization components for accumulated gradients; skip theta indices 0; used for bias term
Theta1_grad = Theta1_grad.+(lambda/m)*Theta1;
Theta2_grad = Theta2_grad.+(lambda/m)*Theta2;





  
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
