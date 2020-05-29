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
H=X*theta; % 12x2 * 2x1 
diff=H-y;
J=(1/(2*m))*(diff'*diff);

#calc the regularization parameter
#don't use first theta
theta(1)=0;
#add on regularization parameter
J=J+(lambda/(2*m))*(theta'*theta);

#calc gradients
sum=diff'*X; % 1x12 * 12x2
grad=(1/m)*sum; % 1x2
#add on regularization param
grad=grad'+(lambda/m)*theta; % 2x1 + 2x1


% =========================================================================

grad = grad(:);

end
