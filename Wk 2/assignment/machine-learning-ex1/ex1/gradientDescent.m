function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    %calculate new thetas
    H=X*theta; %m x 2 * 2 x 1
    Diff=H-y; %m x 1 - m x 1
    Sum=Diff'*X;  %1 x m * m x 2
    GradientDiff=alpha*(1/m)*Sum; %1 x 2 * scalar

    %update theta 0
    theta(1)=theta(1)-GradientDiff(:,1);

    %update theta 1
    theta(2)=theta(2)-GradientDiff(:,2);

    %compute cost
    cost = computeCost(X, y, theta);
    %fprintf('\nIter: %d, Cost: %f', iter, cost);

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = cost;

end

end
