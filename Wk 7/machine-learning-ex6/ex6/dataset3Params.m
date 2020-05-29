function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 999;
sigma = 999;

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

Cvec=[0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
sigmaVec=[0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
%Cvec=[0.01; 0.03];
%sigmaVec=[0.01; 0.03];

lowestError=999;

for i = 1:size(Cvec, 1)
  for j = 1:size(sigmaVec, 1)
    %train model w/ parameters on training data
    model= svmTrain(X, y, Cvec(i), @(x1, x2) gaussianKernel(x1, x2, sigmaVec(j)));

    %use model to make predictions on cross-validation data
    predictions = svmPredict(model, Xval);

    %calc error between trained model and cross-validation truth data
    error=mean(double(predictions ~= yval))

    %record C/sigma combo w/ the smallest error
    if (error < lowestError)
      lowestError = error;
      C = Cvec(i);
      sigma = sigmaVec(j);  
    end
  end
end

% =========================================================================

end
