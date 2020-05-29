function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

%using method of iterating thru all X points
%for i=1:size(idx,1)
%  minCentroidDist=9999999;
%  centroidIdx=0;
  
%  for j=1:K
%    diff=X(i,:)-centroids(j,:);
%    mag=sqrt(diff*diff'); %make sure it adds all dimensions of diff
%    centroidDist=mag^2;
    
%    if (centroidDist <= minCentroidDist)
%      minCentroidDist=centroidDist;
%      centroidIdx=j;
%    end
%  end
  
%  idx(i)=centroidIdx;
%end

%using method of creating a distance-to-centroids table (quicker)
dist = zeros(size(X,1), K);

%iterate through each centroid and calc distances
for j=1:K
  %find diff between each X point and centroid j
  diff = bsxfun(@minus, X, centroids(j,:));
  %calc sum of the squares and save to dist table
  dist(:,j)=sum(diff.^2,2);
end

%find min values (and return vector of those indices)
[minValues, idx] = min(dist, [], 2);


% =============================================================

end

