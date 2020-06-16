% param = fcnPCA(X, prc) trains the PCA
% Inputs
%   X: training data for dimensionality reduction.
%   prc: percentage of dimensional size of principal components to explain the variability
%
% Outputs
%   param: structure of PCA parameters
%   param.mu: the mean of training data
%   param.w: the coefficients of principal components
%   param.ind: the number of dimensions

function param                  = fcnPCA(X, dim)


param.mu                        = mean(X,1);

adj                             = X - param.mu;

[U,D,~]                         = svd(cov(adj));

if nargin < 2
    evalue                          = diag(D);
    explained                       = cumsum(evalue)./sum(evalue);
    sum_explained                   = 0;
    dim                             = 0;
    while sum_explained < 95
        dim                         = dim + 1;
        sum_explained               = sum_explained + explained(dim);
    end
    param.ind                       = 1 : dim;
else
    param.ind                       = 1 : dim;
end
param.w                         = U(:,param.ind);
param.fcn                       = @(X)(X - param.mu) * param.w;
