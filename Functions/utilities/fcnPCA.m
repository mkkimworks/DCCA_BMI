function param                  = fcnPCA(X, k)

[coeff,~,~,~,~,mu]              = pca(X);
param.mu                        = mu;
param.w                         = coeff(:,1 : k);
param.fcn                       = @(X)(X - param.mu) * param.w;
