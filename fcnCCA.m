%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   
%   Title: Decoding kinematic information from primary motor cortical 
%          ensemble activity using a deep canonical correlation analysis
%   Authors: M-K Kim; J-W Sohn; S-P Kim
%   E-mail: mkkim.works@gmail.com
%   Affiliation: Ulsan National Institute of Science and Technology (UNIST)
%
%   This file is derived from cononcorr (2018/07/19)
%   Copyright 1993-2014 The MathWorks, Inc.
% 
%   Modifications by UNIST, April 2020
%
%   param          = fcnCCA(X,Y)
%   Inputs 
%       X: random variable of view 1
%       Y: random variable of view 2
%   Outputs
%       saveStruct: optimized trained-DCCA parameters
%       BayesObject: Bayesian optimization output
%
%   References:
%     [1] Krzanowski, W.J., Principles of Multivariate Analysis,
%         Oxford University Press, Oxford, 1988.
%     [2] Seber, G.A.F., Multivariate Observations, Wiley, New York, 1984.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function param              = fcnCCA(X1,X2)

[N,P1]                      = size(X1);
if size(X2,1) ~= N
    error(message('stats:canoncorr:InputSizeMismatch'));
elseif N == 1
    error(message('stats:canoncorr:NotEnoughData'));
end
P2                          = size(X2,2);

% Centralize the random variables
mu1                         = mean(X1,1);
mu2                         = mean(X2,1);
X1                          = X1 - repmat(mu1, N, 1);
X2                          = X2 - repmat(mu2, N, 1);

% QR decomposed X1
[Q1,T1,perm1]               = qr(X1,0);
rankX                       = sum(abs(diag(T1)) > eps(abs(T1(1)))*max(N,P1));
if rankX < P1
    Q1 = Q1(:,1:rankX); T1 = T1(1:rankX,1:rankX);
end

% QR decomposed X2
[Q2,T2,perm2]               = qr(X2,0);
rankY                       = sum(abs(diag(T2)) > eps(abs(T2(1)))*max(N,P2));
if rankY < P2
    Q2 = Q2(:,1:rankY); T2  = T2(1:rankY,1:rankY);
end

d                           = min(rankX,rankY);

% Compute eigenvector
[L,D,M]                     = svd(Q1' * Q2,0);
A                           = T1 \ L(:,1:d) * sqrt(N-1);
B                           = T2 \ M(:,1:d) * sqrt(N-1);

% correlation between canonical variabels
r                           = min(max(diag(D(:,1:d))', 0), 1); 

A(perm1,:)                  = [A; zeros(P1-rankX,d)];
B(perm2,:)                  = [B; zeros(P2-rankY,d)];

fcn.A                       = @(X1)(X1 - mu1) * A;
fcn.B                       = @(X2)(X2 - mu2) * B;

fcn.invA                    = @(cX)cX * (A'*A)^-1 * A' + mu1;
fcn.invB                    = @(cY)cY * (B'*B)^-1 * B' + mu2;


param.A                     = A;
param.B                     = B;
param.mu1                   = mu1;
param.mu2                    = mu2;
param.r                     = r;
param.fcn                   = fcn;