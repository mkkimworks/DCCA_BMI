function [W1,b1,W2,b2,P1,P2,m1,m2,D,randseed]=randKCCA( ...
  X1,X2,K,M1,s1,rcov1,M2,s2,rcov2,randseed)
% [W1,b1,W2,b2,P1,P2,m1,m2,D,randseed]=randKCCA( ...
%   X1,X2,K,M1,s1,rcov1,M2,s2,rcov2,randseed) trains the randomized 
%     KCCA model of the following paper
%     David Lopez-Paz, Suvrit Sra, Alex Smola, Zoubin Ghahramani and
%     Bernhard Schoelkopf. "Randomized Nonlinear Component Analysis".
%     ICML 2014.
%
% Inputs
%   X1/X2: training data for view 1/view 2, containing samples rowwise.
%   K: dimensionality of CCA projection.
%   M1/M2: the number of random samples for each view.
%   s1/s2: Gaussian kernel width s for each view for the kernel function
%     k(x,y)=exp(-0.5*|x-y|^2/s^2) .
%   rcov1/rcov2: regularization parameter for each view.
%   randseed: random seed for random Fourier features.
%
% Outputs
%   W1/W2: random weights for view 1/view 2.
%   b1/b2: random bias for view 1/view 2.
%   P1/P2: the CCA projection matrix for view 1/view 2.
%   m1/m2: feature mean for view 1/view 2.
%   D: vector of canonical correlations for each of the K dimensions.

if ~exist('randseed','var') || isempty(randseed)
  randseed=0;
end
rng(randseed);

% Generate random features.
D1=size(X1,2);  W1=randn(D1,M1)/s1;  b1=rand(1,M1)*2*pi;
D2=size(X2,2);  W2=randn(D2,M2)/s2;  b2=rand(1,M2)*2*pi;

% Compute things by blocks in case large M1/M2 is used.
N=size(X1,1);   T=6000;  NUMBATCHES=ceil(N/T);

% Estimate mean.
m1=zeros(1,M1); m2=zeros(1,M2);
for j=1:NUMBATCHES
  batchidx=T*(j-1)+1:min(N,T*j);
  FEA1=cos(bsxfun(@plus,X1(batchidx,:)*W1,b1));
  FEA2=cos(bsxfun(@plus,X2(batchidx,:)*W2,b2));
  m1=m1+sum(FEA1,1);
  m2=m2+sum(FEA2,1);
end
m1=m1/N; m2=m2/N;

% Compute covariance.
S11=zeros(M1,M1); S22=zeros(M2,M2); S12=zeros(M1,M2);
for j=1:NUMBATCHES
  batchidx=T*(j-1)+1:min(N,T*j);
  FEA1=cos(bsxfun(@plus,X1(batchidx,:)*W1,b1));  FEA1=bsxfun(@minus,FEA1,m1);
  FEA2=cos(bsxfun(@plus,X2(batchidx,:)*W2,b2));  FEA2=bsxfun(@minus,FEA2,m2);
  S11=S11+FEA1'*FEA1;  S22=S22+FEA2'*FEA2;  S12=S12+FEA1'*FEA2;
end
S11=S11/(N-1);  S12=S12/(N-1);  S22=S22/(N-1);

% Add regularization.
S11=S11+rcov1*eye(M1); S22=S22+rcov2*eye(M2);
[V1,D1]=eig(S11); [V2,D2]=eig(S22);
% For numerical stability.
D1=diag(D1); idx1=find(D1>1e-12); D1=D1(idx1); V1=V1(:,idx1);
D2=diag(D2); idx2=find(D2>1e-12); D2=D2(idx2); V2=V2(:,idx2);
clear S11 S22;

K11=V1*diag(D1.^(-1/2))*V1'; K22=V2*diag(D2.^(-1/2))*V2'; T=K11*S12*K22;
[U,D,V]=svd(T,0);  D=diag(D);
P1=K11*U(:,1:K);  P2=K22*V(:,1:K);  D=D(1:K);
