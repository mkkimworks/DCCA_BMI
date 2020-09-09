function [A, C, Z] = ldsLCCA(X, Kin, k, m)


[d,n]                   = size(X);
% H                       = reshape(X(:,hankel(1:m,m:n)),d*m,[]);
cca_prm                 = fcnCCA(X', Kin);

C                       = cca_prm.A;
Z                       = cca_prm.A' * X;
A                       = Z(:,2:end)/Z(:,1:end-1); % estimated transition
% [U,S,V] = svd(H,'econ');
% C = U(1:d,1:k);
% Z = S(1:k,1:k)*V(:,1:k)';
% A = Z(:,2:end)/Z(:,1:end-1); % estimated transition
% % Y = C*Z; % reconstructions

