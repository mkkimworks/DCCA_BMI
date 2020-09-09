function [A, C, Z] = ldsDCCA(X, k, m)

keyboard

[d,n]                   = size(X);
H                       = reshape(X(:,hankel(1:m,m:n)),d*m,[]);
fa_prm                  = fcnFA(H, k);
C                       = fa_prm.beta(:,1:d)';
Z                       = C' * X;
A                       = Z(:,2:end)/Z(:,1:end-1); % estimated transition
% [U,S,V] = svd(H,'econ');
% C = U(1:d,1:k);
% Z = S(1:k,1:k)*V(:,1:k)';
% A = Z(:,2:end)/Z(:,1:end-1); % estimated transition
% % Y = C*Z; % reconstructions

