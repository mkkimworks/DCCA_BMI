function prm= TrainKalmanSmoother(Z, X, I)


if nargin < 3
    I       = ones(size(Z,1),1);
end
dim         = size(X,2);
Zk          = Z';
Xk          = X';

[N,NumCell] = size(Z);
prm.C       = Zk*Xk'*(Xk*Xk')^-1;
prm.R       = ((Zk - prm.C*Xk)*(Zk - prm.C*Xk)') ./ N;
uT          = unique(I);
Xk1         = [];
Xk2         = [];
for u = 1 : length(uT)
    idx     = uT(u) == I;
    Xk_     = Xk(:,idx);
    Xk1     = [Xk1 Xk_(:,1:end-1)];
    Xk2     = [Xk2 Xk_(:,2:end)];
end
prm.A       = Xk2*Xk1'*(Xk1*Xk1')^-1;
prm.Q       = ((Xk2 - prm.A*Xk1)*(Xk2-prm.A*Xk1)')./size(Xk1,2);
prm.P0      = prm.Q;
prm.mu0     = Xk(:,1);