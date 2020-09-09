function prm= KFTraining(Z, X, I)

if nargin < 3
    I       = ones(size(Z,1),1);
end
dim         = size(X,2);
prm.muZ     = mean(Z,1)';
prm.muX     = mean(X,1)';
Zk          = Z' - prm.muZ;
Xk          = X' - prm.muX;

[N,NumCell] = size(Z);
prm.H       = Zk*Xk'*(Xk*Xk')^-1;
prm.Q       = ((Zk - prm.H*Xk)*(Zk - prm.H*Xk)') ./ N;
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
prm.W       = ((Xk2 - prm.A*Xk1)*(Xk2-prm.A*Xk1)')./size(Xk1,2);
prm.P       = prm.W;
prm.K       = zeros(dim,NumCell);
prm.x0      = zeros(dim,1);

prm.tab     = 1;
prm.dim     = dim;
prm.decoder = 'lkf';