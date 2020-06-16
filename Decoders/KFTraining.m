%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   
%   Title: Decoding kinematic information from primary motor cortical 
%          ensemble activity using a deep canonical correlation analysis
%   Authors: M-K Kim; J-W Sohn; S-P Kim
%   E-mail: mkkim.works@gmail.com
%   Affiliation: Ulsan National Institute of Science and Technology (UNIST)
%
%   prm= KFTraining(Z, X)
%   Input
%       Z: observation
%       X: kinematic states
%   Output
%       prm: trained Kalman filter parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function prm= KFTraining(Z, X)

dim         = size(X,2);
prm.muZ     = mean(Z,1)';
prm.muX     = mean(X,1)';
Zk          = Z' - prm.muZ;
Xk          = X' - prm.muX;

[N,NumCell] = size(Z);
prm.H       = Zk*Xk'*(Xk*Xk')^-1;
prm.Q       = ((Zk - prm.H*Xk)*(Zk - prm.H*Xk)')./N;
prm.STD     = std((Zk - prm.H*Xk),[],2) * std((Zk - prm.H*Xk),[],2)';

Xk1         = Xk(:,1:end-1);
Xk2         = Xk(:,2:end);
prm.A       = Xk2*Xk1'*(Xk1*Xk1')^-1;
prm.W       = ((Xk2 - prm.A*Xk1)*(Xk2-prm.A*Xk1)')./(N-1);

prm.P       = prm.W;
prm.K       = zeros(dim,NumCell);
prm.x0      = zeros(dim,1);
prm.get_zhat= @(xhat) xhat*prm.H' + prm.muZ';
prm.counts  = 0;
