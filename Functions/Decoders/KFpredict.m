function [Dhat,KFparam] = KFpredict(TeZ, KFparam)


zk              = TeZ' - KFparam.muZ;


% prediction
KFparam.x0      = KFparam.A * KFparam.x0;
KFparam.P       = KFparam.A*KFparam.P*KFparam.A' + KFparam.W;

% update
KFparam.K       = KFparam.P*KFparam.H'*(KFparam.H*KFparam.P*KFparam.H'+KFparam.Q)^-1;
KFparam.x0      = KFparam.x0+KFparam.K*(zk-KFparam.H*KFparam.x0);
KFparam.P       = (eye(size(KFparam.A,1))-KFparam.K*KFparam.H)*KFparam.P;
KFparam.x0      = KFparam.x0 + KFparam.muX;
if isnan(sum(KFparam.x0)); KFparam.x0 = zeros(size(KFparam.x0)); end
Dhat            = KFparam.x0';
