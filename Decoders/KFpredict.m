%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   
%   Title: Decoding kinematic information from primary motor cortical 
%          ensemble activity using a deep canonical correlation analysis
%   Authors: M-K Kim; J-W Sohn; S-P Kim
%   E-mail: mkkim.works@gmail.com
%   Affiliation: Ulsan National Institute of Science and Technology (UNIST)
%
%   [hat,param] = KFpredict(Z, param)
%   Input
%       Z: random variable of view 1
%       param: trained Kalman filter parameter
%   Output
%       hat: predicted states
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [hat,param]    = KFpredict(Z, param)
zk                      = Z' - param.muZ;

% prediction
param.x0                = param.A * param.x0;
param.P                 = param.A*param.P*param.A' + param.W;

% update
param.K                 = param.P*param.H'*(param.H*param.P*param.H'+param.Q)^-1;
param.x0                = param.x0+param.K*(zk-param.H*param.x0);
param.P                 = (eye(size(param.A,1))-param.K*param.H)*param.P;
param.counts            = param.counts + 1;

hat                     = (param.x0 + param.muX)';
