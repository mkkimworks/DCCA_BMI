%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   
%   Title: Decoding kinematic information from primary motor cortical 
%          ensemble activity using a deep canonical correlation analysis
%   Authors: M-K Kim; J-W Sohn; S-P Kim
%   E-mail: mkkim.works@gmail.com
%   Affiliation: Ulsan National Institute of Science and Technology (UNIST)
%
%   param = fcnDCCA(X, Y, dcca_param)
%   Input
%       X: random variable of view 1
%       Y: random variable of view 2
%       NumRepeat: number of optimization repetitions
%       dcca_param: pretrained DCCA parameter
%   Output
%       param: trained DCCA parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function param              = fcnDCCA(X, Y, NumRepeat, dcca_param)

if nargin < 3
    [Net,~]                 = DCCA_Optimization(X, Y, NumRepeat, 'MonkeyF');
    dcca_param              = Net.trainedNet;
end

F1Opt                       = dcca_param{1};
F2Opt                       = dcca_param{2};

[X_beta,~]                  = deepnetbwd(X, F1Opt);
[Y_beta,~]                  = deepnetbwd(Y, F2Opt);

param.fcn.A                 = @(X)double(gather(deepnetfwd(X, F1Opt)));
param.fcn.B                 = @(Y)double(gather(deepnetfwd(Y, F2Opt)));
param.fcn.invA              = @(cX)double(gather(deepnetbwd(double(cX), F1Opt, X_beta)));
param.fcn.invB              = @(cY)double(gather(deepnetbwd(double(cY), F2Opt, Y_beta)));
