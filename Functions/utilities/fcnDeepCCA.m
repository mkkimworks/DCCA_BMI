%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   
%   Title: Decoding kinematic information from primary motor cortical 
%          ensemble activity using a deep canonical correlation analysis
%   Authors: M-K Kim; J-W Sohn; S-P Kim
%   E-mail: mkkim.works@gmail.com
%   Affiliation: Ulsan National Institute of Science and Technology (UNIST)
%
%   param = fcnDeepCCA(X1, X2, TrialIND, SubID, net_param, CV_k)
%   Input
%       X1: random variable of view 1
%       X2: random variable of view 2
%       TrialIND: reaching trial time index
%       SubID: subject ID
%       net_param: deep CCA network hyperparameters
%       CV_k: cross-validation sequential index 
%   Output
%       param: trained DCCA parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function param              = fcnDeepCCA(X1, X2, TrialIND, SubID, net_param, CV_k)

if ~isstruct(net_param)
    NumRepeat               = 200;
    [Net,~]                 = DCCA_Optimization(X1, X2, TrialIND, NumRepeat, SubID);
    net_param               = Net.net_param;
    FcnView_A               = Net.DCCAE_results.NetView_A;
    FcnView_B               = Net.DCCAE_results.NetView_B;
    if exist(sprintf('./DCCAE_results/%s',SubID), 'file') == 0
        mkdir(sprintf('./DCCAE_results/%s',SubID));
    end
    save(sprintf("./DCCAE_results/%s/",SubID) + "OptimalDCCA_Param.mat",'FcnView_A','FcnView_B','net_param');
elseif isstruct(net_param)
    Net                     = fcnDCCAE(X1, X2, TrialIND, net_param);
    FcnView_A               = Net.NetView_A;
    FcnView_B               = Net.NetView_B;
    TrainingData_Index      = unique(TrialIND);
    if exist(sprintf('./DCCAE_results/%s',SubID), 'file') == 0
        mkdir(sprintf('./DCCAE_results/%s',SubID));
    end
    save(sprintf("./DCCAE_results/%s/CV_%03d_",SubID, CV_k) + "OptimalDCCA_Param.mat",'FcnView_A','FcnView_B','TrainingData_Index');
else
    if exist(sprintf("./DCCAE_results/%s/",SubID) + "OptimalDCCA_Param.mat",'file') ~= 0
        dcca_param        	= load(sprintf("./DCCAE_results/%s/",SubID) + "OptimalDCCA_Param.mat");
        FcnView_A           = dcca_param.FcnView_A;
        FcnView_B           = dcca_param.FcnView_B;
    else
        error('Require DCCA model parameters.');
    end
end

[X_beta,~]                  = deepnetbwd(X1, FcnView_A);
[Y_beta,~]                  = deepnetbwd(X2, FcnView_B);

param.fcn.FcnView_A         = FcnView_A;
param.fcn.FcnView_B         = FcnView_B;
param.fcn.A                 = @(X1)double(gather(deepnetfwd(X1, FcnView_A)));
param.fcn.B                 = @(X2)double(gather(deepnetfwd(X2, FcnView_B)));
param.fcn.invA              = @(cX)double(gather(deepnetbwd(double(cX), FcnView_A, X_beta)));
param.fcn.invB              = @(cY)double(gather(deepnetbwd(double(cY), FcnView_B, Y_beta)));




    