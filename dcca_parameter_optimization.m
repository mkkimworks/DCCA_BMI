%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Title: Decoding kinematic information from primary motor cortical
%          ensemble activity using a deep canonical correlation analysis
%   Authors: M-K Kim; J-W Sohn; S-P Kim
%   E-mail: mkkim.works@gmail.com
%   Affiliation: Ulsan National Institute of Science and Technology (UNIST)
%   Copyright 2020. Authors All Rights Reserved.
%
%   Using a linear Kalman filter for demonstration
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


clear; clc; close all;

addpath(genpath('./Functions'))

dataset(1)                      = load('./Dataset/MonkeyC_Dataset.mat');      % Dataset
dataset(2)                      = load('./Dataset/MonkeyM_Dataset.mat');      % Dataset
data_name                       = {'MonkeyC_Data','MonkeyM_Data'};
net_param                       = [];
for n = 2 : 2
    data                        = dataset(n);
    dcca_prm                    = fcnDeepCCA(sqrt(data.Z), [data.X sqrt(sum(data.X.^2,2))], data.IND, data_name{n}, net_param, 1);
end

% % Optimized hyperparameters of Deep CCA for monkey C
% net_param(1).NV1                = 1024;
% net_param(1).NV2                = 1024;
% net_param(1).NL1                = 2;
% net_param(1).NL2                = 2;
% net_param(1).rcov1              = 0.0427;
% net_param(1).rcov2              = 0.0280;
% net_param(1).lambda             = 0.0112;
% net_param(1).cca_batch_sz       = 256;
% net_param(1).rec_batch_sz       = 64;
% net_param(1).learning_rate      = 1e-02;
% net_param(1).l2penalty          = 6.84e-04;
% 
% % Optimized hyperparameters of Deep CCA for monkey M
% net_param(2).NV1                = 512;
% net_param(2).NV2                = 256;
% net_param(2).NL1                = 1;
% net_param(2).NL2                = 2;
% net_param(2).rcov1              = 0.0026;
% net_param(2).rcov2              = 0.0880;
% net_param(2).lambda             = 0.0072;
% net_param(2).cca_batch_sz       = 64;
% net_param(2).rec_batch_sz       = 256;
% net_param(2).learning_rate      = 0.01;
% net_param(2).l2penalty          = 2.5414e-04;
