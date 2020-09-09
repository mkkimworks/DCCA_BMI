%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Title: Decoding kinematic information from primary motor cortical
%          ensemble activity using a deep canonical correlation analysis
%   Authors: M-K Kim; J-W Sohn; S-P Kim
%   E-mail: mkkim.works@gmail.com
%   Affiliation: Ulsan National Institute of Science and Technology (UNIST)
%   Copyright 2020. Authors All Rights Reserved.
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


clear; clc; close all;

addpath(genpath('./Functions/'));

dataset(1)                      = load('./Dataset/MonkeyC_Dataset.mat');      % Dataset
dataset(2)                      = load('./Dataset/MonkeyM_Dataset.mat');      % Dataset
data_name                       = {'MonkeyC_Data','MonkeyM_Data'};

% Optimized hyperparameters of Deep CCA for monkey C
net_param(1).NV1                = 1024;
net_param(1).NV2                = 1024;
net_param(1).NL1                = 2;
net_param(1).NL2                = 2;
net_param(1).rcov1              = 0.0427;
net_param(1).rcov2              = 0.0280;
net_param(1).lambda             = 0.0112;
net_param(1).cca_batch_sz       = 256;
net_param(1).rec_batch_sz       = 64;
net_param(1).learning_rate      = 1e-02;
net_param(1).l2penalty          = 6.84e-04;

% Optimized hyperparameters of Deep CCA for monkey M
net_param(2).NV1                = 512;
net_param(2).NV2                = 256;
net_param(2).NL1                = 1;
net_param(2).NL2                = 2;
net_param(2).rcov1              = 0.0026;
net_param(2).rcov2              = 0.0880;
net_param(2).lambda             = 0.0072;
net_param(2).cca_batch_sz       = 64;
net_param(2).rec_batch_sz       = 256;
net_param(2).learning_rate      = 0.01;
net_param(2).l2penalty          = 2.5414e-04;

load('./Parameters/num_comps.mat');  % determine_num_factors; % the number of latents = 15 determined by decoding cross-validation (both datasets)
for n = 1 : 2
    load(sprintf('./CV_Data/NR_%s_Dataset.mat', data_name{n}));
    
    % (1) PCA
    TrZ                         = gaussian_smoothing(Dat.TrZ, bin_length, PC_SD(n));
    PCA_Z                       = pca_prm.fcn(TrZ);
    
    
    % Factor Analysis
    TrZ                         = gaussian_smoothing(Dat.TrZ, bin_length, FA_SD(n));
    FA_Z                        = fa_prm.fcn(TrZ);
    
    % Linear CCA
    TrZ                         = gaussian_smoothing(Dat.TrZ, bin_length, LC_SD(n));
    LCCA_Z                      = lcca_prm.fcn.A(TrZ);
    LCCA_X                      = lcca_prm.fcn.B(TrX);
    
    
    % Deep CCA
    TrZ                         = gaussian_smoothing(Dat.TrZ, bin_length, DC_SD(n));
    DCCA_Z                      = dcca_prm.fcn.A(TrZ);
    DCCA_X                      = dcca_prm.fcn.B(TrX);
    
    raw_prm = [];  fa_prm = []; pca_prm = []; dcca_prm = []; lcca_prm = [];  
%     fcnLSTM(TrainZ, [data.X sqrt(sum(data.X.^2,2))], data.IND, sprintf('%s_FR',data_name{n}), raw_prm); 
    fcnLSTM(PCA_Z, Dat.TrX, data.IND, sprintf('%s_PCA',data_name{n}), pca_prm);
    fcnLSTM(FA_Z,  Dat.TrX, data.IND, sprintf('%s_FA',data_name{n}), fa_prm);
    fcnLSTM(LCCA_Z, LCCA_X, data.IND, sprintf('%s_LCCA',data_name{n}), lcca_prm);
    fcnLSTM(DCCA_Z, DCCA_X, data.IND, sprintf('%s_DCCA',data_name{n}), dcca_prm);
    disp(n);
end






