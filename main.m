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
addpath(genpath('./Dataset/'));


dataset(1)                      = load('MonkeyC_Dataset.mat');    % Dataset for Monkey C
dataset(2)                      = load('MonkeyM_Dataset.mat');    % Dataset for Monkey M


manifold_colors                 = 'kbcmgr';
manifold_types                  = {'Z_E_-_F_R','Z_P_C_A','Z_F_A','Z_L_D_S','Z_L_C_V','Z_D_C_V'};
decoder_types1                  = {'Z_E_-_F_R','PCA_L_K_F','NDF','LCCA_L_K_F','DCCA_L_K_F','smooth LCCA_L_K_F','smooth DCCA_L_K_F'};
decoder_types2                  = {'PC_L_S_T_M','LCCA_L_S_T_M','DCCA_L_S_T_M','smooth LCCA_L_S_T_M','smooth DCCA_L_S_T_M'};
data_name                       = {'MonkeyC_Data','MonkeyM_Data'};

training_set_ratio              = 0.75;
bin_length                      = 0.05;
colors                          = [1 0.5 0 ; 0 0.5 1];
NumTargets                      = [8, length(dataset(2).Target)];

%% Acquire neural representations
% Hyperparameters determined by Bayesian optimization algorithm (dcca_parameter_optimization.m)
load('./Parameters/deep_CCA_hyperparameters.mat');
%% Find the number of factors or principal components with the best performance
load('./Parameters/num_comps.mat');  % determine_num_factors; % the number of latents = 15 determined by decoding cross-validation (both datasets)
load('./Parameters/KernSD.mat'); % set SD for smoothing

GW                              = gausswin(6);
GW                              = GW./sum(GW);
smoother                        = @(X)filtfilt(GW,1,X);
%%
for n = 1 : 2
    data                        = dataset(n);
    Dat                         = DataSegmentation(data, training_set_ratio,n);
    tic
    % (1) PCA
    q                           = num_comps(n).pc;
    TrZ                         = gaussian_smoothing(Dat.TrZ, bin_length, PC_SD(n));
    pca_prm                     = fcnPCA(TrZ, q);
    
    % (2) Factor Analysis
    q                           = num_comps(n).fa;
    TrZ                         = gaussian_smoothing(Dat.TrZ, bin_length, FA_SD(n));
    fa_prm                      = fcnFA(TrZ, q);
    
    % (3) LDS
    q                           = num_comps(n).ld;
    [lds_prm, llh]              = EM_LDS(Dat.TrZ', q);
    
    % (4) Linear CCA
    TrZ                         = gaussian_smoothing(Dat.TrZ, bin_length, LC_SD(n));
    lcca_prm                    = fcnCCA(TrZ, Dat.TrX);
    
    % (5) Deep CCA
    TrZ                         = gaussian_smoothing(Dat.TrZ, bin_length, DC_SD(n));
    dcca_prm                    = fcnDeepCCA(TrZ, Dat.TrX, Dat.TrI, data_name{n}, net_param(n), 0);
    
    if exist(sprintf('./CV_Data/'),'file') == 0
        mkdir(sprintf('./CV_Data/'));
    end
    save(sprintf('./CV_Data/NR_%s_Dataset.mat', data_name{n}), 'Dat', 'pca_prm', 'fa_prm','lcca_prm', 'dcca_prm','lds_prm','lstm_model')
    toc
end

%%
for n = 1 : 2
    data                        = dataset(n);
    load(sprintf('./CV_Data/NR_%s_Dataset.mat', data_name{n}));
    smooth_fcnPCA               = @(Z, n)smoother(pca_prm.fcn(gaussian_smoothing(Z, bin_length, PC_SD(n))));
    smooth_fcnFA                = @(Z, n)smoother(fa_prm.fcn(gaussian_smoothing(Z, bin_length, FA_SD(n))));
    smooth_fcnLCCA              = @(Z, n)smoother(lcca_prm.fcn.A(gaussian_smoothing(Z, bin_length, LC_SD(n))));
    smooth_fcnDCCA              = @(Z, n)smoother(dcca_prm.fcn.A(gaussian_smoothing(Z, bin_length, DC_SD(n))));
    
    % Linear CCA
    TrLCCA_Z                = smooth_fcnLCCA(Dat.TrZ, n); %lcca_prm.fcn.A(gaussian_smoothing(Dat.TrZ, bin_length, LC_SD(n)));
    TeLCCA_Z                = smooth_fcnLCCA(Dat.TeZ, n); %lcca_prm.fcn.A(gaussian_smoothing(Dat.TeZ, bin_length, LC_SD(n)));
    TrLCCA_X                = lcca_prm.fcn.B(Dat.TrX);
    TeLCCA_X                = lcca_prm.fcn.B(Dat.TeX);
    
    % Deep CCA
    TrDCCA_Z                = smooth_fcnDCCA(Dat.TrZ, n); %dcca_prm.fcn.A(gaussian_smoothing(Dat.TrZ, bin_length, DC_SD(n)));
    TeDCCA_Z                = smooth_fcnDCCA(Dat.TeZ, n); %dcca_prm.fcn.A(gaussian_smoothing(Dat.TeZ, bin_length, DC_SD(n)));
    TrDCCA_X                = dcca_prm.fcn.B(Dat.TrX);
    TeDCCA_X                = dcca_prm.fcn.B(Dat.TeX);
       

    %%
    lkf_trz                 = KFTraining(Dat.TrZ, Dat.TrX);
    lkf_lcca                = KFTraining(TrLCCA_Z, Dat.TrX);
    lkf_dcca                = KFTraining(TrDCCA_Z, Dat.TrX);
    
    beta_dcca = []; beta_lcca = []; 
    for k = 1 : 3
        beta_lcca(:,k)      = regress(TrLCCA_X(:,k), [ones(size(TrLCCA_Z,1),1) TrLCCA_Z]);
        beta_dcca(:,k)      = regress(TrDCCA_X(:,k), [ones(size(TrDCCA_Z,1),1) TrDCCA_Z]);
    end
    
    lcca_hat                = [ones(size(TeLCCA_Z,1),1) TeLCCA_Z] * beta_lcca;
    dcca_hat                = [ones(size(TeDCCA_Z,1),1) TeDCCA_Z] * beta_dcca;
    
    lcca_hat                = lcca_prm.fcn.invB(lcca_hat); lcca_hat = lcca_hat(:,1:2);
    dcca_hat                = dcca_prm.fcn.invB(dcca_hat); dcca_hat = dcca_hat(:,1:2);
    
    lcca_lkf_out            = iPredict(TeLCCA_Z, @KFpredict, lkf_lcca, Dat.TeI);
    dcca_lkf_out            = iPredict(TeDCCA_Z, @KFpredict, lkf_dcca, Dat.TeI);
    trz_lkf_out             = iPredict(Dat.TeZ, @KFpredict,  lkf_trz, Dat.TeI);
    
    TU                      = unique(Dat.TeI);
    err_lcca                = [];
    err_dcca                = [];
    err_lcca_lkf            = [];
    err_dcca_lkf            = [];
    err_trz_lkf             = [];
    
    crr_lcca                = [];
    crr_dcca                = [];
    crr_lcca_lkf            = [];
    crr_dcca_lkf            = [];
    for i = 1 : length(TU)
        idx                 = TU(i) == Dat.TeI;
        err_lcca(i)         = mae(lcca_hat(idx,1:2), Dat.TeX(idx,1:2));
        err_dcca(i)         = mae(dcca_hat(idx,1:2), Dat.TeX(idx,1:2));
        err_lcca_lkf(i)     = mae(lcca_lkf_out(idx,1:2), Dat.TeX(idx,1:2));
        err_dcca_lkf(i)     = mae(dcca_lkf_out(idx,1:2), Dat.TeX(idx,1:2));
        err_trz_lkf(i)      = mae(trz_lkf_out(idx,1:2), Dat.TeX(idx,1:2));
        
        crr_lcca(i,:)         = diag(corr(lcca_hat(idx,1:2), Dat.TeX(idx,1:2)));
        crr_dcca(i,:)         = diag(corr(dcca_hat(idx,1:2), Dat.TeX(idx,1:2)));
        crr_lcca_lkf(i,:)     = diag(corr(lcca_lkf_out(idx,1:2), Dat.TeX(idx,1:2)));
        crr_dcca_lkf(i,:)     = diag(corr(dcca_lkf_out(idx,1:2), Dat.TeX(idx,1:2)));
    end
    n_lcca_err(n) = mean(err_lcca);
    n_lcca_lkf(n) = mean(err_lcca_lkf);
    n_dcca_err(n) = mean(err_dcca);
    n_dcca_lkf(n) = mean(err_dcca_lkf);
    n_trz_lkf(n)  = mean(err_trz_lkf);
    
    ccn_lcca_err(n,:) = mean(crr_lcca);
    ccn_lcca_lkf(n,:) = mean(crr_lcca_lkf);
    ccn_dcca_err(n,:) = mean(crr_dcca);
    ccn_dcca_lkf(n,:) = mean(crr_dcca_lkf);
end
mean((n_lcca_err - n_lcca_lkf) ./ (n_lcca_err + n_lcca_lkf)) * 100
mean((n_dcca_err - n_dcca_lkf) ./ (n_dcca_err + n_dcca_lkf)) * 100
mean((n_lcca_err - n_trz_lkf) ./ (n_lcca_err + n_trz_lkf)) * 100
mean((n_dcca_err - n_trz_lkf) ./ (n_dcca_err + n_trz_lkf)) * 100


mean(mean((ccn_lcca_lkf - ccn_lcca_err) ./ (ccn_lcca_lkf + ccn_lcca_err)) * 100)
mean(mean((ccn_dcca_lkf - ccn_dcca_err) ./ (ccn_dcca_lkf + ccn_dcca_err)) * 100)
%% Tuning properties
% Figure 2
cc_lcca = [];
cc_dcca = [];
F1 = figure(1);clf; set(F1,'position',[793   375   495   763]);
F2 = figure(2);clf; set(F2,'position',[793   375   495   763]);
for n = 1 : 2
    load(sprintf('./CV_Data/NR_%s_Dataset.mat', data_name{n}));
    smooth_fcnPCA               = @(Z, n)smoother(pca_prm.fcn(gaussian_smoothing(Z, bin_length, PC_SD(n))));
    smooth_fcnFA                = @(Z, n)smoother(fa_prm.fcn(gaussian_smoothing(Z, bin_length, FA_SD(n))));
    smooth_fcnLCCA              = @(Z, n)smoother(lcca_prm.fcn.A(gaussian_smoothing(Z, bin_length, LC_SD(n))));
    smooth_fcnDCCA              = @(Z, n)smoother(dcca_prm.fcn.A(gaussian_smoothing(Z, bin_length, DC_SD(n))));
    
    % Linear CCA
    TrLCCA_Z                = smooth_fcnLCCA(Dat.TrZ, n); %lcca_prm.fcn.A(gaussian_smoothing(Dat.TrZ, bin_length, LC_SD(n)));
    TeLCCA_Z                = smooth_fcnLCCA(Dat.TeZ, n); %lcca_prm.fcn.A(gaussian_smoothing(Dat.TeZ, bin_length, LC_SD(n)));
    TrLCCA_X                = lcca_prm.fcn.B(Dat.TrX);
    TeLCCA_X                = lcca_prm.fcn.B(Dat.TeX);
    
    % Deep CCA
    TrDCCA_Z                = smooth_fcnDCCA(Dat.TrZ, n); %dcca_prm.fcn.A(gaussian_smoothing(Dat.TrZ, bin_length, DC_SD(n)));
    TeDCCA_Z                = smooth_fcnDCCA(Dat.TeZ, n); %dcca_prm.fcn.A(gaussian_smoothing(Dat.TeZ, bin_length, DC_SD(n)));
    TrDCCA_X                = dcca_prm.fcn.B(Dat.TrX);
    TeDCCA_X                = dcca_prm.fcn.B(Dat.TeX);
    
    TU                      = unique(Dat.TeI);
    for k = 1 : length(TU)
        idx                 = TU(k) == Dat.TeI;
        cc_lcca(k,:,n)      = [corr(TeLCCA_X(idx,1), TeLCCA_Z(idx,1)) corr(TeLCCA_X(idx,2), TeLCCA_Z(idx,2)) corr(TeLCCA_X(idx,3), TeLCCA_Z(idx,3))];
        cc_dcca(k,:,n)      = [corr(TeDCCA_X(idx,1), TeDCCA_Z(idx,1)) corr(TeDCCA_X(idx,2), TeDCCA_Z(idx,2)) corr(TeDCCA_X(idx,3), TeDCCA_Z(idx,3))];
    end
    
    TI                      = unique(Dat.TeI);
    
    for t = 1 : 3
        s1 = subplot(3,2,(t-1)*2+n,'parent',F1); plot(TeLCCA_X(:,t), TeLCCA_Z(:,t),'^','color',colors(n,:),'parent',s1); hold(s1,'on');
        [b,~,~,~,stat] = regress(TeLCCA_Z(:,t), [ones(size(TeLCCA_X,1),1) TeLCCA_X(:,t)]);
        plot(TeLCCA_X(:,t), [ones(size(TeLCCA_X,1),1) TeLCCA_X(:,t)] * b,'k','linewidth',1.5,'parent',s1);
        title(sprintf('\\rho = %.2f (p < 0.01)',sqrt(stat(1))),'parent',s1); if n == 1; axis(s1,[-1 1 -1 1]*3); else; axis(s1,[-1 1 -1 1]*3); end
        s2 = subplot(3,2,(t-1)*2+n,'parent',F2); plot(TeDCCA_X(:,t), TeDCCA_Z(:,t),'^','color',colors(n,:),'parent',s2); hold(s2,'on');
        [b,~,~,~,stat] = regress(TeDCCA_Z(:,t), [ones(size(TeDCCA_X,1),1) TeDCCA_X(:,t)]);
        plot(TeDCCA_X(:,t), [ones(size(TeDCCA_X,1),1) TeDCCA_X(:,t)] * b,'k','linewidth',1.5,'parent',s2);
        title(sprintf('\\rho = %.2f (p < 0.01)',sqrt(stat(1))),'parent',s2); if n == 1; axis(s2,[-1 1 -1 1]*3); else; axis(s2,[-1 1 -1 1]*3); end
        box(s1,'off'); box(s2,'off'); 
    end
end
save_fig(sprintf('./Figures/LCCA_Tuning'), F1);
save_fig(sprintf('./Figures/DCCA_Tuning'), F2);


% Figure 3
F0 = figure(1);clf; set(F0,'position',[751   216   599   924]);
for n = 1 : 2
    load(sprintf('./CV_Data/NR_%s_Dataset.mat', data_name{n}));
    smooth_fcnPCA               = @(Z, n)smoother(pca_prm.fcn(gaussian_smoothing(Z, bin_length, PC_SD(n))));
    smooth_fcnFA                = @(Z, n)smoother(fa_prm.fcn(gaussian_smoothing(Z, bin_length, FA_SD(n))));
    smooth_fcnLCCA              = @(Z, n)smoother(lcca_prm.fcn.A(gaussian_smoothing(Z, bin_length, LC_SD(n))));
    smooth_fcnDCCA              = @(Z, n)smoother(dcca_prm.fcn.A(gaussian_smoothing(Z, bin_length, DC_SD(n))));
    
    time                        = 0 : bin_length : size(Dat.TeX,1)*bin_length - bin_length;
    if n == 1
        kinWeight = 100;
    else
        kinWeight = 1;
    end
    r2_stack                    = [];
    tr_err_stack                = [];

        
        
    r2_stack_                   = []; tr_err = [];
    trial_marker                = find([diff(Dat.TeI);0] == 1);
    
    % FR
    TrZ                         = Dat.TrZ;
    TeZ                         = Dat.TeZ;
    [ind, beta, r2, p]          = fit_sorter(TrZ, Dat.TrX);
    TrZ                         = TrZ(:,ind);
    TeZ                         = TeZ(:,ind);
    r2_stack_                   = [r2_stack_ {r2}];
    Hat_Z                       = [ones(size(Dat.TeX,1),1) Dat.TeX] * beta;
    beta                        = Dat.TrX' * [ones(size(TrZ,1), 1) TrZ]*([ones(size(TrZ,1), 1) TrZ]'* [ones(size(TrZ,1), 1) TrZ])^-1;
    tr_err                      = [tr_err mae([ones(size(TrZ,1), 1) TrZ] * beta', Dat.TrX) * kinWeight];
    
    
    s1 = subplot(6,2,(1-1)*2+n,'parent',F0);
    h1 = plot(time, TeZ(:,1),'color',ones(1,3)*0.6,'linewidth',1.5,'parent',s1); hold(s1,'on');
    h2 = plot(time, Hat_Z(:,1),'r','linewidth',1.0,'parent',s1);
    plot([time(trial_marker);time(trial_marker)], [0 10],'--k','parent',s1);
    axis(s1, [0 10 0 4]); box(s1,'off');
    title(sprintf('r^2 = %.2f (p < 0.01)', r2(1)),'parent',s1)
    if n == 1
        lg = legend([h1 h2],'Actual','Estimated');
    end
    set(s1,'fontsize',11);ylabel(sprintf('Z_E_-_F_R'))
    
        
    % PCA
    TrPCA_Z                     = smooth_fcnPCA(Dat.TrZ, n); % pca_prm.fcn(Dat.TrZ);
    TePCA_Z                     = smooth_fcnPCA(Dat.TeZ, n); % pca_prm.fcn(Dat.TeZ);
    [ind, beta, r2, p]          = fit_sorter(TrPCA_Z, Dat.TrX);
    r2_stack_                   = [r2_stack_ {r2}];
    TrPCA_Z                     = TrPCA_Z(:,ind);
    TePCA_Z                     = TePCA_Z(:,ind);
    HatPC_Z                     = [ones(size(Dat.TeX,1),1) Dat.TeX] * beta;
    beta                        = Dat.TrX' * [ones(size(TrPCA_Z,1), 1) TrPCA_Z]*([ones(size(TrPCA_Z,1), 1) TrPCA_Z]'* [ones(size(TrPCA_Z,1), 1) TrPCA_Z])^-1;
    tr_err                      = [tr_err mae([ones(size(TrPCA_Z,1), 1) TrPCA_Z] * beta', Dat.TrX) * kinWeight];
    
    s1 = subplot(6,2,(2-1)*2+n,'parent',F0);
    plot(time, TePCA_Z(:,1),'color',ones(1,3)*0.6,'linewidth',1.5,'parent',s1); hold(s1,'on');
    plot(time, HatPC_Z(:,1),'r','linewidth',1.0,'parent',s1);
    plot([time(trial_marker);time(trial_marker)], [-25 60],'--k','parent',s1);
    axis(s1, [0 10 -3.5 3.5]); box(s1,'off');
    title(sprintf('r^2 = %.2f (p < 0.01)', r2(1)),'parent',s1)
    set(s1,'fontsize',11);if n == 1; ylabel(sprintf('%s', manifold_types{2})); end
    
        
    % FA
    TrFA_Z                      = smooth_fcnFA(Dat.TrZ, n); % fa_prm.fcn(Dat.TrZ);
    TeFA_Z                      = smooth_fcnFA(Dat.TeZ, n); % fa_prm.fcn(Dat.TeZ);
    [ind, beta, r2, p]          = fit_sorter(TrFA_Z, Dat.TrX);
    r2_stack_                   = [r2_stack_ {r2}];
    TrFA_Z                      = TrFA_Z(:,ind);
    TeFA_Z                      = TeFA_Z(:,ind);
    HatFA_Z                     = [ones(size(Dat.TeX,1),1) Dat.TeX] * beta;
    beta                        = Dat.TrX' * [ones(size(TrFA_Z,1), 1) TrFA_Z]*([ones(size(TrFA_Z,1), 1) TrFA_Z]'* [ones(size(TrFA_Z,1), 1) TrFA_Z])^-1;
    tr_err                      = [tr_err mae([ones(size(TrFA_Z,1), 1) TrFA_Z] * beta', Dat.TrX) * kinWeight];
        
    s1 = subplot(6,2,(3-1)*2+n,'parent',F0);
    plot(time, TeFA_Z(:,1),'color',ones(1,3)*0.6,'linewidth',1.5,'parent',s1); hold(s1,'on');
    plot(time, HatFA_Z(:,1),'r','linewidth',1.0,'parent',s1);
    plot([time(trial_marker);time(trial_marker)], [-25 60],'--k','parent',s1);
    axis(s1, [0 10 -3.5 3.5]); box(s1,'off');
    title(sprintf('r^2 = %.2f (p < 0.01)', r2(1)),'parent',s1)
    set(s1,'fontsize',11);if n == 1; ylabel(sprintf('%s', manifold_types{3})); end
    
        
              
    % LDS
    TrLDS_Z                     = kalmanSmoother(lds_prm, Dat.TrZ')';
    TeLDS_Z                     = kalmanSmoother(lds_prm, Dat.TeZ')';
    [ind, beta, r2, p]          = fit_sorter(TrLDS_Z, Dat.TrX);
    r2_stack_                   = [r2_stack_ {r2}];
    TrLDS_Z                     = TrLDS_Z(:,ind);
    TeLDS_Z                     = TeLDS_Z(:,ind);
    HatLD_Z                     = [ones(size(Dat.TeX,1),1) Dat.TeX] * beta;
    beta                        = Dat.TrX' * [ones(size(TrLDS_Z,1), 1) TrLDS_Z]*([ones(size(TrLDS_Z,1), 1) TrLDS_Z]'* [ones(size(TrLDS_Z,1), 1) TrLDS_Z])^-1;
    tr_err                      = [tr_err mae([ones(size(TrLDS_Z,1), 1) TrLDS_Z] * beta', Dat.TrX) * kinWeight];
    
    s1 = subplot(6,2,(4-1)*2+n,'parent',F0);
    plot(time, TeLDS_Z(:,1),'color',ones(1,3)*0.6,'linewidth',1.5,'parent',s1); hold(s1,'on');
    plot(time, HatLD_Z(:,1),'r','linewidth',1.0,'parent',s1);
    plot([time(trial_marker);time(trial_marker)], [-20 20],'--k','parent',s1);
    axis(s1, [0 10 -3.5 3.5]); box(s1,'off');
    title(sprintf('r^2 = %.2f (p < 0.01)', r2(1)),'parent',s1)
    set(s1,'fontsize',11); if n == 1; ylabel(sprintf('%s', manifold_types{4})); end

    
    
    % Linear CCA
    TrLCCA_Z                    = smooth_fcnLCCA(Dat.TrZ, n); % lcca_prm.fcn.A(Dat.TrZ);
    TeLCCA_Z                    = smooth_fcnLCCA(Dat.TeZ, n); % lcca_prm.fcn.A(Dat.TeZ);
    TrLCCA_X                    = lcca_prm.fcn.B(Dat.TrX);
    TeLCCA_X                    = lcca_prm.fcn.B(Dat.TeX);
    [ind, beta, r2, p]          = fit_sorter(TrLCCA_Z, Dat.TrX);
    r2_stack_                   = [r2_stack_ {r2}];
    HatLC_Z                     = [ones(size(Dat.TeX,1),1) Dat.TeX] * beta;
    beta                        = TrLCCA_X' * [ones(size(TrLCCA_Z,1), 1) TrLCCA_Z]*([ones(size(TrLCCA_Z,1), 1) TrLCCA_Z]'* [ones(size(TrLCCA_Z,1), 1) TrLCCA_Z])^-1;
    tr_err                      = [tr_err mae(lcca_prm.fcn.invB([ones(size(TrLCCA_Z,1), 1) TrLCCA_Z] * beta'), Dat.TrX) * kinWeight];
    
    s1 = subplot(6,2,(5-1)*2+n,'parent',F0);
    plot(time, TeLCCA_Z(:,1),'color',ones(1,3)*0.6,'linewidth',1.5,'parent',s1); hold(s1,'on');
    plot(time, HatLC_Z(:,1),'r','linewidth',1.0,'parent',s1);
    plot([time(trial_marker);time(trial_marker)], [-5 5],'--k','parent',s1);
    axis(s1, [0 10 -5 5]); box(s1,'off');
    title(sprintf('r^2 = %.2f (p < 0.01)', r2(1)),'parent',s1)
    set(s1,'fontsize',11); if n == 1; ylabel(sprintf('%s', manifold_types{5})); end
    
    
        
    % Deep CCA
    TrDCCA_Z                    = smooth_fcnDCCA(Dat.TrZ, n); % dcca_prm.fcn.A(Dat.TrZ);
    TeDCCA_Z                    = smooth_fcnDCCA(Dat.TeZ, n); % dcca_prm.fcn.A(Dat.TeZ);
    TrDCCA_X                    = dcca_prm.fcn.B(Dat.TrX);
    TeDCCA_X                    = dcca_prm.fcn.B(Dat.TeX);
    [ind, beta, r2, p]          = fit_sorter(TrDCCA_Z, Dat.TrX);
    r2_stack_                   = [r2_stack_ {r2}];
    HatDC_Z                     = [ones(size(Dat.TeX,1),1) Dat.TeX] * beta;
    beta                        = TrDCCA_X' * [ones(size(TrDCCA_Z,1), 1) TrDCCA_Z]*([ones(size(TrDCCA_Z,1), 1) TrDCCA_Z]'* [ones(size(TrDCCA_Z,1), 1) TrDCCA_Z])^-1;
    rest                        = dcca_prm.fcn.invB([ones(size(TrDCCA_Z,1), 1) TrDCCA_Z] * beta');
    tr_err                      = [tr_err mae(rest(:,1:3), Dat.TrX) * kinWeight];
    
    s1 = subplot(6,2,(6-1)*2+n,'parent',F0);
    plot(time, TeDCCA_Z(:,1),'color',ones(1,3)*0.6,'linewidth',1.5,'parent',s1); hold(s1,'on');
    plot(time, HatDC_Z(:,1),'r','linewidth',1.0,'parent',s1);
    plot([time(trial_marker);time(trial_marker)], [-5 5],'--k','parent',s1);
    axis(s1, [0 10 -5 5]); box(s1,'off');
    title(sprintf('r^2 = %.2f (p < 0.01)', r2(1)),'parent',s1);
    set(s1,'fontsize',11);if n == 1; ylabel(sprintf('%s', manifold_types{6})); end

    mu= []; er = [];
    for k = 1 : length(r2_stack_)
        mu(k) = mean(r2_stack_{k});
        er(k) = std(r2_stack_{k});
    end
%     s1 = subplot(6,2,(1-1)*2+n,'parent',F0); title(sprintf('r^2 = %.2f %c %.2f (p < 0.01)', mu(1), char(177), er(1)),'parent',s1); 
%     s1 = subplot(6,2,(2-1)*2+n,'parent',F0); title(sprintf('r^2 = %.2f %c %.2f (p < 0.01)', mu(2), char(177), er(2)),'parent',s1); 
%     s1 = subplot(6,2,(3-1)*2+n,'parent',F0); title(sprintf('r^2 = %.2f %c %.2f (p < 0.01)', mu(3), char(177), er(3)),'parent',s1); 
%     s1 = subplot(6,2,(4-1)*2+n,'parent',F0); title(sprintf('r^2 = %.2f %c %.2f (p < 0.01)', mu(4), char(177), er(4)),'parent',s1); 
%     s1 = subplot(6,2,(5-1)*2+n,'parent',F0); title(sprintf('r^2 = %.2f %c %.2f (p < 0.01)', mu(5), char(177), er(5)),'parent',s1); 
%     s1 = subplot(6,2,(6-1)*2+n,'parent',F0); title(sprintf('r^2 = %.2f %c %.2f (p < 0.01)', mu(6), char(177), er(6)),'parent',s1); 
    
    
    F = figure; clf, set(F,'position',[737   326   466   405]); clear h;
    for k = 1 : size(r2_stack_,2)
        h(k) = plot(mean(r2_stack_{k}), tr_err(k),'ko','MarkerFaceColor',manifold_colors(k),'MarkerSize',11); hold on;
    end
    xlabel('r^2'); ylabel('Training errors (cm/s)'); box off;
    if n == 2
        L = legend(h, [manifold_types]); L.Position = [ 0.7046    0.5329    0.1910    0.3864];%$ L.Box = 'on';
    end
    set(gca,'fontsize',12); %y%lim([0.05 0.25])
    xlim([0 1])
    save_fig(sprintf('./Figures/CharacteristicNeuralRep_%d',n), F);   
    
end
save_fig(sprintf('./Figures/RegressionNeuralRep'), F0);


% Figure 4
for n = 1 : 2
    load(sprintf('./CV_Data/NR_%s_Dataset.mat', data_name{n}));
    smooth_fcnPCA               = @(Z, n)smoother(pca_prm.fcn(gaussian_smoothing(Z, bin_length, PC_SD(n))));
    smooth_fcnFA                = @(Z, n)smoother(fa_prm.fcn(gaussian_smoothing(Z, bin_length, FA_SD(n))));
    smooth_fcnLCCA              = @(Z, n)smoother(lcca_prm.fcn.A(gaussian_smoothing(Z, bin_length, LC_SD(n))));
    smooth_fcnDCCA              = @(Z, n)smoother(dcca_prm.fcn.A(gaussian_smoothing(Z, bin_length, DC_SD(n))));
    
    if n == 1
        kinWeight = 100;
    else
        kinWeight = 1;
    end
    r2_stack_                   = [];
    tr_err                      = [];
    % FR
    TrZ                         = Dat.TrZ;
    TeZ                         = Dat.TeZ;
    [ind, beta, r2, p]          = fit_sorter(TrZ, Dat.TrX);
    TrZ                         = TrZ(:,ind);
    TeZ                         = TeZ(:,ind);
    r2_stack_                   = [r2_stack_ {r2}];
    Hat_Z                       = [ones(size(Dat.TeX,1),1) Dat.TeX] * beta;
    beta                        = Dat.TrX' * [ones(size(TrZ,1), 1) TrZ]*([ones(size(TrZ,1), 1) TrZ]'* [ones(size(TrZ,1), 1) TrZ])^-1;
    tr_err                      = [tr_err mae([ones(size(TrZ,1), 1) TrZ] * beta', Dat.TrX) * kinWeight];
    
    % PCA
    TrPCA_Z                     = smooth_fcnPCA(Dat.TrZ, n); %pca_prm.fcn(Dat.TrZ);
    TePCA_Z                     = smooth_fcnPCA(Dat.TeZ, n); %pca_prm.fcn(Dat.TeZ);
    [ind, beta, r2, p]          = fit_sorter(TrPCA_Z, Dat.TrX);
    r2_stack_                   = [r2_stack_ {r2}];
    TrPCA_Z                     = TrPCA_Z(:,ind);
    TePCA_Z                     = TePCA_Z(:,ind);
    HatPC_Z                     = [ones(size(Dat.TeX,1),1) Dat.TeX] * beta;
    beta                        = Dat.TrX' * [ones(size(TrPCA_Z,1), 1) TrPCA_Z]*([ones(size(TrPCA_Z,1), 1) TrPCA_Z]'* [ones(size(TrPCA_Z,1), 1) TrPCA_Z])^-1;
    tr_err                      = [tr_err mae([ones(size(TrPCA_Z,1), 1) TrPCA_Z] * beta', Dat.TrX) * kinWeight];
    
    % FA
    TrFA_Z                      = smooth_fcnFA(Dat.TrZ, n); %fa_prm.fcn(Dat.TrZ);
    TeFA_Z                      = smooth_fcnFA(Dat.TeZ, n); %fa_prm.fcn(Dat.TeZ);
    [ind, beta, r2, p]          = fit_sorter(TrFA_Z, Dat.TrX);
    r2_stack_                   = [r2_stack_ {r2}];
    TrFA_Z                      = TrFA_Z(:,ind);
    TeFA_Z                      = TeFA_Z(:,ind);
    HatFA_Z                     = [ones(size(Dat.TeX,1),1) Dat.TeX] * beta;
    beta                        = Dat.TrX' * [ones(size(TrFA_Z,1), 1) TrFA_Z]*([ones(size(TrFA_Z,1), 1) TrFA_Z]'* [ones(size(TrFA_Z,1), 1) TrFA_Z])^-1;
    tr_err                      = [tr_err mae([ones(size(TrFA_Z,1), 1) TrFA_Z] * beta', Dat.TrX) * kinWeight];
    
    % LDS
    TrLDS_Z                     = kalmanSmoother(lds_prm, Dat.TrZ')';
    TeLDS_Z                     = kalmanSmoother(lds_prm, Dat.TeZ')';
    [ind, beta, r2, p]          = fit_sorter(TrLDS_Z, Dat.TrX);
    r2_stack_                   = [r2_stack_ {r2}];
    TrLDS_Z                     = TrLDS_Z(:,ind);
    TeLDS_Z                     = TeLDS_Z(:,ind);
    HatLD_Z                     = [ones(size(Dat.TeX,1),1) Dat.TeX] * beta;
    beta                        = Dat.TrX' * [ones(size(TrLDS_Z,1), 1) TrLDS_Z]*([ones(size(TrLDS_Z,1), 1) TrLDS_Z]'* [ones(size(TrLDS_Z,1), 1) TrLDS_Z])^-1;
    tr_err                      = [tr_err mae([ones(size(TrLDS_Z,1), 1) TrLDS_Z] * beta', Dat.TrX) * kinWeight];
    
    % Linear CCA
    TrLCCA_Z                    = smooth_fcnLCCA(Dat.TrZ, n); %lcca_prm.fcn.A(Dat.TrZ);
    TeLCCA_Z                    = smooth_fcnLCCA(Dat.TeZ, n); %lcca_prm.fcn.A(Dat.TeZ);
    TrLCCA_X                    = lcca_prm.fcn.B(Dat.TrX);
    TeLCCA_X                    = lcca_prm.fcn.B(Dat.TeX);
    [ind, beta, r2, p]          = fit_sorter(TrLCCA_Z, Dat.TrX);
    r2_stack_                   = [r2_stack_ {r2}];
    HatLC_Z                     = [ones(size(Dat.TeX,1),1) Dat.TeX] * beta;
    beta                        = TrLCCA_X' * [ones(size(TrLCCA_Z,1), 1) TrLCCA_Z]*([ones(size(TrLCCA_Z,1), 1) TrLCCA_Z]'* [ones(size(TrLCCA_Z,1), 1) TrLCCA_Z])^-1;
    tr_err                      = [tr_err mae(lcca_prm.fcn.invB([ones(size(TrLCCA_Z,1), 1) TrLCCA_Z] * beta'), Dat.TrX) * kinWeight];
    
    % Deep CCA
    TrDCCA_Z                    = smooth_fcnDCCA(Dat.TrZ, n); %dcca_prm.fcn.A(Dat.TrZ);
    TeDCCA_Z                    = smooth_fcnDCCA(Dat.TeZ, n); %dcca_prm.fcn.A(Dat.TeZ);
    TrDCCA_X                    = dcca_prm.fcn.B(Dat.TrX);
    TeDCCA_X                    = dcca_prm.fcn.B(Dat.TeX);
    [ind, beta, r2, p]          = fit_sorter(TrDCCA_Z, Dat.TrX);
    r2_stack_                   = [r2_stack_ {r2}];
    HatDC_Z                     = [ones(size(Dat.TeX,1),1) Dat.TeX] * beta;
    beta                        = TrDCCA_X' * [ones(size(TrDCCA_Z,1), 1) TrDCCA_Z]*([ones(size(TrDCCA_Z,1), 1) TrDCCA_Z]'* [ones(size(TrDCCA_Z,1), 1) TrDCCA_Z])^-1;
    rest                        = dcca_prm.fcn.invB([ones(size(TrDCCA_Z,1), 1) TrDCCA_Z] * beta');
    tr_err                      = [tr_err mae(rest(:,1:3), Dat.TrX) * kinWeight];
    
    
    r2_fr_                      = r2_stack_{1};
    r2_pca_                     = r2_stack_{2};
    r2_fa_                      = r2_stack_{3};
    r2_lds_                     = r2_stack_{4};
    r2_lcca_                    = r2_stack_{5};
    r2_dcca_                    = r2_stack_{6};
    
        
    edge_x                      = linspace(min(Dat.TrX(:,1)), max(Dat.TrX(:,1)), 25)';
    edge_y                      = linspace(min(Dat.TrX(:,2)), max(Dat.TrX(:,2)), 25)';
    edges                       = {[edge_x;edge_x(end)+diff(edge_x(end-1:end))] [edge_y;edge_y(end)+diff(edge_y(end-1:end))]};
    
    map_pca                     = tuning_map(TrPCA_Z(:,1), Dat.TrX(:,1:2), edges);
    map_fa                      = tuning_map(TrFA_Z(:,1), Dat.TrX(:,1:2), edges);
    map_lds                     = tuning_map(TrLDS_Z(:,1), Dat.TrX(:,1:2), edges);
    map_lcca                    = tuning_map(TrLCCA_Z(:,1), Dat.TrX(:,1:2), edges);
    map_dcca                    = tuning_map(TrDCCA_Z(:,1), Dat.TrX(:,1:2), edges);
    
    if n == 1
        edge_x                  = edge_x .* kinWeight;
        edge_y                  = edge_y .* kinWeight;
    end
    
    F = figure(2);clf; set(F,'position',[588   179   653   852]);
    s1 = subplot(321); nan_imagesc(edge_x, edge_y, nanmean(map_pca,3),s1);  caxis([-2 2]); title('Z_P_C_A');  ylabel('V_Y');
    s1 = subplot(322); nan_imagesc(edge_x, edge_y, nanmean(map_fa,3),s1);   caxis([-2 2]); title('Z_F_A');
    s1 = subplot(323); nan_imagesc(edge_x, edge_y, nanmean(map_lds,3),s1);   caxis([-2 2]); title('Z_L_D_S');  ylabel('V_Y');
    s1 = subplot(324); nan_imagesc(edge_x, edge_y, nanmean(map_lcca,3),s1); caxis([-2 2]); title('Z_L_C_V'); xlabel('V_X');
    s1 = subplot(325); nan_imagesc(edge_x, edge_y, nanmean(map_dcca,3),s1); caxis([-2 2]); title('Z_D_C_V'); xlabel('V_X');  ylabel('V_Y');
    save_fig(sprintf('./Figures/tuningmap_%s',data_name{n}), F);
    

    y = [r2_pca_ r2_fa_ r2_lds_ r2_lcca_ r2_dcca_]';
    GR = [ones(size(r2_pca_)) ones(size(r2_fa_))*2 ones(size(r2_lds_))*3 ones(size(r2_lcca_))*4 ones(size(r2_dcca_))*5]';
    [~,~,stat] = anovan(y,GR, 'Display','off');
    C= multcompare(stat,'CType','bonferroni', 'Display','off')
    
    mu_fr                       = mean(r2_fr_);
    mu_pca                      = mean(r2_pca_);
    mu_fa                       = mean(r2_fa_);
    mu_lds                      = mean(r2_lds_);
    mu_lcca                     = mean(r2_lcca_);
    mu_dcca                     = mean(r2_dcca_);
    [~,p] = ttest(r2_dcca_, r2_lcca_)
    
    mu                          = [[mu_fr mu_pca mu_fa mu_lds mu_lcca mu_dcca];[mu_fr mu_pca mu_fa mu_lds mu_lcca mu_dcca]];
    xx                          = [[1 : 6]-0.3;[1 : 6]+0.3];
    
    F = figure(1);clf; set(F,'position',[588   216   358   309]);
    for k = 1 : length(r2_stack_)
        plot(k + randn(size(r2_stack_{k})) * std(r2_stack_{k})./sqrt(3), r2_stack_{k}, 'ko','markerfacecolor',manifold_colors(k),'MarkerSize',5);hold on; 
    end
    plot(xx,mu,'r','linewidth',2); 
    box off;ylabel('r^2');
    set(gca,'xtick',1:6, 'xticklabel',manifold_types,'fontsize',11); ylim([0 1]); 
    xtickangle(45);
    save_fig(sprintf('./Figures/r2dist_bar_%s',data_name{n}), F);
end


%% Training LSTM 
% lstm_parameter_optimization
% load('./Parameters/lstm_param.mat');
% vel_mae_lkf = []; lstmrst = [];
% for n = 1 : 2
%     load(sprintf('./CV_Data/NR_%s_Dataset.mat', data_name{n}));
%     smooth_fcnPCA               = @(Z, n)smoother(pca_prm.fcn(gaussian_smoothing(Z, bin_length, PC_SD(n))));
%     smooth_fcnFA                = @(Z, n)smoother(fa_prm.fcn(gaussian_smoothing(Z, bin_length, FA_SD(n))));
%     smooth_fcnLCCA              = @(Z, n)smoother(lcca_prm.fcn.A(gaussian_smoothing(Z, bin_length, LC_SD(n))));
%     smooth_fcnDCCA              = @(Z, n)smoother(dcca_prm.fcn.A(gaussian_smoothing(Z, bin_length, DC_SD(n))));
% 
%     % PCA
%     TrPCA_Z                     = smooth_fcnPCA(Dat.TrZ, n); %pca_prm.fcn(Dat.TrZ);
%     TePCA_Z                     = smooth_fcnPCA(Dat.TeZ, n); %pca_prm.fcn(Dat.TeZ);
%     
%     
%     % FA
%     TrFA_Z                      = smooth_fcnFA(Dat.TrZ, n); %fa_prm.fcn(Dat.TrZ);
%     TeFA_Z                      = smooth_fcnFA(Dat.TeZ, n); %fa_prm.fcn(Dat.TeZ);
%     
%     % LDS
%     TrLDS_Z                     = kalmanSmoother(lds_prm, Dat.TrZ')';
%     TeLDS_Z                     = kalmanSmoother(lds_prm, Dat.TeZ')';
%     
%     % Linear CCA
%     TrLCCA_Z                    = smooth_fcnLCCA(Dat.TrZ, n); %lcca_prm.fcn.A(Dat.TrZ);
%     TeLCCA_Z                    = smooth_fcnLCCA(Dat.TeZ, n); %lcca_prm.fcn.A(Dat.TeZ);
%     TrLCCA_X                    = lcca_prm.fcn.B(Dat.TrX);
%     TeLCCA_X                    = lcca_prm.fcn.B(Dat.TeX);
%     
%     % Deep CCA
%     TrDCCA_Z                    = smooth_fcnDCCA(Dat.TrZ, n); %dcca_prm.fcn.A(Dat.TrZ);
%     TeDCCA_Z                    = smooth_fcnDCCA(Dat.TeZ, n); %dcca_prm.fcn.A(Dat.TeZ);
%     TrDCCA_X                    = dcca_prm.fcn.B(Dat.TrX);
%     TeDCCA_X                    = dcca_prm.fcn.B(Dat.TeX);
%     
%     lstm_model.FR               = fcnLSTM(Dat.TrZ, Dat.TrX,   Dat.TrI, data_name{n}, lstm_param(n).FR);     % done
%     lstm_model.PC               = fcnLSTM(TrPCA_Z, Dat.TrX,   Dat.TrI, data_name{n}, lstm_param(n).PC);     % done
%     lstm_model.FA               = fcnLSTM(TrFA_Z,  Dat.TrX,   Dat.TrI, data_name{n}, lstm_param(n).FA);
%     lstm_model.LDS              = fcnLSTM(TrLDS_Z, Dat.TrX,  Dat.TrI, data_name{n}, lstm_param(n).LDS);
%     lstm_model.LCCA             = fcnLSTM(TrLCCA_Z, Dat.TrX,  Dat.TrI, data_name{n}, lstm_param(n).LCCA);
%     lstm_model.DCCA             = fcnLSTM(TrDCCA_Z, Dat.TrX,  Dat.TrI, data_name{n}, lstm_param(n).DCCA);
%     
%     t_cnt_lkf                   = zeros(NumTargets(n),1);
%     [~,vel_mae_lkf,~, ~, ~, ~]  = fcn_decoded_by_lkf(Dat, TrPCA_Z, TePCA_Z, TrFA_Z, TeFA_Z, TrLDS_Z, TeLDS_Z, TrLCCA_Z, TeLCCA_Z, TrDCCA_Z, TeDCCA_Z, t_cnt_lkf);
%     t_cnt_lkf                   = zeros(NumTargets(n),1);
%     [~,vel_mae_lds,~, ~, ~, ~]  = fcn_decoded_by_lds(Dat, lds_prm, t_cnt_lkf);
%     
%     group                       = zeros(size(vel_mae_lkf(:)));
%     for k = 1 : 5
%         id = (k-1)*size(vel_mae_lkf,1)+1:k*size(vel_mae_lkf,1);
%         group(id)               = k;
%     end
%     [~,~,stat] = friedman(vel_mae_lkf); C = multcompare(stat,'CType','bonferroni','Display','off')
%     
%     out = lstm_model.FR.fcn(Dat.TeZ);
%     lstmrst(1) = mae(Dat.TeX(:,1:2), out(:,1:2));
%     out = lstm_model.PC.fcn(TePCA_Z);
%     lstmrst(2) = mae(Dat.TeX(:,1:2), out(:,1:2));
%     out = lstm_model.FA.fcn(TeFA_Z);
%     lstmrst(3) = mae(Dat.TeX(:,1:2), out(:,1:2));
%     out = lstm_model.LDS.fcn(TeFA_Z);
%     lstmrst(4) = mae(Dat.TeX(:,1:2), out(:,1:2));
%     out = lstm_model.LCCA.fcn(TeLCCA_Z); 
%     lstmrst(5) = mae(Dat.TeX(:,1:2), out(:,1:2))
%     out = lstm_model.DCCA.fcn(TeDCCA_Z); 
%     lstmrst(6) = mae(Dat.TeX(:,1:2), out(:,1:2))
%     save(sprintf('./CV_Data/NR_%s_Dataset.mat', data_name{n}), 'Dat', 'pca_prm', 'fa_prm','lcca_prm', 'dcca_prm','lds_prm', 'lstm_model');
% end

%% Decoding 
warning off
for n = 1 : 2
    t_cnt                   = zeros(NumTargets(n),1);
    t_cnt_lkf               = t_cnt;
    t_cnt_lstm              = t_cnt;
    t_cnt_lds               = t_cnt;
    
    load(sprintf('./CV_Data/NR_%s_Dataset.mat', data_name{n}));
    smooth_fcnPCA               = @(Z, n)smoother(pca_prm.fcn(gaussian_smoothing(Z, bin_length, PC_SD(n))));
    smooth_fcnFA                = @(Z, n)smoother(fa_prm.fcn(gaussian_smoothing(Z, bin_length, FA_SD(n))));
    smooth_fcnLCCA              = @(Z, n)smoother(lcca_prm.fcn.A(gaussian_smoothing(Z, bin_length, LC_SD(n))));
    smooth_fcnDCCA              = @(Z, n)smoother(dcca_prm.fcn.A(gaussian_smoothing(Z, bin_length, DC_SD(n))));

    % PCA
    TrPCA_Z                     = smooth_fcnPCA(Dat.TrZ, n); %pca_prm.fcn(Dat.TrZ);
    TePCA_Z                     = smooth_fcnPCA(Dat.TeZ, n); %pca_prm.fcn(Dat.TeZ);
    
    % FA
    TrFA_Z                      = smooth_fcnFA(Dat.TrZ, n); %fa_prm.fcn(Dat.TrZ);
    TeFA_Z                      = smooth_fcnFA(Dat.TeZ, n); %fa_prm.fcn(Dat.TeZ);
    
    % LDS
    TrLDS_Z                     = kalmanSmoother(lds_prm, Dat.TrZ')';
    TeLDS_Z                     = kalmanSmoother(lds_prm, Dat.TeZ')';
    
    % Linear CCA
    TrLCCA_Z                    = smooth_fcnLCCA(Dat.TrZ, n); %lcca_prm.fcn.A(Dat.TrZ);
    TeLCCA_Z                    = smooth_fcnLCCA(Dat.TeZ, n); %lcca_prm.fcn.A(Dat.TeZ);
    TrLCCA_X                    = lcca_prm.fcn.B(Dat.TrX);
    TeLCCA_X                    = lcca_prm.fcn.B(Dat.TeX);
    
    % Deep CCA
    TrDCCA_Z                    = smooth_fcnDCCA(Dat.TrZ, n); %dcca_prm.fcn.A(Dat.TrZ);
    TeDCCA_Z                    = smooth_fcnDCCA(Dat.TeZ, n); %dcca_prm.fcn.A(Dat.TeZ);
    TrDCCA_X                    = dcca_prm.fcn.B(Dat.TrX);
    TeDCCA_X                    = dcca_prm.fcn.B(Dat.TeX);
    
    % Decoding process (Kalman filter)
    [pos_mae_lkf,vel_mae_lkf,vel_CC_lkf, vel_traj_lkf, pos_traj_lkf, t_cnt_lkf]       = fcn_decoded_by_lkf(Dat, TrPCA_Z, TePCA_Z, TrFA_Z, TeFA_Z, TrLDS_Z, TeLDS_Z, TrLCCA_Z, TeLCCA_Z, TrDCCA_Z, TeDCCA_Z, t_cnt_lkf);
    
    % Decoding process (LSTM)
    [pos_mae_lstm,vel_mae_lstm,vel_CC_lstm, vel_traj_lstm, pos_traj_lstm, t_cnt_lstm] = fcn_decoded_by_lstm(Dat, TePCA_Z, TeFA_Z, TeLDS_Z, TeLCCA_Z, TeDCCA_Z, lstm_model, t_cnt_lstm);
    
    % Decoding process (LDS)
    [pos_mae_lds,vel_mae_lds,vel_CC_lds, vel_traj_lds, pos_traj_lds, t_cnt_lds]       = fcn_decoded_by_lds(Dat, lds_prm, t_cnt_lds);
    
    
    output_lkf.pos_mae  = pos_mae_lkf;
    output_lkf.vel_mae  = vel_mae_lkf;
    output_lkf.vel_CC   = vel_CC_lkf;
    output_lkf.vel_traj = vel_traj_lkf;
    output_lkf.pos_traj = pos_traj_lkf;
    output_lkf.t_cnt    = t_cnt_lkf;

    output_lstm.pos_mae = pos_mae_lstm;
    output_lstm.vel_mae = vel_mae_lstm;
    output_lstm.vel_CC  = vel_CC_lstm;
    output_lstm.vel_traj = vel_traj_lstm;
    output_lstm.pos_traj = pos_traj_lstm;
    output_lstm.t_cnt   = t_cnt_lstm;

    output_lds.pos_mae  = pos_mae_lds;
    output_lds.vel_mae  = vel_mae_lds;
    output_lds.vel_CC   = vel_CC_lds;
    output_lds.vel_traj = vel_traj_lds;
    output_lds.pos_traj = pos_traj_lds;
    output_lds.t_cnt    = t_cnt_lds;

    save(sprintf('DecodingResult_%s.mat',data_name{n}), 'output_lkf' ,'output_lstm', 'output_lds');
end


% Figure 5 - 6
for n = 1 : 2
    load(sprintf('DecodingResult_%s.mat',data_name{n}));
    
    if n == 1
        kinWeight               = 100;
    else
        kinWeight               = 1;
    end
    % Draw Velocity Trace
    load(sprintf('./CV_Data/NR_%s_Dataset.mat', data_name{n}));
    time                        = 0 : bin_length : size(Dat.TeX,1)*bin_length - bin_length;    
    
    vtrue                       = output_lkf.vel_traj{1} * kinWeight;
    Z_vel                       = output_lkf.vel_traj{2} * kinWeight;
    P_vel                       = output_lkf.vel_traj{3} * kinWeight;
    F_vel                       = output_lkf.vel_traj{4} * kinWeight;
    L_vel                       = output_lkf.vel_traj{6} * kinWeight;
    D_vel                       = output_lkf.vel_traj{7} * kinWeight;
    S_vel                       = output_lds.vel_traj{2} * kinWeight;
%     S_vel                       = output_lkf.vel_traj{5} * kinWeight;
    
    L_Z_vel                     = output_lstm.vel_traj{2} * kinWeight;
    L_P_vel                     = output_lstm.vel_traj{3} * kinWeight;
    L_F_vel                     = output_lstm.vel_traj{4} * kinWeight;
    L_S_vel                     = output_lstm.vel_traj{5} * kinWeight;
    L_L_vel                     = output_lstm.vel_traj{6} * kinWeight;
    L_D_vel                     = output_lstm.vel_traj{7} * kinWeight;
    
    
    mark_ind                    = [diff(Dat.TeI);0];
    F = figure(1);clf; set(F,'position',[104         494        1766         395]);
    if n == 1
        xscale                  = [time(1) 10];
        yscale                  = [-0.02 0.02]*kinWeight;
    else
        xscale                  = [0 10];
        yscale                  = [-1 1];
    end
    
    % Z
    subplot(261); plot([time(mark_ind==1);time(mark_ind==1)],yscale'*ones(1,sum(mark_ind==1)),'k:'); hold on; plot(time, vtrue(:,1),'color',ones(1,3)*0.6,'linewidth',1.0); hold on;  plot(time, Z_vel(:,1),'r','linewidth',1.0); plot(time, L_Z_vel(:,1),'b','linewidth',1.0); axis([xscale yscale]); title('Z_E_-_F_R'); set(gca,'fontsize',11); box off;
    subplot(267); plot([time(mark_ind==1);time(mark_ind==1)],yscale'*ones(1,sum(mark_ind==1)),'k:'); hold on; plot(time, vtrue(:,2),'color',ones(1,3)*0.6,'linewidth',1.0); hold on;  plot(time, Z_vel(:,2),'r','linewidth',1.0); plot(time, L_Z_vel(:,2),'b','linewidth',1.0); axis([xscale yscale]); set(gca,'fontsize',11);box off;
    
    % fPCA
    subplot(262); plot([time(mark_ind==1);time(mark_ind==1)],yscale'*ones(1,sum(mark_ind==1)),'k:'); hold on; plot(time, vtrue(:,1),'color',ones(1,3)*0.6,'linewidth',1.0); hold on;  plot(time, P_vel(:,1),'r','linewidth',1.0); plot(time, L_P_vel(:,1),'b','linewidth',1.0); axis([xscale yscale]); title('Z_P_C_A'); set(gca,'fontsize',11);box off;
    subplot(268); plot([time(mark_ind==1);time(mark_ind==1)],yscale'*ones(1,sum(mark_ind==1)),'k:'); hold on; plot(time, vtrue(:,2),'color',ones(1,3)*0.6,'linewidth',1.0); hold on;  plot(time, P_vel(:,2),'r','linewidth',1.0); plot(time, L_P_vel(:,2),'b','linewidth',1.0); axis([xscale yscale]); set(gca,'fontsize',11);box off;
    
    % fFA
    subplot(263); plot([time(mark_ind==1);time(mark_ind==1)],yscale'*ones(1,sum(mark_ind==1)),'k:'); hold on; plot(time, vtrue(:,1),'color',ones(1,3)*0.6,'linewidth',1.0); hold on;  plot(time, F_vel(:,1),'r','linewidth',1.0); plot(time, L_F_vel(:,1),'b','linewidth',1.0); axis([xscale yscale]); title('Z_F_A'); set(gca,'fontsize',11);box off;
    subplot(269); plot([time(mark_ind==1);time(mark_ind==1)],yscale'*ones(1,sum(mark_ind==1)),'k:'); hold on; plot(time, vtrue(:,2),'color',ones(1,3)*0.6,'linewidth',1.0); hold on;  plot(time, F_vel(:,2),'r','linewidth',1.0); plot(time, L_F_vel(:,2),'b','linewidth',1.0); axis([xscale yscale]); set(gca,'fontsize',11);box off;
    
    % LDS
    subplot(264); plot([time(mark_ind==1);time(mark_ind==1)],yscale'*ones(1,sum(mark_ind==1)),'k:'); hold on; plot(time, vtrue(:,1),'color',ones(1,3)*0.6,'linewidth',1.0); hold on;  plot(time, S_vel(:,1),'r','linewidth',1.0); plot(time, L_S_vel(:,1),'b','linewidth',1.0); axis([xscale yscale]); title('Z_L_D_S'); set(gca,'fontsize',11);box off;
    subplot(2,6,10); plot([time(mark_ind==1);time(mark_ind==1)],yscale'*ones(1,sum(mark_ind==1)),'k:'); hold on; plot(time, vtrue(:,2),'color',ones(1,3)*0.6,'linewidth',1.0); hold on;  plot(time, S_vel(:,2),'r','linewidth',1.0);plot(time, L_S_vel(:,2),'b','linewidth',1.0); axis([xscale yscale]); set(gca,'fontsize',11);box off;
    
    % fLCCA
    subplot(265); plot([time(mark_ind==1);time(mark_ind==1)],yscale'*ones(1,sum(mark_ind==1)),'k:'); hold on; plot(time, vtrue(:,1),'color',ones(1,3)*0.6,'linewidth',1.0); hold on;  plot(time, L_vel(:,1),'r','linewidth',1.0); plot(time, L_L_vel(:,1),'b','linewidth',1.0); axis([xscale yscale]); title('Z_L_C_V'); set(gca,'fontsize',11);box off;
    subplot(2,6,11); plot([time(mark_ind==1);time(mark_ind==1)],yscale'*ones(1,sum(mark_ind==1)),'k:'); hold on; plot(time, vtrue(:,2),'color',ones(1,3)*0.6,'linewidth',1.0); hold on;  plot(time, L_vel(:,2),'r','linewidth',1.0); plot(time, L_L_vel(:,2),'b','linewidth',1.0); axis([xscale yscale]); set(gca,'fontsize',11);box off;
    
    % fDCCA
    subplot(266);   plot([time(mark_ind==1);time(mark_ind==1)],yscale'*ones(1,sum(mark_ind==1)),'k:'); hold on; plot(time, vtrue(:,1),'color',ones(1,3)*0.6,'linewidth',1.0); hold on;  plot(time, D_vel(:,1),'r','linewidth',1.0); plot(time, L_D_vel(:,1),'b','linewidth',1.0); axis([xscale yscale]); title('Z_D_C_V'); set(gca,'fontsize',11);box off;
    subplot(2,6,12);plot([time(mark_ind==1);time(mark_ind==1)],yscale'*ones(1,sum(mark_ind==1)),'k:'); hold on; plot(time, vtrue(:,2),'color',ones(1,3)*0.6,'linewidth',1.0); hold on;  plot(time, D_vel(:,2),'r','linewidth',1.0); plot(time, L_D_vel(:,2),'b','linewidth',1.0); axis([xscale yscale]); set(gca,'fontsize',11);    box off;
       
    save_fig(sprintf('./Figures/VelocityTrajectory_%s',data_name{n}), F);
    %----------------------------------------------------------------------
    
    t_cnt                       = output_lkf.t_cnt;
    ptrue                       = [];
    Z_pos                       = [];
    P_pos                       = [];
    F_pos                       = [];
    L_pos                       = [];
    D_pos                       = [];
    S_pos                       = [];
    L_Z_pos                       = [];
    L_P_pos                       = [];
    L_F_pos                       = [];
    L_S_pos                       = [];
    L_L_pos                       = [];
    L_D_pos                       = [];
    for d = 1 : NumTargets(n)       
        try
        ind     = squeeze(sum(sum(output_lkf.pos_traj{1}(:,:,:,d),2),1)) ~= 0;
        ptrue_  = output_lkf.pos_traj{1}(:,:,ind,d);
        Z_pos_  = output_lkf.pos_traj{2}(:,:,ind,d);
        P_pos_  = output_lkf.pos_traj{3}(:,:,ind,d);
        F_pos_  = output_lkf.pos_traj{4}(:,:,ind,d);
        L_pos_  = output_lkf.pos_traj{6}(:,:,ind,d);
        D_pos_  = output_lkf.pos_traj{7}(:,:,ind,d);
        S_pos_  = output_lds.pos_traj{2}(:,:,ind,d);
        
        ptrue(:,:,1:size(ptrue_,3),d)  = ptrue_ * kinWeight;
        Z_pos(:,:,1:size(Z_pos_,3),d)  = Z_pos_ * kinWeight;
        P_pos(:,:,1:size(P_pos_,3),d)  = P_pos_ * kinWeight;
        F_pos(:,:,1:size(F_pos_,3),d)  = F_pos_ * kinWeight;
        L_pos(:,:,1:size(L_pos_,3),d)  = L_pos_ * kinWeight;
        D_pos(:,:,1:size(D_pos_,3),d)  = D_pos_ * kinWeight;
        S_pos(:,:,1:size(S_pos_,3),d)  = S_pos_ * kinWeight;
        
        Z_pos_  = output_lstm.pos_traj{2}(:,:,ind,d);
        P_pos_  = output_lstm.pos_traj{3}(:,:,ind,d);
        F_pos_  = output_lstm.pos_traj{4}(:,:,ind,d);
        S_pos_  = output_lstm.pos_traj{5}(:,:,ind,d);
        L_pos_  = output_lstm.pos_traj{6}(:,:,ind,d);
        D_pos_  = output_lstm.pos_traj{7}(:,:,ind,d);
        
        L_Z_pos(:,:,1:size(Z_pos_,3),d)  = Z_pos_ * kinWeight;
        L_P_pos(:,:,1:size(P_pos_,3),d)  = P_pos_ * kinWeight;
        L_F_pos(:,:,1:size(F_pos_,3),d)  = F_pos_ * kinWeight;
        L_S_pos(:,:,1:size(F_pos_,3),d)  = S_pos_ * kinWeight;
        L_L_pos(:,:,1:size(L_pos_,3),d)  = L_pos_ * kinWeight;
        L_D_pos(:,:,1:size(D_pos_,3),d)  = D_pos_ * kinWeight;
        catch
        end
    end
    
    TU = unique(Dat.TeI);
    
    clear std_vel*
    clear std_pos*
    for t = 1 : length(TU)
        idx = TU(t) == Dat.TeI;
        std_vel(t,:) =std(vtrue(idx,:));
        std_pos(t,:) =std(cumsum(vtrue(idx,:)));
        
        std_vel_pred_Z(t,:) = std(Z_vel(idx,:));
        std_vel_pred_P(t,:) = std(P_vel(idx,:));
        std_vel_pred_F(t,:) = std(F_vel(idx,:));
        std_vel_pred_S(t,:) = std(S_vel(idx,:));
        std_vel_pred_L(t,:) = std(L_vel(idx,:));
        std_vel_pred_D(t,:) = std(D_vel(idx,:));
        
        std_pos_pred_Z(t,:) = std(Z_vel(idx,:));
        std_pos_pred_P(t,:) = std(P_vel(idx,:));
        std_pos_pred_F(t,:) = std(F_vel(idx,:));
        std_pos_pred_S(t,:) = std(S_vel(idx,:));
        std_pos_pred_L(t,:) = std(L_vel(idx,:));
        std_pos_pred_D(t,:) = std(D_vel(idx,:));
    end
    truepos_std(n,:) = mean(std_pos)
    truevel_std(n,:) = mean(std_vel)
    
    lkf_pf_vel(n,:) = mean([mean((std_vel_pred_Z - std_vel) ./ (std_vel_pred_Z + std_vel),1) * 100
    mean((std_vel_pred_P - std_vel) ./ (std_vel_pred_P + std_vel),1) * 100
    mean((std_vel_pred_F - std_vel) ./ (std_vel_pred_F + std_vel),1) * 100
    mean((std_vel_pred_S - std_vel) ./ (std_vel_pred_S + std_vel),1) * 100
    mean((std_vel_pred_L - std_vel) ./ (std_vel_pred_L + std_vel),1) * 100
    mean((std_vel_pred_D - std_vel) ./ (std_vel_pred_D + std_vel),1) * 100],1)
    
    lkf_pf_pos(n,:) = mean([mean((std_pos_pred_Z - std_pos) ./ (std_pos_pred_Z + std_pos),1) * 100
    mean((std_pos_pred_P - std_pos) ./ (std_pos_pred_P + std_pos),1) * 100
    mean((std_pos_pred_F - std_pos) ./ (std_pos_pred_F + std_pos),1) * 100
    mean((std_pos_pred_S - std_pos) ./ (std_pos_pred_S + std_pos),1) * 100
    mean((std_pos_pred_L - std_pos) ./ (std_pos_pred_L + std_pos),1) * 100
    mean((std_pos_pred_D - std_pos) ./ (std_pos_pred_D + std_pos),1) * 100],1)

    for t = 1 : length(TU)
        idx = TU(t) == Dat.TeI;
        std_vel(t,:) =std(vtrue(idx,:));
        std_pos(t,:) =std(cumsum(vtrue(idx,:)));
        
        std_vel_pred_Z(t,:) = std(L_Z_vel(idx,:));
        std_vel_pred_P(t,:) = std(L_P_vel(idx,:));
        std_vel_pred_F(t,:) = std(L_F_vel(idx,:));
        std_vel_pred_S(t,:) = std(L_S_vel(idx,:));
        std_vel_pred_L(t,:) = std(L_L_vel(idx,:));
        std_vel_pred_D(t,:) = std(L_D_vel(idx,:));
        
        std_pos_pred_Z(t,:) = std(L_Z_vel(idx,:));
        std_pos_pred_P(t,:) = std(L_P_vel(idx,:));
        std_pos_pred_F(t,:) = std(L_F_vel(idx,:));
        std_pos_pred_S(t,:) = std(L_S_vel(idx,:));
        std_pos_pred_L(t,:) = std(L_L_vel(idx,:));
        std_pos_pred_D(t,:) = std(L_D_vel(idx,:));
    end
    
    lstm_pf_vel(n,:) = mean([mean((std_vel_pred_Z - std_vel) ./ (std_vel_pred_Z + std_vel),1) * 100
    mean((std_vel_pred_P - std_vel) ./ (std_vel_pred_P + std_vel),1) * 100
    mean((std_vel_pred_F - std_vel) ./ (std_vel_pred_F + std_vel),1) * 100
    mean((std_vel_pred_S - std_vel) ./ (std_vel_pred_S + std_vel),1) * 100
    mean((std_vel_pred_L - std_vel) ./ (std_vel_pred_L + std_vel),1) * 100
    mean((std_vel_pred_D - std_vel) ./ (std_vel_pred_D + std_vel),1) * 100],1)
    
    lstm_pf_pos(n,:) = mean([mean((std_pos_pred_Z - std_pos) ./ (std_pos_pred_Z + std_pos),1) * 100
    mean((std_pos_pred_P - std_pos) ./ (std_pos_pred_P + std_pos),1) * 100
    mean((std_pos_pred_F - std_pos) ./ (std_pos_pred_F + std_pos),1) * 100
    mean((std_pos_pred_S - std_pos) ./ (std_pos_pred_S + std_pos),1) * 100
    mean((std_pos_pred_L - std_pos) ./ (std_pos_pred_L + std_pos),1) * 100
    mean((std_pos_pred_D - std_pos) ./ (std_pos_pred_D + std_pos),1) * 100],1)
    
    
    mu_Z_pos_var = []; 
    er_Z_pos_var = []; 
    mu_P_pos_var = []; 
    er_P_pos_var = []; 
    mu_F_pos_var = []; 
    er_F_pos_var = []; 
    mu_S_pos_var = []; 
    er_S_pos_var = []; 
    mu_L_pos_var = []; 
    er_L_pos_var = []; 
    mu_D_pos_var = []; 
    er_D_pos_var = []; 
    for d = 1 : 8
        mu_Z_pos_var(:,:,d)     = mean(squeeze(std(Z_pos(:,:,1:t_cnt(d),d) - ptrue(:,:,1:t_cnt(d),d),[],1))',1);
        er_Z_pos_var(:,:,d)     = std(squeeze(std(Z_pos(:,:,1:t_cnt(d),d) - ptrue(:,:,1:t_cnt(d),d),[],1))',[],1);
        
        mu_P_pos_var(:,:,d)     = mean(squeeze(std(P_pos(:,:,1:t_cnt(d),d) - ptrue(:,:,1:t_cnt(d),d),[],1))',1);
        er_P_pos_var(:,:,d)     = std(squeeze(std(P_pos(:,:,1:t_cnt(d),d) - ptrue(:,:,1:t_cnt(d),d),[],1))',[],1);
        
        mu_F_pos_var(:,:,d)     = mean(squeeze(std(F_pos(:,:,1:t_cnt(d),d) - ptrue(:,:,1:t_cnt(d),d),[],1))',1);
        er_F_pos_var(:,:,d)     = std(squeeze(std(F_pos(:,:,1:t_cnt(d),d) - ptrue(:,:,1:t_cnt(d),d),[],1))',[],1);
        
        mu_S_pos_var(:,:,d)     = mean(squeeze(std(S_pos(:,:,1:t_cnt(d),d) - ptrue(:,:,1:t_cnt(d),d),[],1))',1);
        er_S_pos_var(:,:,d)     = std(squeeze(std(S_pos(:,:,1:t_cnt(d),d) - ptrue(:,:,1:t_cnt(d),d),[],1))',[],1);
        
        mu_L_pos_var(:,:,d)     = mean(squeeze(std(L_pos(:,:,1:t_cnt(d),d) - ptrue(:,:,1:t_cnt(d),d),[],1))',1);
        er_L_pos_var(:,:,d)     = std(squeeze(std(L_pos(:,:,1:t_cnt(d),d) - ptrue(:,:,1:t_cnt(d),d),[],1))',[],1);
        
        mu_D_pos_var(:,:,d)     = mean(squeeze(std(D_pos(:,:,1:t_cnt(d),d) - ptrue(:,:,1:t_cnt(d),d),[],1))',1);
        er_D_pos_var(:,:,d)     = std(squeeze(std(D_pos(:,:,1:t_cnt(d),d) - ptrue(:,:,1:t_cnt(d),d),[],1))',[],1);
    end
    if n == 2
        mu_Z_pos_var = mean(std(Z_pos(:,:,:,t_cnt==1) - ptrue(:,:,:,t_cnt==1)),4);
        mu_P_pos_var = mean(std(P_pos(:,:,:,t_cnt==1) - ptrue(:,:,:,t_cnt==1)),4);
        mu_F_pos_var = mean(std(F_pos(:,:,:,t_cnt==1) - ptrue(:,:,:,t_cnt==1)),4);
        mu_S_pos_var = mean(std(S_pos(:,:,:,t_cnt==1) - ptrue(:,:,:,t_cnt==1)),4);
        mu_L_pos_var = mean(std(L_pos(:,:,:,t_cnt==1) - ptrue(:,:,:,t_cnt==1)),4);
        mu_D_pos_var = mean(std(D_pos(:,:,:,t_cnt==1) - ptrue(:,:,:,t_cnt==1)),4);
    end
    
    pos_std_lkf(:,:,n) = [mean(mu_Z_pos_var,3) ; mean(mu_P_pos_var,3) ; mean(mu_F_pos_var,3) ; mean(mu_S_pos_var,3) ; mean(mu_L_pos_var,3) ; mean(mu_D_pos_var,3)]
    
    for d = 1 : 8
        mu_Z_pos_var(:,:,d)     = mean(squeeze(std(L_Z_pos(:,:,1:t_cnt(d),d) - ptrue(:,:,1:t_cnt(d),d),[],1))',1);
        er_Z_pos_var(:,:,d)     = std(squeeze(std(L_Z_pos(:,:,1:t_cnt(d),d) - ptrue(:,:,1:t_cnt(d),d),[],1))',[],1);
        
        mu_P_pos_var(:,:,d)     = mean(squeeze(std(L_P_pos(:,:,1:t_cnt(d),d) - ptrue(:,:,1:t_cnt(d),d),[],1))',1);
        er_P_pos_var(:,:,d)     = std(squeeze(std(L_P_pos(:,:,1:t_cnt(d),d) - ptrue(:,:,1:t_cnt(d),d),[],1))',[],1);
        
        mu_F_pos_var(:,:,d)     = mean(squeeze(std(L_F_pos(:,:,1:t_cnt(d),d) - ptrue(:,:,1:t_cnt(d),d),[],1))',1);
        er_F_pos_var(:,:,d)     = std(squeeze(std(L_F_pos(:,:,1:t_cnt(d),d) - ptrue(:,:,1:t_cnt(d),d),[],1))',[],1);
        
        mu_S_pos_var(:,:,d)     = mean(squeeze(std(L_S_pos(:,:,1:t_cnt(d),d) - ptrue(:,:,1:t_cnt(d),d),[],1))',1);
        er_S_pos_var(:,:,d)     = std(squeeze(std(L_S_pos(:,:,1:t_cnt(d),d) - ptrue(:,:,1:t_cnt(d),d),[],1))',[],1);
        
        mu_L_pos_var(:,:,d)     = mean(squeeze(std(L_L_pos(:,:,1:t_cnt(d),d) - ptrue(:,:,1:t_cnt(d),d),[],1))',1);
        er_L_pos_var(:,:,d)     = std(squeeze(std(L_L_pos(:,:,1:t_cnt(d),d) - ptrue(:,:,1:t_cnt(d),d),[],1))',[],1);
        
        mu_D_pos_var(:,:,d)     = mean(squeeze(std(L_D_pos(:,:,1:t_cnt(d),d) - ptrue(:,:,1:t_cnt(d),d),[],1))',1);
        er_D_pos_var(:,:,d)     = std(squeeze(std(L_D_pos(:,:,1:t_cnt(d),d) - ptrue(:,:,1:t_cnt(d),d),[],1))',[],1);
    end
    if n == 2
        mu_Z_pos_var = mean(std(L_Z_pos(:,:,:,t_cnt==1) - ptrue(:,:,:,t_cnt==1)),4);
        mu_P_pos_var = mean(std(L_P_pos(:,:,:,t_cnt==1) - ptrue(:,:,:,t_cnt==1)),4);
        mu_F_pos_var = mean(std(L_F_pos(:,:,:,t_cnt==1) - ptrue(:,:,:,t_cnt==1)),4);
        mu_S_pos_var = mean(std(L_S_pos(:,:,:,t_cnt==1) - ptrue(:,:,:,t_cnt==1)),4);
        mu_L_pos_var = mean(std(L_L_pos(:,:,:,t_cnt==1) - ptrue(:,:,:,t_cnt==1)),4);
        mu_D_pos_var = mean(std(L_D_pos(:,:,:,t_cnt==1) - ptrue(:,:,:,t_cnt==1)),4);
    end
    pos_std_lstm(:,:,n) = [mean(mu_Z_pos_var,3) ; mean(mu_P_pos_var,3) ; mean(mu_F_pos_var,3) ; mean(mu_S_pos_var,3) ; mean(mu_L_pos_var,3) ; mean(mu_D_pos_var,3)]
    
    
    
    if n == 1
        F = figure(1);clf; set(F,'position',[13         519        2192         352]);
        h1 = subplot(161); plot(0, 0, 'o','MarkerEdgeColor','k','markerfacecolor','y','markersize',15); hold(h1,'on'); fcnErrorbar2D({ptrue, t_cnt},'color',ones(1,3)*0.6,'handles',h1); fcnErrorbar2D({L_Z_pos, t_cnt},'color','b','handles',h1, 'title','Z_E_-_F_R','fontsize',12); fcnErrorbar2D({Z_pos, t_cnt},'color','r','handles',h1, 'title','Z_E_-_F_R','fontsize',12); axis([-.1 .1 -.1 .1]*100); ylabel('P_Y'); set(gca,'DataAspectRatio',[1 1 1]);
        h1 = subplot(162); plot(0, 0, 'o','MarkerEdgeColor','k','markerfacecolor','y','markersize',15); hold(h1,'on'); fcnErrorbar2D({ptrue, t_cnt},'color',ones(1,3)*0.6,'handles',h1); fcnErrorbar2D({L_P_pos, t_cnt},'color','b','handles',h1, 'title','PCA','fontsize',12); fcnErrorbar2D({P_pos, t_cnt},'color','r','handles',h1, 'title','Z_P_C_A','fontsize',12); axis([-.1 .1 -.1 .1]*100); set(gca,'DataAspectRatio',[1 1 1]);
        h1 = subplot(163); plot(0, 0, 'o','MarkerEdgeColor','k','markerfacecolor','y','markersize',15); hold(h1,'on'); fcnErrorbar2D({ptrue, t_cnt},'color',ones(1,3)*0.6,'handles',h1); fcnErrorbar2D({L_F_pos, t_cnt},'color','b','handles',h1, 'title','FA','fontsize',12); fcnErrorbar2D({F_pos, t_cnt},'color','r','handles',h1, 'title','Z_F_A','fontsize',12); axis([-.1 .1 -.1 .1]*100); ylabel('P_Y');set(gca,'DataAspectRatio',[1 1 1]);
        h1 = subplot(164); plot(0, 0, 'o','MarkerEdgeColor','k','markerfacecolor','y','markersize',15); hold(h1,'on'); fcnErrorbar2D({ptrue, t_cnt},'color',ones(1,3)*0.6,'handles',h1); fcnErrorbar2D({L_S_pos, t_cnt},'color','b','handles',h1, 'title','Z_L_D_S','fontsize',12); fcnErrorbar2D({S_pos, t_cnt},'color','r','handles',h1, 'title','Z_L_D_S','fontsize',12);axis([-.1 .1 -.1 .1]*100);set(gca,'DataAspectRatio',[1 1 1]);
        h1 = subplot(165); plot(0, 0, 'o','MarkerEdgeColor','k','markerfacecolor','y','markersize',15); hold(h1,'on'); fcnErrorbar2D({ptrue, t_cnt},'color',ones(1,3)*0.6,'handles',h1); fcnErrorbar2D({L_L_pos, t_cnt},'color','b','handles',h1, 'title','LCCA','fontsize',12); fcnErrorbar2D({L_pos, t_cnt},'color','r','handles',h1, 'title','Z_L_C_V','fontsize',12);axis([-.1 .1 -.1 .1]*100);xlabel('P_X');ylabel('P_Y');set(gca,'DataAspectRatio',[1 1 1]);
        h1 = subplot(166); plot(0, 0, 'o','MarkerEdgeColor','k','markerfacecolor','y','markersize',15); hold(h1,'on'); fcnErrorbar2D({ptrue, t_cnt},'color',ones(1,3)*0.6,'handles',h1); fcnErrorbar2D({L_D_pos, t_cnt},'color','b','handles',h1, 'title','DCCA','fontsize',12); fcnErrorbar2D({D_pos, t_cnt},'color','r','handles',h1, 'title','Z_D_C_V','fontsize',12);axis([-.1 .1 -.1 .1]*100);xlabel('P_X');set(gca,'DataAspectRatio',[1 1 1]);
        save_fig(sprintf('./Figures/PositionTrajectory_%s',data_name{n}), F);
    else
%         F = figure(1);clf; set(F,'position',[364 424 1082 217]); 
%         view_sel_trials = [4 10 49 57 69 73 121];
%         for sub_t = 1 : 5
%             sub_i = view_sel_trials(sub_t);
%             h1 = subplot(1,5,sub_t);
%             hp0 = plot(0, 0, 'o','MarkerEdgeColor','k','markerfacecolor','y','markersize',15); hold(h1,'on');
%             hp1 = plot(ptrue(:,1,1,sub_i), ptrue(:,2,1,sub_i),'color',ones(1,3)*0.6,'linewidth',1.5);
%             hp2 = plot(P_pos(:,1,1,sub_i), P_pos(:,2,1,sub_i),'b','linewidth',1.5);
%             hp3 = plot(F_pos(:,1,1,sub_i), F_pos(:,2,1,sub_i),'c','linewidth',1.5);
%             hp4 = plot(S_pos(:,1,1,sub_i), S_pos(:,2,1,sub_i),'m','linewidth',1.5);
%             hp5 = plot(L_pos(:,1,1,sub_i), L_pos(:,2,1,sub_i),'m','linewidth',1.5);
%             hp6 = plot(D_pos(:,1,1,sub_i), D_pos(:,2,1,sub_i),'r','linewidth',1.5);
%             
%             axis(h1,[-10 10 -10 10]); set(h1, 'fontsize',12);
%             if sub_t == 1
%                 L = legend([hp0, hp1, hp2, hp3, hp4, hp5, hp6],{'Start point', 'Actual','PCA','FA','NDF','LCCA','DCCA'});
%                 L.Position = [0.0175    0.3323    0.0837    0.5195]; L.Box= 'off';
%             end
%             set(h1, 'DataAspectRatio',[1 1 1]);
%             title(sprintf('Trial %d', sub_t));
%         end
    end
    
end


%% Figure 7 - 8
label = {'Z_E_-_F_R','Z_P_C_A','Z_F_A','Z_L_D_S','Z_L_C_V','Z_D_C_V'};

F = figure;clf; set(F,'position',[466          80        1017         916]);
for n = 1 : 2
    if n == 1
        kinWeight               = 100;
    else
        kinWeight               = 1;
    end
    load(sprintf('DecodingResult_%s.mat',data_name{n}));
    
    vel_corr_lkf                = output_lkf.vel_CC;
    vel_mae_lkf                 = output_lkf.vel_mae;
    pos_mae_lkf                 = output_lkf.pos_mae;
    vel_mae_lstm                = output_lstm.vel_mae;
    pos_mae_lstm                = output_lstm.pos_mae;
    
    vel_corr_lds                = squeeze(output_lds.vel_CC)';
    vel_mae_lds                 = output_lds.vel_mae(:,1);
    pos_mae_lds                 = output_lds.pos_mae(:,1);

%     vel_mae                     = vel_mae_lkf*kinWeight;
%     pos_mae                     = pos_mae_lkf*kinWeight;
%     L_vel_mae                   = vel_mae_lstm*kinWeight;
%     L_pos_mae                   = pos_mae_lstm*kinWeight;
    vel_mae                     = [vel_mae_lkf(:,1:3) vel_mae_lds vel_mae_lkf(:,[5 6])]*kinWeight;
    pos_mae                     = [pos_mae_lkf(:,1:3) pos_mae_lds pos_mae_lkf(:,[5 6])]*kinWeight;
    L_vel_mae                   = vel_mae_lstm*kinWeight;
    L_pos_mae                   = pos_mae_lstm*kinWeight;
    
    
    
    mu_vel                      = mean(vel_mae,1);
    mu_pos                      = mean(pos_mae,1);
    er_vel                      = std(vel_mae,[],1)./sqrt(size(vel_mae,1));
    er_pos                      = std(pos_mae,[],1)./sqrt(size(vel_mae,1));
    
    L_mu_vel                    = mean(L_vel_mae,1);
    L_mu_pos                    = mean(L_pos_mae,1);
    L_er_vel                    = std(L_vel_mae,[],1)./sqrt(size(L_vel_mae,1));
    L_er_pos                    = std(L_pos_mae,[],1)./sqrt(size(L_vel_mae,1));
    
  
    
    disp('Velocity LKF')
    [~,tbl,stat] = friedman(vel_mae,1,'off'); C = multcompare(stat,'CType','bonferroni','Display','off')
    disp('Velocity LSTM')
    [~,tbl,stat] = friedman(L_vel_mae,1,'off'); C = multcompare(stat,'CType','bonferroni','Display','off')
    
    s1 = subplot(2,2,(n-1)*2+1); 
    b1 = bar(1 : 3, mu_vel(1:3),'FaceColor','r','FaceAlpha',0.5,'linewidth',1.5,'parent',s1); hold(s1,'on'); errorbar(1 : 3, mu_vel(1:3), er_vel(1:3), 'k','LineStyle','none','linewidth',1.5,'parent',s1);
    b1 = bar(4, mu_vel(4),'FaceColor','r','FaceAlpha',0.5,'linewidth',1.5,'parent',s1); hold(s1,'on'); errorbar(4, mu_vel(4), er_vel(4), 'k','LineStyle','none','linewidth',1.5,'parent',s1);
    b1 = bar(5 : 6, mu_vel(5:6),'FaceColor','r','FaceAlpha',0.5,'linewidth',1.5,'parent',s1); hold(s1,'on'); errorbar(5:6, mu_vel(5:6), er_vel(5:6), 'k','LineStyle','none','linewidth',1.5,'parent',s1);
    b2 = bar(7 : 12, L_mu_vel,'FaceColor','b','FaceAlpha',0.5,'linewidth',1.5,'parent',s1); hold(s1,'on'); errorbar(7 : 12, L_mu_vel, L_er_vel, 'k','LineStyle','none','linewidth',1.5,'parent',s1);
    set(s1,'xtick',1:12,'xticklabel',[label label(1:3) {'Z_L_D_S'} label(5:end)]','fontsize',12); ylabel('E_V_E_L (cm/s)');xtickangle(90);
    box off;
    ylim([0.07 0.3])
    
    disp('Position LKF')
    [~,tbl,stat] = friedman(pos_mae,1,'off'); C = multcompare(stat,'CType','bonferroni','Display','off')
    disp('Position LSTM')
    [~,tbl,stat] = friedman(L_pos_mae,1,'off'); C = multcompare(stat,'CType','bonferroni','Display','off')
    s1 = subplot(2,2,n*2); 
    b1 = bar(1 : 3, mu_pos(1:3),'FaceColor','r','FaceAlpha',0.5,'linewidth',1.5,'parent',s1); hold(s1,'on'); errorbar(1 : 3, mu_pos(1:3), er_pos(1:3), 'k','LineStyle','none','linewidth',1.5,'parent',s1);
    b1 = bar(4, mu_pos(4),'FaceColor','r','FaceAlpha',0.5,'linewidth',1.5,'parent',s1); hold(s1,'on'); errorbar(4, mu_pos(4), er_pos(4), 'k','LineStyle','none','linewidth',1.5,'parent',s1);
    b1 = bar(5 : 6, mu_pos(5:6),'FaceColor','r','FaceAlpha',0.5,'linewidth',1.5,'parent',s1); hold(s1,'on'); errorbar(5:6, mu_pos(5:6), er_pos(5:6), 'k','LineStyle','none','linewidth',1.5,'parent',s1);
    b2 = bar(7 : 12, L_mu_pos,'FaceColor','b','FaceAlpha',0.5,'linewidth',1.5,'parent',s1); hold(s1,'on'); errorbar(7 : 12, L_mu_pos, L_er_pos, 'k','LineStyle','none','linewidth',1.5,'parent',s1);
    set(s1,'xtick',1:12,'xticklabel',[label label(1:3) {'Z_L_D_S'} label(5:end)]','fontsize',12);  ylabel('E_P_O_S (cm/s)');xtickangle(90);
    box off;
    ylim([0.5 1.5])
end
save_fig(sprintf('./Figures/segPosPerf_%s',data_name{n}), F);

for n = 1 : 2
    if n == 1
        kinWeight               = 100;
    else
        kinWeight               = 1;
    end
    load(sprintf('DecodingResult_%s.mat',data_name{n}));
    
    mu_vel_corr_lkf             = mean(output_lkf.vel_CC,3)';
    sd_vel_corr_lkf             = std(output_lkf.vel_CC,[],3)';
    mu_vel_corr_lstm            = mean(output_lstm.vel_CC,3)';
    sd_vel_corr_lstm            = std(output_lstm.vel_CC,[],3)';
    
    mu_vel_corr_lds             = mean(squeeze(output_lds.vel_CC),2);
    sd_vel_corr_lds             = std(squeeze(output_lds.vel_CC),[],2);
    
    mu_vel_corr_lkf(:,4) = mu_vel_corr_lds;
    sd_vel_corr_lkf(:,4) = sd_vel_corr_lds;
    
    fprintf('\n lkf: ');
    for j = 1 : 6
        fprintf('(%.2f%s%.2f, %.2f%s%.2f) \t', mu_vel_corr_lkf(1,j), char(177), sd_vel_corr_lkf(1,j), mu_vel_corr_lkf(2,j), char(177), sd_vel_corr_lkf(2,j))
    end
    fprintf('\n lstm: ');
    for j = 1 : 6
        fprintf('(%.2f%s%.2f, %.2f%s%.2f) \t', mu_vel_corr_lstm(1,j), char(177), sd_vel_corr_lstm(1,j), mu_vel_corr_lstm(2,j), char(177), sd_vel_corr_lstm(2,j))
    end
end


for n = 1 : 2
    if n == 1
        kinWeight               = 100;
    else
        kinWeight               = 1;
    end
    load(sprintf('DecodingResult_%s.mat',data_name{n}));
    
    vel_corr_lkf                = output_lkf.vel_CC;
    vel_mae_lkf                 = output_lkf.vel_mae;
    pos_mae_lkf                 = output_lkf.pos_mae;
    vel_mae_lstm                = output_lstm.vel_mae;
    pos_mae_lstm                = output_lstm.pos_mae;
    
    vel_corr_lds                = squeeze(output_lds.vel_CC)';
    vel_mae_lds                 = output_lds.vel_mae(:,1);
    pos_mae_lds                 = output_lds.pos_mae(:,1);

%     vel_mae                     = vel_mae_lkf*kinWeight;
%     pos_mae                     = pos_mae_lkf*kinWeight;
%     L_vel_mae                   = vel_mae_lstm*kinWeight;
%     L_pos_mae                   = pos_mae_lstm*kinWeight;
    vel_mae                     = [vel_mae_lkf(:,1:3) vel_mae_lds vel_mae_lkf(:,[5 6])]*kinWeight;
    pos_mae                     = [pos_mae_lkf(:,1:3) pos_mae_lds pos_mae_lkf(:,[5 6])]*kinWeight;
    L_vel_mae                   = vel_mae_lstm*kinWeight;
    L_pos_mae                   = pos_mae_lstm*kinWeight;
    
    prc_vel(n) = mean((vel_mae(:,end) - L_vel_mae(:,1)) ./ (vel_mae(:,end) + L_vel_mae(:,1))*100)
    prc_pos(n) = mean((pos_mae(:,end) - L_pos_mae(:,1)) ./ (pos_mae(:,end) + L_pos_mae(:,1))*100)

    
    prc_vel(n) = mean((vel_mae(:,end) - L_vel_mae(:,1))./(vel_mae(:,end) + L_vel_mae(:,1)))
    prc_pos(n) = mean((pos_mae(:,end) - L_pos_mae(:,1))./(pos_mae(:,end) + L_pos_mae(:,1)))
    
    mu_vel                      = mean(vel_mae,1);
    mu_pos                      = mean(pos_mae,1);
    sd_vel                      = std(vel_mae);
    sd_pos                      = std(pos_mae);
    
    label
    fprintf('\n velocity: ');
    for j = 1 : 6
        fprintf('%.2f%s%.2f \t', mu_vel(j), char(177), sd_vel(j))
    end
    fprintf('\n position: ');
    for j = 1 : 6
        fprintf('%.2f%s%.2f \t', mu_pos(j), char(177), sd_pos(j))
    end
    
    
    er_vel                      = std(vel_mae,[],1)./sqrt(size(vel_mae,1));
    er_pos                      = std(pos_mae,[],1)./sqrt(size(vel_mae,1));
    
    L_mu_vel                    = mean(L_vel_mae,1);
    L_sd_vel                    = std(L_vel_mae);
    L_mu_pos                    = mean(L_pos_mae,1);
    L_sd_pos                    = std(L_pos_mae);
    L_er_vel                    = std(L_vel_mae,[],1)./sqrt(size(L_vel_mae,1));
    L_er_pos                    = std(L_pos_mae,[],1)./sqrt(size(L_vel_mae,1));
    
    fprintf('\n velocity: ');
    for j = 1 : 6
        fprintf('%.2f%s%.2f \t', L_mu_vel(j), char(177), L_sd_vel(j))
    end
    fprintf('\n position: ');
    for j = 1 : 6
        fprintf('%.2f%s%.2f \t', L_mu_pos(j), char(177), L_sd_pos(j))
    end
    
    
    group                       = zeros(size(vel_mae(:)));
    for k = 1 : 6
        id = (k-1)*size(vel_mae,1)+1:k*size(vel_mae,1);
        group(id)               = k;
    end
    
    disp('Velocity Between Features')
    [~,tbl,stat] = friedman([vel_mae;L_vel_mae],2,'off'); C = multcompare(stat,'CType','bonferroni','Display','off');C(C(:,end) < 0.05,[1:2 end])
    [~,tbl,stat] = friedman([vel_mae(:) L_vel_mae(:)],6,'off'); C = multcompare(stat,'CType','bonferroni','Display','off')
    disp('Velocity Between Decoder')
    for k = 1 : 6
        ranksum(vel_mae(:,k), L_vel_mae(:,k))
    end
    F = figure;clf; set(F,'position',[748  295  433  370]);
    plot(1 : 6, mu_vel,'r','linewidth',1.5); hold on; errorbar(1 : 6, mu_vel, er_vel, 'rs','linewidth',1.5);
    plot(1 : 6, L_mu_vel,'b','linewidth',1.5); hold on; errorbar(1 : 6, L_mu_vel, L_er_vel, 'bs','linewidth',1.5);
    errorbar(4, mu_vel(4), er_vel(4), 's','color','r','linewidth',1.5);
    set(gca,'xticklabel',label','fontsize',12); ylabel('E_V_E_L (cm/s)');
    xlim([0.5 6.5]); box off; if n == 1; ylim([0.1 0.35]); else; ylim([0.1 0.2]);end
    xtickangle(45);
    save_fig(sprintf('./Figures/VelPerf_%s',data_name{n}), F);
    
    
    [~,tbl,stat] = friedman([pos_mae(:) L_pos_mae(:)],6,'off'); C = multcompare(stat,'CType','bonferroni','Display','off')
    disp('Position between features')
    [~,tbl,stat] = friedman([pos_mae;L_pos_mae], 1,'off'); C = multcompare(stat,'CType','bonferroni','Display','off');C(C(:,end) < 0.05,[1:2 end])
    disp('Position Between Decoder')
    for k = 1 : 6
        ranksum(pos_mae(:,k), L_pos_mae(:,k))
    end
    [~,~,stat] = friedman(L_pos_mae, 1,'off'); C = multcompare(stat,'CType','bonferroni','Display','off')
    F = figure;clf; set(F,'position',[748  295  433  370]);
    plot(1 : 6, mu_pos,'r','linewidth',1.5); hold on; errorbar(1 : 6, mu_pos, er_pos, 'rs','linewidth',1.5);
    plot(1 : 6, L_mu_pos,'b','linewidth',1.5); hold on; errorbar(1 : 6, L_mu_pos, er_pos, 'bs','linewidth',1.5);
    errorbar(4, mu_pos(4), er_pos(4), 'rs','linewidth',1.5);
    set(gca,'xticklabel',label','fontsize',12); ylabel('E_P_O_S (cm)');
    xlim([0.5 6.5]); box off; if n == 1; ylim([0.4 1.8]); else; ylim([0.7 1.1]);end
    xtickangle(45);
    save_fig(sprintf('./Figures/PosPerf_%s',data_name{n}), F);
end


