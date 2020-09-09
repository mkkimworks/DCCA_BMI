%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Title: Decoding kinematic information from primary motor cortical
%          ensemble activity using a deep canonical correlation analysis
%   Authors: M-K Kim; J-W Sohn; S-P Kim
%   E-mail: mkkim.works@gmail.com
%   Affiliation: Ulsan National Institute of Science and Technology (UNIST)
%   Copyright 2020. Authors All Rights Reserved.
%
%   Determine the dimensionality of population activity
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


clear; clc; close all;
addpath(genpath('./Functions/'));



dataset(1)                          = load('./Dataset/MonkeyC_Dataset.mat');      % Dataset for Monkey C
dataset(2)                          = load('./Dataset/MonkeyM_Dataset.mat');      % Dataset for Monkey M

K_fold                              = 4;
training_set_ratio                  = 0.75;
bin_length                          = 0.05;
kernSDList                          = 20 : 10 : 250; %msec
comps                               = [2 5 : 5 : 25];
num_kSD                             = length(kernSDList);
num_Comp                            = length(comps);

for n = 1 : 2
    data                            = dataset(n);
    Dat                             = DataSegmentation(data, training_set_ratio,n);
    
    PC_result                       = [];
    FA_result                       = [];
    LD_result                       = [];
    for q = comps
        er_PC = zeros(K_fold,1); er_FA = zeros(K_fold,1); er_LDS = zeros(K_fold,1);
        for CV_k = 1 : K_fold
            seg                         = K_fold_CrossValid(Dat.TrI, Dat.TrZ, Dat.TrX, Dat.TrG, K_fold, CV_k);
            % PCA
            pca_prm                     = fcnPCA(seg.TrZ, q);
            
            % FA
            fa_prm                      = fcnFA(seg.TrZ, q);
            
            % LDS
            [lds_prm, lds_ll]           = EM_LDS(seg.TrZ', q);
            
            
            YC_pca                      = predictNeuralActivity(seg.TeZ, pca_prm, 'pca');
            YC_fa                       = predictNeuralActivity(seg.TeZ, fa_prm, 'fa');
            YC_lds                      = predictNeuralActivity(seg.TeZ, lds_prm, 'lds');
            
            er_PC(CV_k)                 = sum((YC_pca(:) - seg.TeZ(:)).^2);
            er_FA(CV_k)                 = sum((YC_fa(:) - seg.TeZ(:)).^2);
            er_LDS(CV_k)                = sum((YC_lds(:) - seg.TeZ(:)).^2);

            disp([q CV_k])
        end
        PC_result                       = cat(2, PC_result, er_PC)
        FA_result                       = cat(2, FA_result, er_FA)
        LD_result                       = cat(2, LD_result, er_LDS)
    end
    [~,q_pc]                            = min(mean(PC_result,1)); q_pc = comps(q_pc);
    [~,q_fa]                            = min(mean(FA_result,1)); q_fa = comps(q_fa);
    [~,q_lds]                           = min(mean(LD_result,1)); q_lds= comps(q_lds); if q_lds > 20 ; q_lds = 20; end
    
    smooth_er_PC                        = zeros(num_kSD,K_fold); 
    smooth_er_FA                        = zeros(num_kSD,K_fold); 
    for i = 1 : num_kSD
        fTrZ                            = gaussian_smoothing(Dat.TrZ, bin_length, kernSDList(i));
        for CV_k = 1 : K_fold
            seg                         = K_fold_CrossValid(Dat.TrI, fTrZ, Dat.TrX, Dat.TrG, K_fold, CV_k);
            
            % PCA
            pca_prm                     = fcnPCA(seg.TrZ, q_pc);
            % FA
            fa_prm                      = fcnFA(seg.TrZ, q_fa);
            
            beta = seg.TrX' * [ones(size(seg.TrZ,1),1) pca_prm.fcn(seg.TrZ)] * ([ones(size(seg.TrZ,1),1) pca_prm.fcn(seg.TrZ)]' * [ones(size(seg.TrZ,1),1) pca_prm.fcn(seg.TrZ)])^-1;
            smooth_er_PC(i,CV_k)        = mae([ones(size(seg.TeZ,1),1) pca_prm.fcn(seg.TeZ)] * beta', seg.TeX);
            beta = seg.TrX' * [ones(size(seg.TrZ,1),1) fa_prm.fcn(seg.TrZ)] * ([ones(size(seg.TrZ,1),1) fa_prm.fcn(seg.TrZ)]' * [ones(size(seg.TrZ,1),1) fa_prm.fcn(seg.TrZ)])^-1;
            smooth_er_FA(i,CV_k)        = mae([ones(size(seg.TeZ,1),1) fa_prm.fcn(seg.TeZ)] * beta', seg.TeX);
            disp([i CV_k])
        end
    end   
    [~,ind] =min(mean(smooth_er_PC,2));kernSDList(ind);
    save(sprintf('./Parameters/determine_comp_%d.mat',n), 'PC_result', 'FA_result', 'LD_result','smooth_er_PC','smooth_er_FA');
end

for n = 1 : 2
    load(sprintf('./Parameters/determine_comp_%d.mat', n))
    
    err_pc                              = mean(PC_result,1);
    err_fa                              = mean(FA_result,1);
    err_ld                              = mean(LD_result,1);

    [merr,ind]                          = min(err_pc,[],2);
    num_comps(n).pc                     = comps(ind);

    [merr,ind]                          = min(err_fa,[],2);
    num_comps(n).fa                     = comps(ind);
    
    [merr,ind]                          = min(err_ld,[],2);
    num_comps(n).ld                     = comps(ind);
    
    stack_err(:,1,n)                    = err_pc;
    stack_err(:,2,n)                    = err_fa;
    stack_err(:,3,n)                    = err_ld;
end
save ./Parameters/num_comps.mat num_comps;

%%
for n = 1 : 2
    load(sprintf('./Parameters/determine_comp_%d.mat',n));
    F = figure(1);clf; set(F,'position',[748   438   337   327]);
    colors = lines(3);
    for k = 1 : 3
        h(k) = semilogy(comps, stack_err(:,k,n),'o-','color',colors(k,:)); hold on;
        [value, ind] = min(stack_err(:,k,n),[],1);
        semilogy(comps(ind), value,'o','MarkerFaceColor', colors(k,:),'linewidth',1.5,'color',colors(k,:))
    end
    ylim([0 stack_err(5,1,n)])
    ylabel('Prediction error'); xlabel('q'); 
    set (gca,'fontsize',12); box off; 
    if n == 1
        lg = legend(h, 'PCA','FA','LDS'); lg.Box = 'off'; %lg.Position = [0.1868    0.7077    0.2406    0.1802];
    end
    save_fig(sprintf('./Figures/NumComponents_sub_%d',n), F);
end

%%

%%
data_name                           = {'CV_MonkeyC_Data','CV_MonkeyM_Data'};
load('./Parameters/deep_CCA_hyperparameters.mat');
load('./Parameters/num_comps.mat');
kernSDList                          = 20 : 10 : 250; %msec
num_kSD                             = length(kernSDList);
for n = 1 : 2
    data                            = dataset(n);
    Dat                             = DataSegmentation(data, training_set_ratio,n);
    smooth_er_Z                         = zeros(num_kSD,K_fold);
    smooth_er_PC                        = zeros(num_kSD,K_fold);
    smooth_er_FA                        = zeros(num_kSD,K_fold);
    smooth_er_LC                        = zeros(num_kSD,K_fold);
    smooth_er_DC                        = zeros(num_kSD,K_fold);
    for i = 1 : num_kSD
        fTrZ                            = gaussian_smoothing(Dat.TrZ, bin_length, kernSDList(i));
        for CV_k = 1 : K_fold
            seg                         = K_fold_CrossValid(Dat.TrI, fTrZ, Dat.TrX, Dat.TrG, K_fold, CV_k);
            
            % PCA
            pca_prm                     = fcnPCA(seg.TrZ, num_comps(n).pc);
            % FA
            fa_prm                      = fcnFA(seg.TrZ, num_comps(n).fa);
            % LCCA
            lcca_prm                    = fcnCCA(Dat.TrZ, Dat.TrX);
            % DCCA
            dcca_prm                    = fcnDeepCCA(Dat.TrZ, Dat.TrX, Dat.TrI, data_name{n}, net_param(n), CV_k);
            
            beta = seg.TrX' * [ones(size(seg.TrZ,1),1) (seg.TrZ)] * ([ones(size(seg.TrZ,1),1) (seg.TrZ)]' * [ones(size(seg.TrZ,1),1) (seg.TrZ)])^-1;
            smooth_er_Z(i,CV_k)        = mae([ones(size(seg.TeZ,1),1) (seg.TeZ)] * beta', seg.TeX);
            beta = seg.TrX' * [ones(size(seg.TrZ,1),1) pca_prm.fcn(seg.TrZ)] * ([ones(size(seg.TrZ,1),1) pca_prm.fcn(seg.TrZ)]' * [ones(size(seg.TrZ,1),1) pca_prm.fcn(seg.TrZ)])^-1;
            smooth_er_PC(i,CV_k)        = mae([ones(size(seg.TeZ,1),1) pca_prm.fcn(seg.TeZ)] * beta', seg.TeX);
            beta = seg.TrX' * [ones(size(seg.TrZ,1),1) fa_prm.fcn(seg.TrZ)] * ([ones(size(seg.TrZ,1),1) fa_prm.fcn(seg.TrZ)]' * [ones(size(seg.TrZ,1),1) fa_prm.fcn(seg.TrZ)])^-1;
            smooth_er_FA(i,CV_k)        = mae([ones(size(seg.TeZ,1),1) fa_prm.fcn(seg.TeZ)] * beta', seg.TeX);
            beta = lcca_prm.fcn.B(seg.TrX)' * [ones(size(seg.TrZ,1),1) lcca_prm.fcn.A(seg.TrZ)] * ([ones(size(seg.TrZ,1),1) lcca_prm.fcn.A(seg.TrZ)]' * [ones(size(seg.TrZ,1),1) lcca_prm.fcn.A(seg.TrZ)])^-1;
            smooth_er_LC(i,CV_k)        = mae([ones(size(seg.TeZ,1),1) lcca_prm.fcn.A(seg.TeZ)] * beta', lcca_prm.fcn.B(seg.TeX));
            beta = dcca_prm.fcn.B(seg.TrX)' * [ones(size(seg.TrZ,1),1) dcca_prm.fcn.A(seg.TrZ)] * ([ones(size(seg.TrZ,1),1) dcca_prm.fcn.A(seg.TrZ)]' * [ones(size(seg.TrZ,1),1) dcca_prm.fcn.A(seg.TrZ)])^-1;
            smooth_er_DC(i,CV_k)        = mae([ones(size(seg.TeZ,1),1) dcca_prm.fcn.A(seg.TeZ)] * beta', dcca_prm.fcn.B(seg.TeX));
            disp([i CV_k])
        end
    end
    [~,ind] =min(mean(smooth_er_Z,2));  FR_SD(n) = kernSDList(ind);
    [~,ind] =min(mean(smooth_er_PC,2)); PC_SD(n) = kernSDList(ind);
    [~,ind] =min(mean(smooth_er_FA,2)); FA_SD(n) = kernSDList(ind);
    [~,ind] =min(mean(smooth_er_LC,2)); LC_SD(n) = kernSDList(ind);
    [~,ind] =min(mean(smooth_er_DC,2)); DC_SD(n) = kernSDList(ind);
end
save ./Parameters/KernSD1.mat FR_SD PC_SD FA_SD LC_SD DC_SD smooth_er_Z smooth_er_PC smooth_er_FA smooth_er_LC smooth_er_DC;
