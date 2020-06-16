%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Title: Decoding kinematic information from primary motor cortical
%          ensemble activity using a deep canonical correlation analysis
%   Authors: M-K Kim; J-W Sohn; S-P Kim
%   E-mail: mkkim.works@gmail.com
%   Affiliation: Ulsan National Institute of Science and Technology (UNIST)
%   Copyright 2020. Authors All Rights Reserved.
%
%
%   Only used Dataset CRT in this example
%    
%   Matlab version: R2019b ('9.7.0.1296695 (R2019b) Update 4')
%   Utilized toolboxes
%   - Deep Learning Toolbox in Matlab
%   - DCCA toolbox (https://ttic.uchicago.edu/~wwang5/dccae.html)
%   Dataset
%   - Dataset CRT (authorization required {non-distributable})
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


clear; clc; close all;

addpath('./DCCA')
addpath('./DCCA/deepnet')
addpath('./Decoders/');
addpath('./DrawPlot/');
load('./Dataset/monkey_data.mat');      % Dataset
load('./Dataset/DCCA_param.mat');       % example: trained DCCA parameter
load('./Dataset/LSTM_param.mat');           % example: trained LSTM parameter

feature_labels              = {'Z_E_-_F_R', 'Z_L_C_V','Z_D_C_V'};
sz                          = get(0, 'ScreenSize');
sz                          = sz(3:4);

dim                         = size(TrX,2);
interp_len                  = 20;
TargetDegrees               = unique(Target);
Trial_ID                    = unique(TeID);
NumTargets                  = length(TargetDegrees);
LenTrials                   = length(Trial_ID);


% Linear CCA
lcca_prm                    = fcnCCA(TrZ, TrX);
TrLCCA_Z                    = lcca_prm.fcn.A(TrZ);
TeLCCA_Z                    = lcca_prm.fcn.A(TeZ);
TrLCCA_X                    = lcca_prm.fcn.B(TrX);
TeLCCA_X                    = lcca_prm.fcn.B(TeX);

% Deep CCA
NumOptRepeats               = 500;
dcca_prm                    = fcnDCCA(TrZ, TrX, NumOptRepeats, DCCA_param);
TrDCCA_Z                    = dcca_prm.fcn.A(TrZ);
TeDCCA_Z                    = dcca_prm.fcn.A(TeZ);
TrDCCA_X                    = dcca_prm.fcn.B(TrX);
TeDCCA_X                    = dcca_prm.fcn.B(TeX);
%%
% Training Kalman filter
R_FR_KF                     = KFTraining(TrZ, TrX);
LCCA_KF                     = KFTraining(TrLCCA_Z, TrLCCA_X);
DCCA_KF                     = KFTraining(TrDCCA_Z, TrDCCA_X);

% Training LSTM-RNN
NumOptRepeats               = 500;
LSTM_R                      = fcnLSTM(TrZ, TrX, NumOptRepeats, lstm_param{1});
LSTM_L                      = fcnLSTM(TrLCCA_Z, TrLCCA_X, NumOptRepeats, lstm_param{2});
LSTM_D                      = fcnLSTM(TrDCCA_Z, TrDCCA_X, NumOptRepeats, lstm_param{3});

% Simulation
TargetInd                   = zeros(NumTargets,1);
vED_LKF                     = zeros(LenTrials, 3);
pED_LKF                     = zeros(LenTrials, 3);
vED_RNN                     = zeros(LenTrials, 3);
pED_RNN                     = zeros(LenTrials, 3);

True_POS                    = zeros(interp_len, dim-1, LenTrials, NumTargets);
R_phat_LKF                  = zeros(interp_len, dim-1, LenTrials, NumTargets);
L_phat_LKF                  = zeros(interp_len, dim-1, LenTrials, NumTargets);
D_phat_LKF                  = zeros(interp_len, dim-1, LenTrials, NumTargets);

R_phat_RNN                  = zeros(interp_len, dim-1, LenTrials, NumTargets);
L_phat_RNN                  = zeros(interp_len, dim-1, LenTrials, NumTargets);
D_phat_RNN                  = zeros(interp_len, dim-1, LenTrials, NumTargets);
for i = 1 : LenTrials
    targ_i                  = Target(i) == TargetDegrees;
    TargetInd(targ_i)       = TargetInd(targ_i) + 1;
    
    ind                     = Trial_ID(i) == TeID;
    vtrue                   = TeX(ind,1:end-1);
    roi                     = ~isnan(sum(vtrue,2));
    ptrue                   = cumsum([0 0; vtrue(roi,:)]);
    Z_R_FR                  = TeZ(ind,:);
    Z_LCCA                  = TeLCCA_Z(ind,:);
    Z_DCCA                  = TeDCCA_Z(ind,:);
    
    len                     = sum(ind);
    R_vhat_LKF              = zeros(len, dim);
    L_vhat_LKF              = zeros(len, dim);
    D_vhat_LKF              = zeros(len, dim);
    
    R_vhat_RNN              = double(predict(LSTM_R, Z_R_FR'))';
    L_vhat_RNN              = double(predict(LSTM_L, Z_LCCA'))';
    D_vhat_RNN              = double(predict(LSTM_D, Z_DCCA'))';
    for k = 1 : len
        [R_vhat_LKF(k,:), R_FR_KF] = KFpredict(Z_R_FR(k,:), R_FR_KF);
        [L_vhat_LKF(k,:), LCCA_KF] = KFpredict(Z_LCCA(k,:), LCCA_KF);
        [D_vhat_LKF(k,:), DCCA_KF] = KFpredict(Z_DCCA(k,:), DCCA_KF);
    end
    L_vhat_LKF              = lcca_prm.fcn.invB(L_vhat_LKF);
    D_vhat_LKF              = dcca_prm.fcn.invB(D_vhat_LKF); D_vhat_LKF = D_vhat_LKF(:,1:end-1);
    L_vhat_RNN              = lcca_prm.fcn.invB(L_vhat_RNN);
    D_vhat_RNN              = dcca_prm.fcn.invB(D_vhat_RNN); D_vhat_RNN = D_vhat_RNN(:,1:end-1);
    
    R_FR_KF                 = InitLKF(R_FR_KF);
    LCCA_KF                 = InitLKF(LCCA_KF);
    DCCA_KF                 = InitLKF(DCCA_KF);
    
    R_vhat_LKF              = R_vhat_LKF(roi,1:end-1);
    L_vhat_LKF              = L_vhat_LKF(roi,1:end-1);
    D_vhat_LKF              = D_vhat_LKF(roi,1:end-1);
    
    R_vhat_RNN              = R_vhat_RNN(roi,1:end-1);
    L_vhat_RNN              = L_vhat_RNN(roi,1:end-1);
    D_vhat_RNN              = D_vhat_RNN(roi,1:end-1);
    
    VTrue                   = vtrue(roi,:);
    Velocity(i).LKF         = table(VTrue, R_vhat_LKF, L_vhat_LKF, D_vhat_LKF);
    Velocity(i).RNN         = table(VTrue, R_vhat_RNN, L_vhat_RNN, D_vhat_RNN);
    
    
    R_phat_LKF_temp         = cumsum([0 0; R_vhat_LKF]);
    L_phat_LKF_temp         = cumsum([0 0; L_vhat_LKF]);
    D_phat_LKF_temp         = cumsum([0 0; D_vhat_LKF]);
    
    R_phat_RNN_temp         = cumsum([0 0; R_vhat_RNN]);
    L_phat_RNN_temp         = cumsum([0 0; L_vhat_RNN]);
    D_phat_RNN_temp         = cumsum([0 0; D_vhat_RNN]);
    
    Position(i).LKF         = table(ptrue, R_phat_LKF_temp, L_phat_LKF_temp, D_phat_LKF_temp);
    Position(i).RNN         = table(ptrue, R_phat_RNN_temp, L_phat_RNN_temp, D_phat_RNN_temp);
    
    vED_LKF(i,:)            = [CalcED(R_vhat_LKF, vtrue) CalcED(L_vhat_LKF, vtrue) CalcED(D_vhat_LKF, vtrue)];
    pED_LKF(i,:)            = [CalcED(R_phat_LKF_temp, ptrue) CalcED(L_phat_LKF_temp, ptrue) CalcED(D_phat_LKF_temp, ptrue)];
    vED_RNN(i,:)            = [CalcED(R_vhat_RNN, vtrue) CalcED(L_vhat_RNN, vtrue) CalcED(D_vhat_RNN, vtrue)];
    pED_RNN(i,:)            = [CalcED(R_phat_RNN_temp, ptrue) CalcED(L_phat_RNN_temp, ptrue) CalcED(D_phat_RNN_temp, ptrue)];
    len                     = size(ptrue,1);
    
    True_POS(:,:,TargetInd(targ_i),targ_i)         = pchip(linspace(0, 1, len), ptrue', linspace(0, 1, interp_len))';

    R_phat_LKF(:,:,TargetInd(targ_i),targ_i)       = pchip(linspace(0, 1, len), R_phat_LKF_temp', linspace(0, 1, interp_len))';
    L_phat_LKF(:,:,TargetInd(targ_i),targ_i)       = pchip(linspace(0, 1, len), L_phat_LKF_temp', linspace(0, 1, interp_len))';
    D_phat_LKF(:,:,TargetInd(targ_i),targ_i)       = pchip(linspace(0, 1, len), D_phat_LKF_temp', linspace(0, 1, interp_len))';
    
    R_phat_RNN(:,:,TargetInd(targ_i),targ_i)       = pchip(linspace(0, 1, len), R_phat_RNN_temp', linspace(0, 1, interp_len))';
    L_phat_RNN(:,:,TargetInd(targ_i),targ_i)       = pchip(linspace(0, 1, len), L_phat_RNN_temp', linspace(0, 1, interp_len))';
    D_phat_RNN(:,:,TargetInd(targ_i),targ_i)       = pchip(linspace(0, 1, len), D_phat_RNN_temp', linspace(0, 1, interp_len))';
end

%% Figure 2
col = [1 0.5 0 ; 0 0.5 1];

fig = figure('position',[ 565   255   503   723]); clf;

[len,dim]                   = size(TeLCCA_Z);
id                          = randperm(len); id = id(1 : 500);
for d = 1 : dim
    X                       = TeLCCA_X(:,d);
    Y                       = TeLCCA_Z(:,d);
    X                       = X - nanmean(X);
    Y                       = Y - nanmean(Y);
    beta                    = regress(Y, [ones(size(X)) X]);
    Yhat                    = [ones(size(X)) X] * beta;
    subplot(3,2,(d-1)*2+1); hold on;
    plot(X, Y, '^','color', col(1,:),'MarkerSize',5);
    plot(X, Yhat, 'color', [0 0 0],'linewidth',2);
    box off;
    set(gca,'xticklabel',{}, 'yticklabel',{});
    xlabel(sprintf('X_L_C_V (dim %d)', d))
    ylabel(sprintf('Z_L_C_V (dim %d)', d))
    
    X                       = TeDCCA_X(:,d);
    Y                       = TeDCCA_Z(:,d);
    X                       = X - nanmean(X);
    Y                       = Y - nanmean(Y);
    beta                    = regress(Y, [ones(size(X)) X]);
    Yhat                    = [ones(size(X)) X] * beta;
    subplot(3,2,d*2); hold on;
    plot(X, Y,'^','color',col(1,:),'MarkerSize',5);
    plot(X, Yhat, 'color', [0 0 0],'linewidth',2);
    box off;
    set(gca,'xticklabel',{}, 'yticklabel',{});
    xlabel(sprintf('X_D_C_V (dim %d)', d))
    ylabel(sprintf('Z_D_C_V (dim %d)', d))
end

%% Figure 3
X                           = TrX;
for d = 1 : size(TrZ,2)
    [b,~,~,~,stats]             = regress(TrZ(:,d),[ones(size(X,1),1) X]);
    raw_r2(d,1)                 = stats(1);
    raw_pv(d,1)                 = stats(end-1);
end
[~,max_cell]                    = sort(raw_r2); max_cell = flipud(max_cell);


r2 = []; pv = []; 
b0 = []; b1 = []; b2 = []; b3 = [];
for d = 1 : 3
    [b0(:,d),~,~,~,stats]   = regress(TrZ(:,max_cell(d)),[ones(size(X,1),1) X]);
    r2(d,1)                 = stats(1);
    pv(d,1)                 = stats(end-1);
    
    [b2(:,d),~,~,~,stats]   = regress(TrLCCA_Z(:,d),[ones(size(X,1),1) X]);
    r2(d,3)                 = stats(1);
    pv(d,3)                 = stats(end-1);
    
    [b3(:,d),~,~,~,stats]   = regress(TrDCCA_Z(:,d),[ones(size(X,1),1) X]);
    r2(d,4)                 = stats(1);
    pv(d,4)                 = stats(end-1);
end

Zhat_0                      = [ones(size(TeX,1),1) TeX] * b0;
Zhat_2                      = [ones(size(TeX,1),1) TeX] * b2;
Zhat_3                      = [ones(size(TeX,1),1) TeX] * b3;
    
uni                         = unique(TeID);
id                          = find(TeID >= uni(5) & TeID <= uni(15));
time                        = (1:length(id)) * 0.05 - 0.05;
del_t                       = ~isnan(sum(Zhat_0(id,:),2));
Zhat_0                      = Zhat_0(id(del_t),:);
Zhat_2                      = Zhat_2(id(del_t),:);
Zhat_3                      = Zhat_3(id(del_t),:);

time                        = time(del_t);

TeZ_                        = TeZ(id(del_t),:);
TeLCCA_Z_                   = TeLCCA_Z(id(del_t),:);
TeDCCA_Z_                   = TeDCCA_Z(id(del_t),:);
TeID_                       = TeID(id(del_t));
onset                       = [1;find(diff(TeID_))+1];% onset = onset(1:end-1);

fig = figure('position',[565 741 1261 237]); clf; col = [1. 0. 0 ; 0 0 0];
for d = 1 : 3
    sh1(d) = subplot(3,3,(d-1)*3+1,'parent',fig); hold on;
    plot(time',TeZ_(:,max_cell(d,1)),'color',col(2,:),'linewidth',1.5); plot(time,Zhat_0(:,d),'color',col(1,:),'linewidth',1.5);
    xlim([time(1) time(end)]); set(gca,'xticklabel',{}, 'yticklabel',{});
    if d == 1; title(sprintf('Z_E_-_F_R')); end
    ylabel(sprintf('dim %d',d));
    if d == 3;plot(time(onset), ones(size(onset))*0,'k^','MarkerEdgeColor','k','MarkerFaceColor','k','MarkerSize',5); end
    sh2(d) = subplot(3,3,(d-1)*3+2,'parent',fig); hold on;
    plot(time,TeLCCA_Z_(:,d),'color',col(2,:),'linewidth',1.5); plot(time,Zhat_2(:,d),'color',col(1,:),'linewidth',1.5);
    xlim([time(1) time(end)]); set(gca,'xticklabel',{}, 'yticklabel',{});
    if d == 1; title(sprintf('Z_L_C_V')); end
    ylabel(sprintf('dim %d',d));
    if d == 3;plot(time(onset), ones(size(onset))*-5,'k^','MarkerEdgeColor','k','MarkerFaceColor','k','MarkerSize',5); end
    sh4(d) = subplot(3,3,(d-1)*3+3,'parent',fig); hold on;
    plot(time,TeDCCA_Z_(:,d),'color',col(2,:),'linewidth',1.5); plot(time,Zhat_3(:,d),'color',col(1,:),'linewidth',1.5);
    xlim([time(1) time(end)]); set(gca,'xticklabel',{}, 'yticklabel',{});
    if d == 1; title(sprintf('Z_D_C_V')); end
    ylabel(sprintf('dim %d',d));
    if d == 3;plot(time(onset), ones(size(onset))*-5,'k^','MarkerEdgeColor','k','MarkerFaceColor','k','MarkerSize',5); end
end
for d = 1 : 3
    ylim_1(d,:) = sh1(d).YLim;
    ylim_2(d,:) = sh2(d).YLim;
    ylim_4(d,:) = sh4(d).YLim;
end
ax1 = [min(ylim_1(:,1)) max(ylim_1(:,2))];
ax2 = [min(ylim_2(:,1)) max(ylim_2(:,2))];
ax4 = [min(ylim_4(:,1)) max(ylim_4(:,2))];

for d = 1 : 3
    sh1(d).YLim = ax1;
    sh2(d).YLim = ax2;
    sh4(d).YLim = ax4;
end


%% Figure 4
r2 = []; 
X                               = TrX;
for d = 1 : dim
    [b,~,~,~,stats]             = regress(TrLCCA_Z(:,d),[ones(size(X,1),1) X]);
    r2(d,1)                     = stats(1);
    pv(d,1)                     = stats(end-1);
    [b,~,~,~,stats]             = regress(TrDCCA_Z(:,d),[ones(size(X,1),1) X]);
    r2(d,2)                     = stats(1);
    pv(d,2)                     = stats(end-1);
end

fig = figure(1);clf; set(fig, 'position',[680   551   271   427])
[rh,rind] = histcounts(raw_r2, 0:0.01:1);

for j = 1 : length(rind)-1
    ID = find(raw_r2 >= rind(j) & raw_r2 <= rind(j+1));
    pf =  normrnd(0, rh(j)/sum(rh)*0.5, [length(ID),1]);
    plot(pf + 1, raw_r2(ID,1),'k.','markersize',10); hold on;
end
plot([0.8 1.2], [mean(raw_r2) mean(raw_r2)],'r','linewidth',1.5)
plot(2, r2(:,1),'k.','markersize',10); hold on; plot([1.8 2.2], [mean(r2(:,1)) mean(r2(:,1))],'r','linewidth',1.5)
plot(3, r2(:,2),'k.','markersize',10); hold on; plot([2.8 3.2], [mean(r2(:,2)) mean(r2(:,2))],'r','linewidth',1.5)
axis([0.5 3.5 0 1]); box off;
set(gca,'xtick',1:3, 'xticklabel',{'Z_E_-_F_R', 'Z_L_C_V','Z_D_C_V'}); ylabel('r^2');

sorted_r2_unit_ind          = [152, 65, 155]; % from max_cell

X                           = TrX(:,1:2); 
[XX,YY]                     = meshgrid((-1:0.01:1)', (-1:0.01:1)');
InterpXY                    = [XX(:) YY(:)];
prb                         = mvnpdf(InterpXY,[0 0],eye(2)*.0001); prb = reshape(prb, sqrt(length(prb)),sqrt(length(prb)));

len                         = 20;
x_vel                       = linspace(min(X(:,1)),max(X(:,1)), len)';      x_vel = [x_vel ; x_vel(end)+(x_vel(end)-x_vel(end-1))];
y_vel                       = linspace(min(X(:,2)),max(X(:,2)), len)';      y_vel = [y_vel ; y_vel(end)+(y_vel(end)-y_vel(end-1))];
fig = figure(2);clf; colormap(jet); set(fig, 'position',[594   627   966   711]);
for d = 1 : 3
    origin                  = nan(length(x_vel)-1, length(y_vel)-1);
    lcca_z                  = nan(length(x_vel)-1, length(y_vel)-1);
    dcca_z                  = nan(length(x_vel)-1, length(y_vel)-1);
    for u1 = 1 : length(x_vel)-1
        indx                = X(:,1) >= x_vel(u1) & X(:,1) < x_vel(u1+1);
        for u2 = 1 : length(y_vel)-1
            indy                = X(:,2) >= y_vel(u2) & X(:,2) < y_vel(u2+1);
            ind                 = indx & indy;
            origin(u1,u2)       = nanmean(TrZ(ind,sorted_r2_unit_ind(d)));
            lcca_z(u1,u2)       = nanmean(TrLCCA_Z(ind,d));
            dcca_z(u1,u2)       = nanmean(TrDCCA_Z(ind,d));
        end
    end
    
    o_nanid                     = isnan(origin);
    l_nanid                     = isnan(lcca_z);
    d_nanid                     = isnan(dcca_z);
    
    origin(o_nanid)             = 0;
    lcca_z(l_nanid)             = 0;
    dcca_z(d_nanid)             = 0;
    
    f_origin                    = origin;
    f_lcca_z                    = lcca_z;
    f_dcca_z                    = dcca_z;
    
    f_origin(o_nanid)           = nan;
    f_lcca_z(l_nanid)           = nan;
    f_dcca_z(d_nanid)           = nan;
    f_origin                    = (f_origin - nanmean(f_origin(:)))./nanstd(f_origin(:));
    f_lcca_z                    = (f_lcca_z - nanmean(f_lcca_z(:)))./nanstd(f_lcca_z(:));
    f_dcca_z                    = (f_dcca_z - nanmean(f_dcca_z(:)))./nanstd(f_dcca_z(:));
    
    sh= subplot(3,3,(d-1)*3+1); pcolor(x_vel(1:end-1), y_vel(1:end-1), f_origin,'parent',sh); shading flat; caxis(sh, [-2 2]); set(gca,'fontsize',13); xlabel('V_X'); ylabel('V_Y');
    sh= subplot(3,3,(d-1)*3+2); pcolor(x_vel(1:end-1), y_vel(1:end-1), f_lcca_z,'parent',sh); shading flat; caxis(sh, [-2 2]); set(gca,'fontsize',13); xlabel('V_X'); ylabel('V_Y');
    sh= subplot(3,3,(d-1)*3+3); pcolor(x_vel(1:end-1), y_vel(1:end-1), f_dcca_z,'parent',sh); shading flat; caxis(sh, [-2 2]); set(gca,'fontsize',13); xlabel('V_X'); ylabel('V_Y');
end


%% Figure 5
uni                             = unique(TeID);
id                              = TeID >= uni(20) & TeID <= uni(30);
time                            = (1:sum(id)) * 0.05 - 0.05;
onset                           = [1;find(diff(TeID(id)))+1];
col     = [0.4 * ones(1,3) ; 0.7 * ones(1,3)];
LKF_lists                   = Velocity(1).LKF.Properties.VariableNames;
RNN_lists                   = Velocity(1).RNN.Properties.VariableNames;
fig2    = figure('position',[565         484        1279         155]); clf; 
for zt = 1 : 3     
    true_v                      = [];
    LKF_v                       = [];
    LSTM_v                      = [];
    for k = 1 : size(Velocity,2)
        true_v                  = [true_v ; Velocity(k).LKF.VTrue*100];
        LKF_v                   = [LKF_v ; Velocity(k).LKF.(LKF_lists{zt+1})*100];
        LSTM_v                  = [LSTM_v ; Velocity(k).RNN.(RNN_lists{zt+1})*100];
    end
        
    sh = subplot(1,3,zt,'parent',fig2);hold(sh,'on');
    plot(time, true_v(id,1)+3,'color',col(2,:),'linewidth',2,'parent',sh)
    plot(time, LKF_v(id,1)+3,'color','r','linewidth',1.5,'parent',sh)
    plot(time, LSTM_v(id,1)+3,'color','b','linewidth',1.5,'parent',sh)
    
    plot(time, true_v(id,1),'color',col(2,:),'linewidth',1.5,'parent',sh)
    plot(time, LKF_v(id,1),'color','r','linewidth',1.5,'parent',sh)
    plot(time, LSTM_v(id,1),'color','b','linewidth',1.5,'parent',sh)
    axis(sh,[time(1) time(end) -2.5 5])
    plot(time(onset), ones(size(onset))*-2.5,'^','MarkerFaceColor','k','MarkerEdgeColor','k','MarkerSize',5,'Parent',sh);
    set(sh,'yticklabel',{});
    title(sprintf('%s',feature_labels{zt}));
    if zt == 3; legend('True velocity','LKF','LSTM-RNN'); end;
    xlabel('time [sec]');
end

%% Figure 6
figure; 
h1 = subplot(131); fcnErrorbar2D({True_POS, TargetInd},'color',ones(1,3)*0.6,'handles',h1); hold(h1,'on'); fcnErrorbar2D({R_phat_RNN, TargetInd},'color','b','handles',h1); fcnErrorbar2D({R_phat_LKF, TargetInd},'color','r','handles',h1, 'title','Z_E_-_F_R'); 
h1 = subplot(132); fcnErrorbar2D({True_POS, TargetInd},'color',ones(1,3)*0.6,'handles',h1); hold(h1,'on'); fcnErrorbar2D({L_phat_RNN, TargetInd},'color','b','handles',h1); fcnErrorbar2D({L_phat_LKF, TargetInd},'color','r','handles',h1, 'title','Z_E_-_F_R'); 
h1 = subplot(133); fcnErrorbar2D({True_POS, TargetInd},'color',ones(1,3)*0.6,'handles',h1); hold(h1,'on'); fcnErrorbar2D({D_phat_RNN, TargetInd},'color','b','handles',h1); fcnErrorbar2D({D_phat_LKF, TargetInd},'color','r','handles',h1, 'title','Z_E_-_F_R'); 




%% Figure 7 - 8

NumTrials                       = size(Velocity,2);
vLKF_Z                           = zeros(NumTrials,1);
vLKF_L                           = zeros(NumTrials,1);
vLKF_D                           = zeros(NumTrials,1);
vRNN_Z                           = zeros(NumTrials,1);
vRNN_L                           = zeros(NumTrials,1);
vRNN_D                           = zeros(NumTrials,1);

pLKF_Z                           = zeros(NumTrials,1);
pLKF_L                           = zeros(NumTrials,1);
pLKF_D                           = zeros(NumTrials,1);
pRNN_Z                           = zeros(NumTrials,1);
pRNN_L                           = zeros(NumTrials,1);
pRNN_D                           = zeros(NumTrials,1);
for k = 1 : NumTrials
    vtrue                       = Velocity(k).LKF.VTrue;
    z_efr                       = Velocity(k).LKF.R_vhat_LKF;
    z_lcv                       = Velocity(k).LKF.L_vhat_LKF;
    z_dcv                       = Velocity(k).LKF.D_vhat_LKF;
    vLKF_Z(k)                   = CalcED(z_efr, vtrue);
    vLKF_L(k)                   = CalcED(z_lcv, vtrue);
    vLKF_D(k)                   = CalcED(z_dcv, vtrue);
    
    ptrue                       = Position(k).LKF.ptrue;
    z_efr                       = Position(k).LKF.R_phat_LKF_temp;
    z_lcv                       = Position(k).LKF.L_phat_LKF_temp;
    z_dcv                       = Position(k).LKF.D_phat_LKF_temp;
    pLKF_Z(k)                   = CalcED(z_efr, ptrue);
    pLKF_L(k)                   = CalcED(z_lcv, ptrue);
    pLKF_D(k)                   = CalcED(z_dcv, ptrue);
    
    
    z_efr                       = Velocity(k).RNN.R_vhat_RNN;
    z_lcv                       = Velocity(k).RNN.L_vhat_RNN;
    z_dcv                       = Velocity(k).RNN.D_vhat_RNN;
    vRNN_Z(k)                   = CalcED(z_efr, vtrue);
    vRNN_L(k)                   = CalcED(z_lcv, vtrue);
    vRNN_D(k)                   = CalcED(z_dcv, vtrue);
    z_efr                       = Position(k).RNN.R_phat_RNN_temp;
    z_lcv                       = Position(k).RNN.L_phat_RNN_temp;
    z_dcv                       = Position(k).RNN.D_phat_RNN_temp;
    pRNN_Z(k)                   = CalcED(z_efr, ptrue);
    pRNN_L(k)                   = CalcED(z_lcv, ptrue);
    pRNN_D(k)                   = CalcED(z_dcv, ptrue);
end

mu_lkf                          = mean([vLKF_Z vLKF_L vLKF_D]);
mu_rnn                          = mean([vRNN_Z vRNN_L vRNN_D]);
er_lkf                          = std([vLKF_Z vLKF_L vLKF_D])./sqrt(NumTrials);
er_rnn                          = std([vRNN_Z vRNN_L vRNN_D])./sqrt(NumTrials);

fig = figure('position',[680   488   700   490]);clf; hold on;
errorbar(1 : 3, mu_lkf, er_lkf,'ro-','linewidth',2,'markersize',10); hold on;
errorbar(1 : 3, mu_rnn, er_rnn,'bo-','linewidth',2,'markersize',10)
set(gca,'xtick',1:3,'xticklabel', feature_labels,'fontsize',13); xlim([0.7 3.3]);
ylabel('E_V_E_L (cm/s)','fontsize',13);

mu_lkf                          = mean([pLKF_Z pLKF_L pLKF_D]);
mu_rnn                          = mean([pRNN_Z pRNN_L pRNN_D]);
er_lkf                          = std([pLKF_Z pLKF_L pLKF_D])./sqrt(NumTrials);
er_rnn                          = std([pRNN_Z pRNN_L pRNN_D])./sqrt(NumTrials);

fig = figure('position',[680   488   700   490]);clf; hold on;
errorbar(1 : 3, mu_lkf, er_lkf,'ro-','linewidth',2,'markersize',10); hold on;
errorbar(1 : 3, mu_rnn, er_rnn,'bo-','linewidth',2,'markersize',10)
set(gca,'xtick',1:3,'xticklabel', feature_labels,'fontsize',13); xlim([0.7 3.3]);
ylabel('E_P_O_S (cm)','fontsize',13);

%-- Statistical test: Friedman test
[p,tbl,stats]                   = friedman([[vLKF_Z vLKF_L vLKF_D];[vRNN_Z vRNN_L vRNN_D]],2);
C                               = multcompare(stats,'CType','bonferroni');

[p,tbl,stats]                   = friedman([[pLKF_Z pLKF_L pLKF_D];[pRNN_Z pRNN_L pRNN_D]],2);
C                               = multcompare(stats,'CType','bonferroni');
    

