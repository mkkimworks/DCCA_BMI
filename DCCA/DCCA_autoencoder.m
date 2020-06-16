function [F1opt,F2opt,train_file] = DCCA_autoencoder(X1, X2, subname, h_param)


%% Hyperparameters for DCCAE network architecture.
dim_X1          = size(X1,2);
dim_X2          = size(X2,2);
K               = min([dim_X1 dim_X2]);
% Regularizations for each view.
rcov1           = h_param.rcov1;%1e-4; 
rcov2           = h_param.rcov2;%1e-4;
% Hidden activation type.
hiddentype      = 'sigmoid';
outputtype      = 'sigmoid';
% Architecture for view 1 feature extraction network.
NN1             = [h_param.NV1 * ones(1, h_param.NL1) K];
% Architecture for view 2 feature extraction network.
NN2             = [h_param.NV2 * ones(1, h_param.NL2) K];
% Architecture for view 1 reconstruction network.
NN3             = [h_param.NV1 * ones(1, h_param.NL1) dim_X1];
% Architecture for view 2 reconstruction network.
NN4             = [h_param.NV2 * ones(1, h_param.NL2) dim_X2];
% Weight decay parameter.
l2penalty       = h_param.l2penalty;%1e-4;

randseed        = 8409;

N               = size(X1,1);
N_              = floor(N * .85)+1;
XV1             = X1(N_:end,:);
XV2             = X2(N_:end,:);
%% Run DCCAE with SGD.
% Reconstruction error term weight.
lambda          = h_param.lambda;%0.001;
% Minibatchsize for the correlation term.
cca_batchsize   = h_param.cca_batch_sz;%800;
% Minibatchsize for reconstruction error term.
rec_batchsize   = h_param.rec_batch_sz;%100;
% Learning rate.
eta0            = h_param.learning_rate;%0.01;
% Rate in which learning rate decays over iterations.
% 1 means constant learning rate.
decay           = 1;
% Momentum.
momentum        = 0.99;
% How many passes of the data you run SGD with.
maxepoch        = 500;
% Pretraining is used to speedup autoencoder training.
pretrainnet     = '';
[F1opt,F2opt,train_file]   = DCCAE_train( ...
                                            X1,X2,XV1,XV2,[],[],K,lambda,hiddentype,outputtype,...
                                            NN1,NN2,NN3,NN4,rcov1,rcov2,l2penalty, cca_batchsize,rec_batchsize,...
                                            eta0,decay,momentum,maxepoch,randseed,pretrainnet,subname);

% if isempty(pretrainnet)
%     dat         = load(train_file);
%     
%     F1          = dat.F1;
%     F2          = dat.F2;
%     G1          = dat.G1;
%     G2          = dat.G2;
%     save(sprintf('RBM_PRETRAIN_%s',subname), 'F1','F2','G1','G2', 'dat');
% end
% Testing the learned networks.
% X1proj=gather(deepnetfwd(X1,F1opt));
% XV1proj=gather(deepnetfwd(XV1,F1opt));
% XTe1proj=gather(deepnetfwd(XTe1,F1opt));