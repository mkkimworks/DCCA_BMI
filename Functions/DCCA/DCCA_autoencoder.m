% (C) 2015 by Weiran Wang, Raman Arora, Karen Livescu and Jeff Bilmes
% Download from https://ttic.uchicago.edu/~wwang5/dccae.html
function [F1opt,F2opt,specification] = DCCA_autoencoder(TrX1, TrX2, TeX1, TeX2, h_param)
try
%% Hyperparameters for DCCAE network architecture.
dim_X1          = size(TrX1,2);
dim_X2          = size(TrX2,2);
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
maxepoch        = 1000;
% Pretraining is used to speedup autoencoder training.
pretrainnet     = '';
[F1opt,F2opt,specification]   = DCCAE_train( ...
                                            TrX1,TrX2,TeX1,TeX2,[],[],K,lambda,hiddentype,outputtype,...
                                            NN1,NN2,NN3,NN4,rcov1,rcov2,l2penalty, cca_batchsize,rec_batchsize,...
                                            eta0,decay,momentum,maxepoch,randseed,pretrainnet);
catch
    web('https://ttic.uchicago.edu/~wwang5/dccae.html')
end