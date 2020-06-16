function output                             = DeepCCA(X1, X2)

len                                         = floor(size(X1,1) * 0.7);
ind                                         = randperm(size(X1,1));
trInd                                       = ind(1:len);
vaInd                                       = ind(len+1:end);

Xin                                         = X1(trInd,:);
Yin                                         = X2(trInd,:);
XV1                                         = X1(vaInd,:);
XV2                                         = X2(vaInd,:);


K                                           = min([size(X2,2) size(X1,2)]);
%% Hyperparameters for DCCA network architecture.
% Regularizations for each view.
% rcov1=1e-4; rcov2=1e-4;
rcov1=1e-6; rcov2=1e-6;
% Hidden activation type.
hiddentype='sigmoid';
% Architecture (hidden layer sizes) for view 1 neural network.
NN1=[64 64 K];
% Architecture (hidden layer sizes)  for view 2 neural network.
NN2=[64 64 K];
% Weight decay parameter.
l2penalty=1e-8;



%% Run DCCA with SGD. No pretraining is used.
% Minibatchsize.
batchsize=size(X1,1);
% Learning rate.
eta0=0.001;
% Rate in which learning rate decays over iterations.
% 1 means constant learning rate.
decay=0.95;
% Momentum.
momentum=0.99;
% How many passes of the data you run SGD with.
maxepoch=1000;

[F1opt,F2opt]                               = DCCA_train(Xin,Yin,XV1,XV2,[],[],K,hiddentype,NN1,NN2, rcov1,rcov2,l2penalty,batchsize,eta0,decay,momentum,maxepoch);
% Testing the learned networks.
% X1proj                                      = gather(deepnetfwd(X1,F1opt)); 
% XV1proj                                     = gather(deepnetfwd(XV1,F1opt));
output.F1Opt                                = F1opt;
output.F2Opt                                = F2opt;

% r2                                          = zeros(1,size(X1proj,2));
% pv                                          = zeros(1,size(X1proj,2));
% for c = 1 : size(X1proj,2)
%     [~,~,~,~,stats]                         = regress(XV1proj(:,c), [ones(size(XV2,1),1) XV2]);
%     r2(c)                                   = stats(1);
%     pv(c)                                   = stats(end-1);
% end
% mean(r2)

