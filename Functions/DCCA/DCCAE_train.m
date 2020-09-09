function [F1opt,F2opt,filename]=DCCAE_train( ...
    X1,X2,XV1,XV2,XTe1,XTe2,K,lambda,hiddentype,outputtype,...
    NN1,NN2,NN3,NN4,rcov1,rcov2,l2penalty, cca_batchsize,rec_batchsize,...
    eta0,decay,momentum,maxepoch,randseed,pretrainnet)
% [F1opt,F2opt,CORR_train,CORR_tune,CORR_test]=DCCAtrain_SGD( ...
%   X1,X2,XV1,XV2,XTe1,XTe2,K,lambda,hiddentype,outputtype,...
%   NN1,NN2,NN3,NN4,rcov1,rcov2,l2penalty,cca_batchsize,rec_batchsize,...
%   eta0,decay,momentum,maxepoch,randseed,pretrainnet) trains the DCCAE
%   model using stochastic gradient descent.
%
% Inputs
%   X1: training data for view 1, each row contains a data sample.
%   X2: training data for view 2, each row contains a data sample.
%   XV1: tuning data for view 1.
%   XV2: tuning data for view 2.
%   XTe1: optional testing data for view 1, can be empty.
%   XTe2: optional testing data for view 2, can be empty.
%   K: dimension of DCCA projection.
%   lambda: trade-off parameter, our objective is
%     -correlation+lambda*(view 1 reconstruction error + view 1
%     reconstruction error).
%   hiddentype: type of hidden units for the networks. Can be 'linear',
%     'sigmoid','tanh','relu','cubic'.
%   outputtype: type of output units for the networks. Same to hiddentype.
%   NN1: vector of hidden layer sizes for view 1 featur extraction network.
%   NN2: vector of hidden layer sizes for view 2 featur extraction network.
%   NN3: vector of hidden layer sizes for view 1 reconstruction network.
%   NN4: vector of hidden layer sizes for view 2 reconstruction network.
%   rcov1: regularization parameter for view 1.
%   rcov2: regularization parameter for view 2.
%   l2penalty: weight decay parameters for all weight parameters.
%   cca_batchsize: number of samples in a minibatch for estimating gradient
%     for the canonical correlation term.
%   rec_batchsize: number of samples in a minibatch for estimating gradient
%     for the reconstruction error term.
%   eta0: initial learning rate.
%   decay: the geometric rate in which the learning rate decays.
%   momentum: momentum parameter for SGD, with value in [0 1).
%   maxepoch: number of passes over the training set.
%   randseed: random seed for initializing weights and shuffling training
%     set at each epoch.
%   pretrainnet: path to the file containing pretrained networks.
%
% Outputs
%   F1opt: trained network for view 1, it is a cell array containing each
%     all layers of the network. Each layer has a field 'type' indicating
%     the type of hidden activation, a field 'units' indicating the output
%     dimension of the layer, a filed 'l' indicating the weight decay
%     parameter, and a field 'W' containing the weight matrix.
%   F2opt: trained network for view 2, same structure as F1opt.

if isempty(XTe1) || isempty(XTe2)
    CORR_test=[];
end
if ~exist('randseed','var') || isempty(randseed)
    randseed=0;
end
rng('default');
rng(randseed);

% mkdir("./DCCAE_results/" + mother_path + "/")
filename=['result_K=' num2str(K) '_lambda=' num2str(lambda) ...
    '_rcov1=' num2str(rcov1) '_rcov2=' num2str(rcov2) ...
    '_l2penalty=' num2str(l2penalty) ...
    '_recbatchsize=' num2str(rec_batchsize) ...
    '_ccabatchsize=' num2str(cca_batchsize) ...
    '_eta0=' num2str(eta0) '_decay=' num2str(decay) ...
    '_momentum=' num2str(momentum) '_maxepoch=' num2str(maxepoch)];

[N,D1]=size(X1); [~,D2]=size(X2);

if exist(filename,'file')
    load(filename,'randseed','its','steps','F1','F2','G1','G2','F1opt','F2opt', ...
        'TT','eta','delta','optvalid','CORR_train','CORR_tune','CORR_test','LE1','LE2','LE3','RE1','RE2','RE3','OBJ');
    if its>=maxepoch;
        fprintf('Neural networks have already been trained!\nExiting ...\n');
        return;
    else
        fprintf('Neural networks trained halfway!\nLoading ...\n');
    end
else
    
%     fprintf('Result will be saved in %s\n',filename);
    [N,D1]=size(X1); [~,D2]=size(X2);
    
    %% Set the architecture for each view in DCCA.
    Layersizes1=[D1 NN1];
    Layertypes1={};
    for nn1=1:length(NN1);
        Layertypes1=[Layertypes1, {hiddentype}];
    end
    if length(Layertypes1)>0 Layertypes1{end}='linear'; end
    F1=deepnetinit(Layersizes1,Layertypes1);
    
    Layersizes2=[D2 NN2];
    Layertypes2={};
    for nn2=1:length(NN2);
        Layertypes2=[Layertypes2, {hiddentype}];
    end
    if length(Layertypes2)>0 Layertypes2{end}='linear'; end
    F2=deepnetinit(Layersizes2,Layertypes2);
    
    %% First view decoder.
    Layersizes3=[NN1(end) NN3];
    Layertypes3={};
    for nn3=1:length(NN3);
        Layertypes3=[Layertypes3, {hiddentype}];
    end
    if length(Layertypes3)>0 Layertypes3{end}=outputtype; end
    G1=deepnetinit(Layersizes3,Layertypes3);
    
    %% Second view decoder.
    Layersizes4=[NN2(end) NN4];
    Layertypes4= {};
    for nn4=1:length(NN4);
        Layertypes4=[Layertypes4, {hiddentype}];
    end
    if length(Layertypes4)>0 Layertypes4{end}=outputtype; end
    G2=deepnetinit(Layersizes4,Layertypes4);
    
    clear Layertypes1
    clear Layertypes2
    clear Layertypes3
    clear Layertypes4
    
    %% Load pretrained networks if specified.
    if ~exist('pretrainnet','var') || isempty(pretrainnet)
    else
        load(pretrainnet,'F1','F2','G1','G2');
    end
    
    %% Compute correlations and errors.
    FX1=deepnetfwd(X1,F1);
    FX2=deepnetfwd(X2,F2);
    [A,B,m1,m2,D]=linCCA(FX1,FX2,K,rcov1,rcov2);  clear FX1 FX2;
    SIGN=sign(A(1,:)+eps);
    A=bsxfun(@times,A,SIGN);  B=bsxfun(@times,B,SIGN);
    f1.type='linear'; f1.units=K; f1.W=[A;-m1*A]; F1tmp=F1; F1tmp{end+1}=f1;
    f2.type='linear'; f2.units=K; f2.W=[B;-m2*B]; F2tmp=F2; F2tmp{end+1}=f2;
    clear A B m1 m2 D;
    
    X_train=deepnetfwd(X1,F1tmp);
    CORR_train=DCCA_corr(X_train,deepnetfwd(X2,F2tmp),K); clear X_train;
    X_tune=deepnetfwd(XV1,F1tmp);
    CORR_tune=DCCA_corr(X_tune,deepnetfwd(XV2,F2tmp),K); clear X_tune;
    if ~isempty(XTe1) && ~isempty(XTe2)
        X_test=deepnetfwd(XTe1,F1tmp);
        CORR_test=DCCA_corr(X_test,deepnetfwd(XTe2,F2tmp),K); clear X_test;
    else
        CORR_test=[];
    end
    F1opt=F1tmp; F2opt=F2tmp;
    clear F1tmp F2tmp;
    obj=-CORR_train(end);
    
    % Reconstruction RMSE.
    FG1tmp={F1{:} G1{:}};
    [~,etmp]=deepnetfwd(X1,FG1tmp,X1); e1=sqrt(etmp/size(X1,1)); obj=obj+lambda*etmp/N;
    [~,etmp]=deepnetfwd(XV1,FG1tmp,XV1); e2=sqrt(etmp/size(XV1,1));
    if ~isempty(XTe1) && ~isempty(XTe2)
        [~,etmp]=deepnetfwd(XTe1,FG1tmp,XTe1); e3=sqrt(etmp/size(XTe1,1));
    else
        e3=inf;
    end
    LE1=e1; LE2=e2; LE3=e3;
    clear FG1tmp etmp;
    
    % Reconstruction RMSE.
    FG2tmp={F2{:} G2{:}};
    [~,etmp]=deepnetfwd(X2,FG2tmp,X2); e1=sqrt(etmp/size(X2,1)); obj=obj+lambda*etmp/N;
    [~,etmp]=deepnetfwd(XV2,FG2tmp,XV2); e2=sqrt(etmp/size(XV2,1));
    if ~isempty(XTe1) && ~isempty(XTe2)
        [~,etmp]=deepnetfwd(XTe2,FG2tmp,XTe2); e3=sqrt(etmp/size(XTe2,1));
    else
        e3=inf;
    end
    
    RE1=e1; RE2=e2; RE3=e3;
    clear FG2tmp etmp;
    OBJ=obj; clear obj;
    
    % Intermediate result.
    its=0; TT=0; steps=0; delta=0; eta=eta0; optvalid=-CORR_tune(end)+lambda*(LE2(end)^2+RE2(end)^2);    
end

%% Concatenate the weights.
VV=[];
Nlayers=length(F1); net1=cell(1,Nlayers);
for k=1:Nlayers
    VV=[VV; F1{k}.W(:)];
    net1{k}=rmfield(F1{k},'W');
end
cut1=length(VV);
Nlayers=length(F2); net2=cell(1,Nlayers);
for k=1:Nlayers
    VV=[VV; F2{k}.W(:)];
    net2{k}=rmfield(F2{k},'W');
end
cut2=length(VV);
Nlayers=length(G1); net3=cell(1,Nlayers);
for k=1:Nlayers
    VV=[VV; G1{k}.W(:)];
    net3{k}=rmfield(G1{k},'W');
end
cut3=length(VV);
Nlayers=length(G2); net4=cell(1,Nlayers);
for k=1:Nlayers
    VV=[VV; G2{k}.W(:)];
    net4{k}=rmfield(G2{k},'W');
end
cut4=length(VV);

%% For the initialization, we still need to compute l2 weight decay.
if length(OBJ)==1
    OBJ=OBJ+l2penalty*sum(VV.^2);
end

%% Use GPU if equipped. GPU significantly speeds up optimization.
if gpuDeviceCount>0
%     fprintf('GPU detected. Trying to use it ...\n');
%     try
%         VV=gpuArray(VV);
%         X1=gpuArray(X1);
%         X2=gpuArray(X2);
%         fprintf('Using GPU ...\n');
%     catch
%     end
end
valid_lag = 0;
numbatches=ceil(N/rec_batchsize);
while its<maxepoch
    
    eta=eta0*decay^(its-1); % Reduce learning rate.
    t0=tic;
    % Randomly shuffle training set.
    rp=randperm(N);
    for i=1:numbatches
        
        % Weight decay.
        grad=2*l2penalty*VV;
        
        % Reconstruction gradient.
        if lambda>0
            idx1=(i-1)*rec_batchsize+1;
            idx2=min(i*rec_batchsize,N);
            idx=[rp(idx1:idx2),rp(1:max(0,i*rec_batchsize-N))];
            X1batch=X1(idx,:); X2batch=X2(idx,:);
            
            [~,grad2]=deepnetgrad(VV([1:cut1, cut2+1:cut3]), ...
                X1batch,X1batch,{net1{:},net3{:}});
            
            grad2=grad2/rec_batchsize;
            grad2=[grad2(1:cut1); zeros(cut2-cut1,1); grad2(cut1+1:end); zeros(cut4-cut3,1)];
            grad=grad + lambda*grad2;
            
            [~,grad2]=deepnetgrad(VV([cut1+1:cut2, cut3+1:cut4]), ...
                X2batch,X2batch,{net2{:},net4{:}});
            grad2=grad2/rec_batchsize;
            grad2=[zeros(cut1,1); grad2(1:cut2-cut1); zeros(cut3-cut2,1); grad2(cut2-cut1+1:end)];
            grad=grad + lambda*grad2;
        end
        
        DCCARP=randperm(N); DCCABATCHIDX=DCCARP(1:cca_batchsize);
        [~,grad1]=DCCA_grad(VV(1:cut2), ...
            X1(DCCABATCHIDX,:),X2(DCCABATCHIDX,:),net1,net2,K,rcov1,rcov2);
        grad1=[grad1; zeros(cut4-cut2,1)];
        grad=grad+grad1;
        
        delta=momentum*delta - eta*grad;  % With momentum.
        VV=VV + delta;
        
        steps=steps+1;
    end
    
    % Record the time spent for each epoch.
    its=its+1; TT=[TT, toc(t0)];
    
    %% Assemble the networks.
    idx=0;
    D=size(X1,2);
    for j=1:length(F1)
        if strcmp(F1{j}.type,'conv')
            convdin=F1{j}.filternumrows*F1{j}.filternumcols*F1{j}.numinputmaps;
            convdout=F1{j}.numoutputmaps;
            W_seg=VV(idx+1:idx+(convdin+1)*convdout);
            F1{j}.W=reshape(W_seg,convdin+1,convdout);
            idx=idx+(convdin+1)*convdout;
            D=F1{j}.units;
        else
            units=F1{j}.units;
            W_seg=VV(idx+1:idx+(D+1)*units);
            F1{j}.W=reshape(W_seg,D+1,units);
            idx=idx+(D+1)*units; D=units;
        end
    end
    
    D=size(X2,2);
    for j=1:length(F2)
        if strcmp(F2{j}.type,'conv')
            convdin=F2{j}.filternumrows*F2{j}.filternumcols*F2{j}.F2{j}.numinputmaps;
            convdout=F2{j}.numoutputmaps;
            W_seg=VV(idx+1:idx+(convdin+1)*convdout);
            F2{j}.W=reshape(W_seg,convdin+1,convdout);
            idx=idx+(convdin+1)*convdout;
            D=F2{j}.units;
        else
            units=F2{j}.units;
            W_seg=VV(idx+1:idx+(D+1)*units);
            F2{j}.W=reshape(W_seg,D+1,units);
            idx=idx+(D+1)*units; D=units;
        end
    end
    
    D=F1{end}.units;
    for j=1:length(G1)
        units=G1{j}.units;
        W_seg=VV(idx+1:idx+(D+1)*units);
        G1{j}.W=reshape(W_seg,D+1,units);
        idx=idx+(D+1)*units; D=units;
    end
    
    D=F2{end}.units;
    for j=1:length(G2)
        units=G2{j}.units;
        W_seg=VV(idx+1:idx+(D+1)*units);
        G2{j}.W=reshape(W_seg,D+1,units);
        idx=idx+(D+1)*units; D=units;
    end
    
    %% Compute correlations and errors.
    FX1=deepnetfwd(X1,F1);  FX2=deepnetfwd(X2,F2);
    [A,B,m1,m2,D]=linCCA(FX1,FX2,K,rcov1,rcov2); clear FX1 FX2;
    SIGN=sign(A(1,:)+eps);  A=bsxfun(@times,A,SIGN);  B=bsxfun(@times,B,SIGN);
    f1.type='linear'; f1.units=K; f1.W=[A;-m1*A]; F1tmp=F1; F1tmp{end+1}=f1;
    f2.type='linear'; f2.units=K; f2.W=[B;-m2*B]; F2tmp=F2; F2tmp{end+1}=f2;
    clear A B m1 m2 D;
    
    X_train=deepnetfwd(X1,F1tmp);
    CORR_train=[CORR_train, DCCA_corr(X_train,deepnetfwd(X2,F2tmp),K)]; clear X_train;
    X_tune=deepnetfwd(XV1,F1tmp);
    CORR_tune=[CORR_tune, DCCA_corr(X_tune,deepnetfwd(XV2,F2tmp),K)]; clear X_tune;
    if ~isempty(XTe1) && ~isempty(XTe2)
        X_test=deepnetfwd(XTe1,F1tmp);
        CORR_test=[CORR_test, DCCA_corr(X_test,deepnetfwd(XTe2,F2tmp),K)]; clear X_test;
    end
    obj=-CORR_train(end);
    
    % Reconstruction RMSE.
    FG1tmp={F1{:} G1{:}};
    [~,etmp]=deepnetfwd(X1,FG1tmp,X1); e1=sqrt(etmp/size(X1,1));
    obj=obj+lambda*etmp/N;
    [~,etmp]=deepnetfwd(XV1,FG1tmp,XV1); e2=sqrt(etmp/size(XV1,1));
    if ~isempty(XTe1) && ~isempty(XTe2)
        [~,etmp]=deepnetfwd(XTe1,FG1tmp,XTe1); e3=sqrt(etmp/size(XTe1,1));
    else
        e3=inf;
    end
    LE1=[LE1, e1]; LE2=[LE2, e2]; LE3=[LE3, e3];
    
    % Reconstruction RMSE.
    FG2tmp={F2{:} G2{:}};
    [~,etmp]=deepnetfwd(X2,FG2tmp,X2); e1=sqrt(etmp/size(X2,1));
    obj=obj+lambda*etmp/N;
    [~,etmp]=deepnetfwd(XV2,FG2tmp,XV2); e2=sqrt(etmp/size(XV2,1));
    if ~isempty(XTe1) && ~isempty(XTe2)
        [~,etmp]=deepnetfwd(XTe2,FG2tmp,XTe2); e3=sqrt(etmp/size(XTe2,1));
    else
        e3=inf;
    end
    RE1=[RE1, e1]; RE2=[RE2, e2]; RE3=[RE3, e3];
    
    %% Compute objective.
    obj=obj+l2penalty*sum(VV.^2);
    OBJ=[OBJ, obj];
    %% Check learning progress.
    tmpvalid=-CORR_tune(end)+lambda*(LE2(end)^2+RE2(end)^2);
    if tmpvalid < optvalid
        optvalid=tmpvalid;
        fprintf('DCCAE epoch %d: Objective improved on tuning set!\n',its);
        F1opt=F1tmp;
        F2opt=F2tmp;

        valid_lag = 0;
    else
        valid_lag = valid_lag + 1;
    end
    if valid_lag == 10
        disp('Breaked. ');
        break;
    end
    clear F1tmp F2tmp FG1tmp FG2tmp etmp
end
% save(filename,'randseed','its','steps','F1','F2','G1','G2','F1opt','F2opt', ...
%         'TT','eta','delta','optvalid','CORR_train','CORR_tune','CORR_test',...
%         'LE1','LE2','LE3','RE1','RE2','RE3','OBJ','its');