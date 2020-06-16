function [savedStruct,BayesObject]          = LSTM_Optimization(X1, X2, NumRepeats, sub_name)
opt.NetworkNodes                            = {'16','32','64','128','256','512'};
opt.NetworkLayer                            = {'1','2','3','4'};
opt.batch_sz                                = {'32','64','128','256'};
opt.learning_rate                           = {'0.00001','0.0001','0.001','0.01'};
opt.l2penalty                               = [1e-6 1e-1];

optimVars                                   = [  optimizableVariable('NetworkNodes',    opt.NetworkNodes,'Type','categorical')
                                                 optimizableVariable('NetworkLayer',    opt.NetworkLayer,'Type','categorical')
                                                 optimizableVariable('batch_sz',        opt.batch_sz,'Type','categorical')
                                                 optimizableVariable('learning_rate',   opt.learning_rate,'Type','categorical')
                                                 optimizableVariable('l2penalty',       opt.l2penalty,'Type','real')
                                              ];
opt.sub_name                                = sub_name;
ObjFcn                                      = LSTM_ObjFcn(X1, X2, opt);
% ObjFcn                                      = makeObjFcn(Xin,Yin,vXin,vYin,opt);
try
BayesObject                                 = bayesopt(ObjFcn,optimVars,...
                                                        'MaxObj',NumRepeats,...
                                                        'UseParallel',false);

bestIdx                                     = BayesObject.IndexOfMinimumTrace(end);
fileName                                    = BayesObject.UserDataTrace{bestIdx};
savedStruct                                 = load(fileName);
catch
    keyboard
end
end

function ObjFcn                           	= LSTM_ObjFcn(XTrain,YTrain,opt)
ObjFcn                                      = @valErrorFun;
    function [valError,cons,fileName]       = valErrorFun(optVars)
        N                                           = size(XTrain,1);
        Val_len                                     = N - floor(N * 0.85) + 1;
        TrIND                                       = 1 : N;
        TeIND                                       = randperm(Val_len);
        TrIND(TeIND)                                = [];
        
        h_param.NV                                  = round(str2num(opt.NetworkNodes{optVars.NetworkNodes}));
        h_param.NL                                  = round(str2num(opt.NetworkLayer{optVars.NetworkLayer}));
        h_param.batch_sz                            = round(str2num(opt.batch_sz{optVars.batch_sz}));
        h_param.learning_rate                       = str2num(opt.learning_rate{optVars.learning_rate});
        h_param.l2penalty                           = optVars.l2penalty;
        
        layers                              = [
                                                sequenceInputLayer(size(XTrain,2))
                                                makeLayers(h_param.NV, h_param.NL)
                                                fullyConnectedLayer(size(YTrain,2))
                                                regressionLayer];
        XValidation                         = XTrain(TeIND,:);
        YValidation                         = YTrain(TeIND,:);
        validationFrequency                 = floor(size(YTrain,1)/h_param.batch_sz);
        options                             = trainingOptions(  'adam',...
                                                                'ExecutionEnvironment','gpu', ...
                                                                'GradientDecayFactor',0.95,...
                                                                'SquaredGradientDecayFactor',0.99,...
                                                                'InitialLearnRate', h_param.learning_rate,...
                                                                'MaxEpochs',1000, ...
                                                                'LearnRateSchedule','none',...
                                                                'MiniBatchSize',h_param.batch_sz,...
                                                                'L2Regularization',h_param.l2penalty,...
                                                                'Shuffle','every-epoch',...
                                                                'Plots','training-progress',...%%
                                                                'ValidationData',{XValidation',YValidation'},...
                                                                'ValidationPatience',Inf,...
                                                                'ValidationFrequency',validationFrequency,...
                                                                'OutputFcn',@(info)stopIfAccuracyNotImproving(info,10));
        
        
        
        trainedNet                                  = trainNetwork(XTrain(TrIND,:)',YTrain(TrIND,:)',layers,options); 
        close(findall(groot,'Tag','NNET_CNN_TRAININGPLOT_FIGURE'))
        
        YPredicted                                  = predict(trainedNet, XValidation'); 
        valError                                    = mean(sqrt(sum((YPredicted(:,~isnan(sum(YValidation,2)))' - YValidation(~isnan(sum(YValidation,2)),:)).^2,2)));
        
        options                                     = h_param;
        
        mkdir(sprintf('%s/LSTM/BayesianOpt/%s',pwd, opt.sub_name));
        fileName =sprintf("%s/LSTM/BayesianOpt/%s/",pwd, opt.sub_name) + num2str(valError) + ".mat";
        save(fileName,'trainedNet','valError','options')
        cons = [];
        
    end
end


function stop = stopIfAccuracyNotImproving(info,N)
stop = false;

% Keep track of the best validation accuracy and the number of validations for which
% there has not been an improvement of the accuracy.
persistent bestValAccuracy
persistent valLag
persistent smoothed_rmse

% Clear the variables when training starts.
if info.State == "start"
    bestValAccuracy = 100;
    valLag = 0;
    smoothed_rmse = 100;
elseif ~isempty(info.ValidationRMSE)
    smoothed_rmse = smoothed_rmse * 0.5 + info.ValidationRMSE * 0.5;
    if smoothed_rmse < bestValAccuracy
        valLag = 0;
        bestValAccuracy = smoothed_rmse;
    else
        valLag = valLag + 1;
    end
    
    % If the validation lag is at least N, that is, the validation accuracy
    % has not improved for at least N validations, then return true and
    % stop training.
    if valLag >= N
        stop = true;
    end
    
end

end

function layers = makeLayers(numNodes,numLayers)
layers = [];
for i = 1 : numLayers
    layers                              = [layers ; lstmLayer(numNodes,'OutputMode','sequence')];
end
end