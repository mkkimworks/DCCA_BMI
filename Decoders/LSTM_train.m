function trainedNet                         = LSTM_trainer(XTrain, YTrain, opt)
[N,~]                                       = size(XTrain);
tN                                          = floor(N * 0.85);

Xin                                         = XTrain(1:tN,:);
Yin                                         = YTrain(1:tN,:);
vXin                                        = XTrain(tN+1:end,:);
vYin                                        = YTrain(tN+1:end,:);
numnode                                     = opt.node; 
miniBatchSize                               = opt.batch; 
Learning_rate                               = opt.lr;%0.01 
L2Regularization                            = opt.l2r; %1e-5;
layers                              = [
    sequenceInputLayer(size(XTrain,2))
    makeLayers(numnode, 1)
    fullyConnectedLayer(size(YTrain,2))
    regressionLayer];


validationFrequency                 = floor(size(YTrain,1)/miniBatchSize);
options                             = trainingOptions(  'adam',...
                                                        'ExecutionEnvironment','gpu', ...
                                                        'GradientDecayFactor',0.95,...
                                                        'SquaredGradientDecayFactor',0.99,...
                                                        'InitialLearnRate',Learning_rate,...
                                                        'MaxEpochs',1000, ...
                                                        'LearnRateSchedule','piecewise',...
                                                        'LearnRateDropPeriod',100,...
                                                        'LearnRateDropFactor',0.1,...
                                                        'MiniBatchSize',miniBatchSize,...
                                                        'L2Regularization',L2Regularization,...
                                                        'Shuffle','every-epoch',...
                                                        'Plots','training-progress',...%%
                                                        'ValidationData',{vXin',vYin'},...
                                                        'ValidationPatience',Inf,...
                                                        'ValidationFrequency',validationFrequency,...
                                                        'OutputFcn',@(info)stopIfAccuracyNotImproving(info,10));



trainedNet                          = trainNetwork(Xin',Yin',layers,options);
close(findall(groot,'Tag','NNET_CNN_TRAININGPLOT_FIGURE'))
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