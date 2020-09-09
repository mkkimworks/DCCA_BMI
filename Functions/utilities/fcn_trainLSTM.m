function LSTM_results                      = fcn_trainLSTM(XTrain, YTrain, TrialIND, netVar)

S                                           = segment_data(TrialIND, 0.8, 0); 

InputSize                                   = size(XTrain,2);
OutputSize                                  = size(YTrain,2);
N                                           = size(YTrain,1);
layers                                      = [ sequenceInputLayer(InputSize)
                                                makeLayers(netVar.NV, netVar.NL)
                                                fullyConnectedLayer(OutputSize)
                                                regressionLayer];
XValidation                                 = XTrain(S.teid,:);
YValidation                                 = YTrain(S.teid,:);
validationFrequency                         = floor(N/netVar.batch_sz);
options                                     = trainingOptions(  'adam',...
                                                                'ExecutionEnvironment','gpu', ...
                                                                'GradientDecayFactor',0.95,...
                                                                'SquaredGradientDecayFactor',0.99,...
                                                                'InitialLearnRate', netVar.learning_rate,...
                                                                'MaxEpochs',10000, ...
                                                                'LearnRateSchedule','none',...
                                                                'MiniBatchSize',netVar.batch_sz,...
                                                                'L2Regularization',netVar.l2penalty,...
                                                                'Shuffle','every-epoch',...
                                                                'Plots','none',...%%
                                                                'ValidationData',{XValidation',YValidation'},...
                                                                'ValidationPatience',Inf,...
                                                                'ValidationFrequency',validationFrequency,...
                                                                'OutputFcn',@(info)stopIfAccuracyNotImproving(info,15));



trainedNet                                  = trainNetwork(XTrain(S.trid,:)', YTrain(S.trid,:)',layers,options);
close(findall(groot,'Tag','NNET_CNN_TRAININGPLOT_FIGURE'))

% Testing the learned networks.
YPredicted                                  = predict(trainedNet, XValidation')';
nan_ind                                     = ~isnan(sum(YValidation,2));
hat                                         = double(YPredicted(nan_ind,1:2));
True                                        = double(YValidation(nan_ind,1:2));
valError                                    = mae(True, hat);
LSTM_results.valError                       = valError;
LSTM_results.Net                            = trainedNet;
LSTM_results.specification                  = options;
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
    layers                              = [layers ; lstmLayer(numNodes,'OutputMode','sequence');dropoutLayer(0.2)];
end
end