function param              = fcnLSTM(X, Y, TrialIND, SubID, net_param)

if ~isstruct(net_param)
    NumRepeat               = 200;
    [Net,~]                 = LSTM_Optimization(X, Y, TrialIND, NumRepeat, SubID);
    net_param               = Net.net_param;
    FcnDecoder              = Net.LSTM_results.Net;
    if exist(sprintf('./LSTM_results/%s',SubID), 'file') == 0
        mkdir(sprintf('./LSTM_results/%s',SubID));
    end
    save(sprintf("./LSTM_results/%s/",SubID) + "OptimalLSTM_Param.mat",'FcnDecoder','net_param');
elseif isstruct(net_param)
    Net                     = fcn_trainLSTM(X, Y, TrialIND, net_param);
    FcnDecoder              = Net.Net;
    TrainingData_Index      = unique(TrialIND);
    param.LSTM_Model            = FcnDecoder;
    param.TrainingData_Index    = TrainingData_Index;
    param.fcn                   = @(cY)double(gather(predict(FcnDecoder, double(cY'))))';

else
    if exist(sprintf("./LSTM_results/%s/",SubID) + "OptimalLSTM_Param.mat",'file') ~= 0
        lstm_param        	= load(sprintf("./LSTM_results/%s/",SubID) + "OptimalLSTM_Param.mat");
        FcnDecoder          = lstm_param.FcnDecoder;
    else
        error('Require LSTM model parameters.');
    end
end

