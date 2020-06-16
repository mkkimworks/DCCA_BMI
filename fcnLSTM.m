function lstm_param         = fcnLSTM(X,Y, NumRepeats, lstm_param)

if nargin < 4
    [Net,~]                 = LSTM_Optimization(X, Y, NumRepeats, 'MonkeyF');
    lstm_param              = Net.trainedNet;
end
