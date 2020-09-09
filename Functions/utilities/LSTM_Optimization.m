function [savedStruct,BayesObject]          = LSTM_Optimization(X, Y, TrialIND, NumRepeat, sub_name)

opt.NetworkNodes                            = {'15','30','35','40'};
opt.batch_sz                                = {'16','32','64','128'};
opt.learning_rate                           = {'0.0001','0.001','0.01'};
opt.l2penalty                               = [1e-6 1e-1];



optimVars                                   = [  optimizableVariable('NetworkNodes',    opt.NetworkNodes,'Type','categorical')
                                                 optimizableVariable('batch_sz',        opt.batch_sz,'Type','categorical')
                                                 optimizableVariable('learning_rate',   opt.learning_rate,'Type','categorical')
                                                 optimizableVariable('l2penalty',       opt.l2penalty,'Type','real')
                                              ];
opt.sub_name                                = sub_name;
ObjFcn                                      = LSTM_ObjFcn(X, Y, TrialIND, opt);
try
BayesObject                                 = bayesopt(ObjFcn,optimVars,...
                                                        'MaxObj',NumRepeat,...
                                                        'UseParallel',false);

bestIdx                                     = BayesObject.IndexOfMinimumTrace(end);
fileName                                    = BayesObject.UserDataTrace{bestIdx};
savedStruct                                 = load(fileName);
catch
    keyboard
end
end

function ObjFcn                           	= LSTM_ObjFcn(XTrain,YTrain, TrialIND, opt)
ObjFcn                                      = @valErrorFun;
    function [valError,cons,fileName]       = valErrorFun(optVars)
        net_param.NV                                = round(str2num(opt.NetworkNodes{optVars.NetworkNodes}));
        net_param.NL                                = 1;
        net_param.batch_sz                          = round(str2num(opt.batch_sz{optVars.batch_sz}));
        net_param.learning_rate                     = str2num(opt.learning_rate{optVars.learning_rate});
        net_param.l2penalty                         = optVars.l2penalty;
        LSTM_results                                = fcn_trainLSTM(XTrain, YTrain, TrialIND, net_param);
        valError                                    = LSTM_results.valError;
        if exist(sprintf('./LSTM_results/%s',opt.sub_name), 'file') == 0
            mkdir(sprintf('./LSTM_results/%s',opt.sub_name));
        end
        
        fileName =sprintf("./LSTM_results/%s/",opt.sub_name) + "trained_error_(" + num2str(valError) + ").mat";
        save(fileName,'LSTM_results','net_param')
        cons = [];
    end
end


