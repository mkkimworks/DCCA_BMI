%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   
%   Title: Decoding kinematic information from primary motor cortical 
%          ensemble activity using a deep canonical correlation analysis
%   Authors: M-K Kim; J-W Sohn; S-P Kim
%   E-mail: mkkim.works@gmail.com
%   Affiliation: Ulsan National Institute of Science and Technology (UNIST)
%
%   [savedStruct,BayesObject]          = DCCA_Optimization(X1, X2, sub_name)
%   Inputs 
%       X1: random variable of view 1
%       X2: random variable of view 2
%       NumRepeat: number of optimization repetitions
%       sub_name: path to save DCCA paramters
%   Outputs
%       saveStruct: optimized trained-DCCA parameters
%       BayesObject: Bayesian optimization output
%
%   DCCA Toolbox is available from https://ttic.uchicago.edu/~wwang5/dccae.html
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [savedStruct,BayesObject]          = DCCA_Optimization(X1, X2, NumRepeat, sub_name)

opt.NetworkNodes                            = {'16','32','64','128'};
opt.NetworkLayer                            = {'2','3','4','5'};
opt.rcov                                    = [1e-6 1e-1];
opt.lambda                                  = [1e-6 1e-1];
opt.cca_batch_sz                            = {'32','64','128','256'};
opt.rec_batch_sz                            = {'32','64','128','256'};
opt.learning_rate                           = {'0.00001','0.0001','0.001','0.01'};
opt.l2penalty                               = [1e-6 1e-1];

optimVars                                   = [  optimizableVariable('NetworkNodes_X',  opt.NetworkNodes,'Type','categorical')
                                                 optimizableVariable('NetworkNodes_Y',  opt.NetworkNodes,'Type','categorical')
                                                 optimizableVariable('NetworkLayer_X',  opt.NetworkLayer,'Type','categorical')
                                                 optimizableVariable('NetworkLayer_Y',  opt.NetworkLayer,'Type','categorical')
                                                 optimizableVariable('rcov1',           opt.rcov,'Type','real')
                                                 optimizableVariable('rcov2',           opt.rcov,'Type','real')
                                                 optimizableVariable('lambda',          opt.lambda,'Type','real')
                                                 optimizableVariable('cca_batch_sz',    opt.cca_batch_sz,'Type','categorical')
                                                 optimizableVariable('rec_batch_sz',    opt.rec_batch_sz,'Type','categorical')
                                                 optimizableVariable('learning_rate',   opt.learning_rate,'Type','categorical')
                                                 optimizableVariable('l2penalty',       opt.l2penalty,'Type','real')
                                              ];
opt.sub_name                                = sub_name;
ObjFcn                                      = GCCA_ObjFcn(X1, X2, opt);
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

function ObjFcn = GCCA_ObjFcn(XTrain,YTrain,opt)
ObjFcn = @valErrorFun;
    function [valError,cons,fileName] = valErrorFun(optVars)
        try
            N                                           = size(XTrain,1);
            Val_len                                     = N - floor(N * 0.85) + 1;
            TrIND                                       = 1 : N;
            TeIND                                       = randperm(Val_len);
            TrIND(TeIND)                                = [];

            h_param.NV1                                 = round(str2num(opt.NetworkNodes{optVars.NetworkNodes_X}));
            h_param.NV2                                 = round(str2num(opt.NetworkNodes{optVars.NetworkNodes_Y}));
            h_param.NL1                                 = round(str2num(opt.NetworkLayer{optVars.NetworkLayer_X}));
            h_param.NL2                                 = round(str2num(opt.NetworkLayer{optVars.NetworkLayer_Y}));
            h_param.rcov1                               = optVars.rcov1;
            h_param.rcov2                               = optVars.rcov2;
            h_param.lambda                              = optVars.lambda;
            h_param.cca_batch_sz                        = round(str2num(opt.cca_batch_sz{optVars.cca_batch_sz}));
            h_param.rec_batch_sz                        = round(str2num(opt.rec_batch_sz{optVars.rec_batch_sz}));
            h_param.learning_rate                       = str2num(opt.learning_rate{optVars.learning_rate});
            h_param.l2penalty                           = optVars.l2penalty;
            [F1opt,F2opt,filepath]                      = DCCA_autoencoder(XTrain(TrIND,:), YTrain(TrIND,:), opt.sub_name, h_param);
            
            % Testing the learned networks.
            XV1proj                                     = gather(deepnetfwd(XTrain(TeIND,:),F1opt));
            XV2proj                                     = gather(deepnetfwd(YTrain(TeIND,:),F2opt));
            
            r2                                          = diag(corr(double(XV1proj), double(XV2proj)));
            valError                                    = mean(1 - abs(r2));

            trainedNet                                  = {F1opt, F2opt};            
            options                                     = h_param;
            options.Pretrain_Param                      = filepath;
            
            mkdir(sprintf('./BayesianOpt/%s',opt.sub_name));
            fileName =sprintf("./BayesianOpt/%s/",opt.sub_name) + num2str(valError) + ".mat";
            save(fileName,'trainedNet','valError','options')
            cons = [];
        catch
            keyboard
        end
        
    end
end

