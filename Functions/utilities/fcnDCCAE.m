function DCCAE_results                      = fcnDCCAE(XTrain, YTrain, TrialIND, netVar)

% keyboard
S                                           = segment_data(TrialIND, 0.9, 0); 

[F1opt,F2opt,specification]                 = DCCA_autoencoder(XTrain(S.trid,:), YTrain(S.trid,:),XTrain(S.teid,:),YTrain(S.teid,:), netVar);

% Testing the learned networks.
XV1proj                                     = gather(deepnetfwd(XTrain(S.teid,:),F1opt));
XV2proj                                     = gather(deepnetfwd(YTrain(S.teid,:),F2opt));

DCCAE_results.r2                            = diag(corr(double(XV1proj), double(XV2proj)));
DCCAE_results.valError                      = mean(1 - abs(DCCAE_results.r2));

DCCAE_results.NetView_A                     = F1opt;
DCCAE_results.NetView_B                     = F2opt;
DCCAE_results.specification                 = specification;