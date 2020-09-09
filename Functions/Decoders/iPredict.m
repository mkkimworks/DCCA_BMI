function [Xhat, PRM] = iPredict(Z, Predict, PRM, ID)
try
if nargin < 3
    ID                  = 1 : size(Z,1);
end
UI                      = unique(ID);
dim                     = PRM.dim;
Xhat                    = [];
for k = 1 : length(UI)
    idx                 = UI(k) == ID;
    zin                 = Z(idx,:);
    hat                 = nan(sum(idx),dim);
    for t = PRM.tab : sum(idx)
        [hat(t,:), PRM] = Predict(zin(t : -1 : t-PRM.tab+1,:), PRM);
    end
    Xhat                = [Xhat ; hat];
    if strcmp(PRM.decoder, 'lkf')
        PRM             = InitKF(PRM);
    end
end
catch
    keyboard
end