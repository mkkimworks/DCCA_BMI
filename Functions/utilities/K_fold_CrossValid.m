function Data                        = K_fold_CrossValid(IND, Z, X, Target, K, k)

try
    k_folds                          = K;
    u_ind                            = unique(IND);
    
    new_IND                          = [];
    for i = 1 : length(u_ind)
        len                          = sum(u_ind(i) == IND);
        new_IND                      = [new_IND ; ones(len,1) * i];
    end
    IND                              = new_IND;
    u_ind                            = unique(IND);
    block                            = floor(length(u_ind)/k_folds);
    
    te_ind                           = u_ind((k-1)*block+1:block*k);
    tr_ind                           = u_ind;
    tr_ind(te_ind)                   = [];
    
    if isstruct(Target)
        Target                       = Target.deg;
    end
        
    TeI                              = [];
    TeZ                              = [];
    TeX                              = [];
    TeG                              = [];
    for i = 1 : length(te_ind)
        ind_                         = te_ind(i) == IND;
        TeI                          = [TeI ; IND(ind_)];
        TeZ                          = [TeZ ; Z(ind_,:)];
        TeX                          = [TeX ; X(ind_,:)];
        if ~isempty(X(ind_,:))
            TeG                      = [TeG ; Target(te_ind(i))];
        end
    end
    
    TrI                              = [];
    TrZ                              = [];
    TrX                              = [];
    TrG                              = [];
    for i = 1 : length(tr_ind)
        ind_                         = tr_ind(i) == IND;
        TrI                          = [TrI ; IND(ind_)];
        TrZ                          = [TrZ ; Z(ind_,:)];
        TrX                          = [TrX ; X(ind_,:)];
        if ~isempty(X(ind_,:))
            TrG                      = [TrG ; Target(tr_ind(i))];
        end
    end
    
    
    Data.NumTeTrials                 = size(te_ind,1);
    Data.NumTrTrials                 = size(tr_ind,1);
    Data.TrI                         = TrI;
    Data.TeI                         = TeI;
    Data.TrZ                         = TrZ;
    Data.TeZ                         = TeZ;
    Data.TrX                         = TrX;
    Data.TeX                         = TeX;
    
    Data.TrG                         = TrG;
    Data.TeG                         = TeG;
    Data.dim                         = size(X,2);
    Data.TargetIndex                 = unique(Target);
    Data.NumTargets                  = size(Data.TargetIndex,1);
    Data.dimZ                        = size(Data.TrZ,2);
    
catch
    keyboard
end