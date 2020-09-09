function Data                   = CrossValid(IND, Z, X, Target, Prc)

try
S                                = segment_data(IND, Prc, 1);


Data.NumTeTrials                 = S.tesz;
Data.NumTrTrials                 = S.trsz;
Data.TrI                         = IND(S.trid);
Data.TeI                         = IND(S.teid);
Data.TrZ                         = Z(S.trid,:);
Data.TeZ                         = Z(S.teid,:);
Data.TrX                         = X(S.trid,:);
Data.TeX                         = X(S.teid,:);
if isstruct(Target)
    Target                       = Target.deg;
end
Data.TrG                         = Target(1:S.trsz);
Data.TeG                         = Target(1+S.trsz:end);
Data.dim                         = size(X,2);
Data.TargetIndex                 = unique(Target);
Data.NumTargets                  = size(Data.TargetIndex,1);
Data.dimZ                        = size(Data.TrZ,2);
catch
    keyboard
end