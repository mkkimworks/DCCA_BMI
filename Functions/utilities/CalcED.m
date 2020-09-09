function [mED, ED] = CalcED(vhat, vtrue)
nan_id = ~isnan(sum(vtrue,2));
ED = sqrt(sum((vtrue(nan_id,:) - vhat(nan_id,:)).^2,2));
mED= nanmean(ED);