function [Z_all,Z_stack] = DrawDynamics(X, IND, Target)

TU          = unique(IND);
TargetIndex = unique(Target);
dim         = size(X,2);
NumTargets  = size(TargetIndex,1);

ZZ          = zeros(20, dim, NumTargets);
Zcnt        = zeros(NumTargets,1);
for i = 1 : length(TU)
    ind = find(IND == TU(i));
    d_ind = Target(i) == TargetIndex;
    ZZ(:,:,d_ind) = ZZ(:,:,d_ind) + pchip(linspace(0,1,size(X(ind,:),1)), X(ind,:)', linspace(0,1,20))';
    Zcnt(d_ind) = Zcnt(d_ind) + 1;
end
Z_all = []; Z_stack = [];
for d = 1 : NumTargets
    Z_all_ = ZZ(:,:,d) ./ Zcnt(d);
    Z_all = [Z_all ; Z_all_];
    Z_stack(:,:,d) = Z_all_;
end