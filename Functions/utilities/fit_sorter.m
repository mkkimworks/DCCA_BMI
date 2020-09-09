function [ind, beta, r2, p]           = fit_sorter(Z, X)

[N, zdim] = size(Z);
[~, xdim] = size(X);

stat                            = zeros(4, zdim);
beta                            = zeros(xdim+1, zdim);
for k = 1 : zdim
    [beta(:,k), ~,~,~,stat(:,k)]= regress(Z(:,k), [ones(N,1) X]);
end
[~, ind]                        = sortrows(stat',1);
ind                             = flipud(ind);
stat                            = stat(:,ind);
r2                              = stat(1,:);
p                               = stat(3,:);
beta                            = beta(:,ind);