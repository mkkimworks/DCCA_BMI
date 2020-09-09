function map = tuning_map(Z, X, edge)

Z = (Z - mean(Z)) ./ std(Z);
VX = edge{1};
VY = edge{2};

map = nan(length(VX)-1, length(VY)-1);
for vx = 1 : length(VX)-1
    indx = X(:,1) >= VX(vx) & X(:,1) < VX(vx+1);
    for vy = 1 : length(VY)-1
        indy = X(:,2) >= VY(vy) & X(:,2) < VY(vy+1);
        ind = indx & indy;
        map(vx, vy) = nanmean(Z(ind));
    end
end