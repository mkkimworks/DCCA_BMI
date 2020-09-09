function [pos_mae,vel_mae,vel_CC, vel_traj, pos_traj, t_cnt] = fcn_decoded_by_lds(Dat, lds_prm, t_cnt)

trKF_pc                 = kalmanSmoother(lds_prm, Dat.TrZ')';
teKF_pc                 = kalmanSmoother(lds_prm, Dat.TeZ')';
SB                      = [ones(size(trKF_pc,1),1) trKF_pc]';
Ls_pc                   = Dat.TrX' * SB'*(SB*SB')^-1;

Z_vel                   = [ones(size(teKF_pc,1),1) teKF_pc] * Ls_pc';

TargetIndex             = unique([Dat.TrG;Dat.TeG]);
fix_len                 = 15;
TU                      = unique(Dat.TeI);
pos_mae_                = zeros(size(TU,1),1);
for t = 1 : size(TU,1)
    ind                 = TU(t) == Dat.TeI;
    d_ind               = TargetIndex == Dat.TeG(t);
    t_cnt(d_ind)        = t_cnt(d_ind) + 1;
    
    vel_mae(t,:)        = mae(Z_vel(ind,1:2), Dat.TeX(ind,1:2));
    vel_CC(:,:,t)       = diag(corr(Z_vel(ind,1:2), Dat.TeX(ind,1:2)))';
    
    ptrue               = [0 0 ; cumsum(Dat.TeX(ind,1:2))];
    z_pos               = [0 0 ; cumsum(Z_vel(ind,1:2))];
    
    pos_mae_(t,:)       = mae(ptrue, z_pos);
    
    ptrue_tmp(:,:,t_cnt(d_ind),d_ind) = pchip(linspace(0, 1, sum(ind)+1), ptrue', linspace(0, 1, fix_len))';
    zphat_tmp(:,:,t_cnt(d_ind),d_ind) = pchip(linspace(0, 1, sum(ind)+1), z_pos', linspace(0, 1, fix_len))';
end
pos_mae                 = pos_mae_;
vel_traj                = {Dat.TeX(:,1:2) Z_vel(:,1:2)};
pos_traj                = {ptrue_tmp zphat_tmp};