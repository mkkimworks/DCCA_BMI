function [pos_mae,vel_mae,vel_CC, vel_traj, pos_traj, t_cnt] = fcn_decoded_by_lstm(Dat, TePCA_Z, TeFA_Z, TeLDS_Z, TeLCCA_Z, TeDCCA_Z, lstm_prm, t_cnt)

Z_vel                   = lstm_prm.FR.fcn(Dat.TeZ);                     Z_vel = Z_vel(:,1:2);
P_vel                   = lstm_prm.PC.fcn(TePCA_Z);                     P_vel = P_vel(:,1:2);
F_vel                   = lstm_prm.FA.fcn(TeFA_Z);                      F_vel = F_vel(:,1:2);
S_vel                   = lstm_prm.LDS.fcn(TeLDS_Z);                    S_vel = S_vel(:,1:2);
L_vel                   = lstm_prm.LCCA.fcn(TeLCCA_Z);                  L_vel = L_vel(:,1:2);
D_vel                   = lstm_prm.DCCA.fcn(TeDCCA_Z);                  D_vel = D_vel(:,1:2);


TargetIndex             = unique([Dat.TrG;Dat.TeG]);
fix_len                 = 15;
TU                      = unique(Dat.TeI);
pos_mae_                = zeros(size(TU,1),6);
for t = 1 : size(TU,1)
    ind                 = TU(t) == Dat.TeI;
    d_ind               = TargetIndex == Dat.TeG(t);
    t_cnt(d_ind)        = t_cnt(d_ind) + 1;
    
    vel_mae(t,:)        = [ mae(Z_vel(ind,1:2), Dat.TeX(ind,1:2))
                            mae(P_vel(ind,1:2), Dat.TeX(ind,1:2))
                            mae(F_vel(ind,1:2), Dat.TeX(ind,1:2))
                            mae(S_vel(ind,1:2), Dat.TeX(ind,1:2))
                            mae(L_vel(ind,1:2), Dat.TeX(ind,1:2))
                            mae(D_vel(ind,1:2), Dat.TeX(ind,1:2))
        ];
    vel_CC(:,:,t)       = [ diag(corr(Z_vel(ind,1:2), Dat.TeX(ind,1:2)))'
                            diag(corr(P_vel(ind,1:2), Dat.TeX(ind,1:2)))'
                            diag(corr(F_vel(ind,1:2), Dat.TeX(ind,1:2)))'
                            diag(corr(S_vel(ind,1:2), Dat.TeX(ind,1:2)))'
                            diag(corr(L_vel(ind,1:2), Dat.TeX(ind,1:2)))'
                            diag(corr(D_vel(ind,1:2), Dat.TeX(ind,1:2)))'
                            ];
    
    ptrue               = [0 0 ; cumsum(Dat.TeX(ind,1:2))];
    z_pos               = [0 0 ; cumsum(Z_vel(ind,1:2))];
    p_pos               = [0 0 ; cumsum(P_vel(ind,1:2))];
    f_pos               = [0 0 ; cumsum(F_vel(ind,1:2))];
    s_pos               = [0 0 ; cumsum(S_vel(ind,1:2))];
    l_pos               = [0 0 ; cumsum(L_vel(ind,1:2))];
    d_pos               = [0 0 ; cumsum(D_vel(ind,1:2))];
    
    pos_mae_(t,:)       = [mae(ptrue, z_pos)
                            mae(ptrue, p_pos)
                            mae(ptrue, f_pos)
                            mae(ptrue, s_pos)
                            mae(ptrue, l_pos)
                            mae(ptrue, d_pos)];
    
    ptrue_tmp(:,:,t_cnt(d_ind),d_ind) = pchip(linspace(0, 1, sum(ind)+1), ptrue', linspace(0, 1, fix_len))';
    zphat_tmp(:,:,t_cnt(d_ind),d_ind) = pchip(linspace(0, 1, sum(ind)+1), z_pos', linspace(0, 1, fix_len))';
    pphat_tmp(:,:,t_cnt(d_ind),d_ind) = pchip(linspace(0, 1, sum(ind)+1), p_pos', linspace(0, 1, fix_len))';
    fphat_tmp(:,:,t_cnt(d_ind),d_ind) = pchip(linspace(0, 1, sum(ind)+1), f_pos', linspace(0, 1, fix_len))';
    sphat_tmp(:,:,t_cnt(d_ind),d_ind) = pchip(linspace(0, 1, sum(ind)+1), s_pos', linspace(0, 1, fix_len))';
    lphat_tmp(:,:,t_cnt(d_ind),d_ind) = pchip(linspace(0, 1, sum(ind)+1), l_pos', linspace(0, 1, fix_len))';
    dphat_tmp(:,:,t_cnt(d_ind),d_ind) = pchip(linspace(0, 1, sum(ind)+1), d_pos', linspace(0, 1, fix_len))';
end
pos_mae                 = pos_mae_;
vel_traj                = {Dat.TeX Z_vel P_vel F_vel S_vel L_vel D_vel};
pos_traj                = {ptrue_tmp zphat_tmp pphat_tmp fphat_tmp sphat_tmp lphat_tmp dphat_tmp};