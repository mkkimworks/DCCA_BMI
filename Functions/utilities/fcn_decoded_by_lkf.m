function [pos_mae,vel_mae,vel_CC, vel_traj, pos_traj, t_cnt] = fcn_decoded_by_lkf(Dat, TrPCA_Z, TePCA_Z, TrFA_Z, TeFA_Z, TrLDS_Z, TeLDS_Z, TrLCCA_Z, TeLCCA_Z, TrDCCA_Z, TeDCCA_Z, t_cnt)

Z_KF                    = KFTraining(Dat.TrZ, Dat.TrX(:,1:2), Dat.TrI);
P_KF                    = KFTraining(TrPCA_Z, Dat.TrX(:,1:2), Dat.TrI);
F_KF                    = KFTraining(TrFA_Z, Dat.TrX(:,1:2), Dat.TrI);
S_KF                    = KFTraining(TrLDS_Z, Dat.TrX(:,1:2), Dat.TrI);
L_KF                    = KFTraining(TrLCCA_Z, Dat.TrX(:,1:2), Dat.TrI);
D_KF                    = KFTraining(TrDCCA_Z, Dat.TrX(:,1:2), Dat.TrI);


Z_vel                   = iPredict(Dat.TeZ, @KFpredict, Z_KF, Dat.TeI);
P_vel                   = iPredict(TePCA_Z, @KFpredict, P_KF, Dat.TeI);
F_vel                   = iPredict(TeFA_Z, @KFpredict, F_KF, Dat.TeI);
S_vel                   = iPredict(TeLDS_Z, @KFpredict, S_KF, Dat.TeI);
L_vel                   = iPredict(TeLCCA_Z, @KFpredict, L_KF, Dat.TeI);
D_vel                   = iPredict(TeDCCA_Z, @KFpredict, D_KF, Dat.TeI);



                        
TargetIndex             = unique([Dat.TrG;Dat.TeG]);
fix_len                 = 15;
TU                      = unique(Dat.TeI);
pos_mae_                = zeros(size(TU,1),6);
vel_mae                 = [];
vel_CC                  = [];
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
vel_traj                = {Dat.TeX(:,1:2) Z_vel(:,1:2) P_vel(:,1:2) F_vel(:,1:2) S_vel(:,1:2) L_vel(:,1:2) D_vel(:,1:2)};
pos_traj                = {ptrue_tmp zphat_tmp pphat_tmp fphat_tmp sphat_tmp lphat_tmp dphat_tmp};