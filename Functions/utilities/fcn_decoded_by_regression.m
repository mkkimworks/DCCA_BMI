function [pos_mae,vel_mae,vel_CC, traj, t_cnt] = fcn_decoded_by_regression(Dat, TrPCA_Z, TePCA_Z, TrFA_Z, TeFA_Z, TrLCCA_Z, TeLCCA_Z, TrLCCA_X, TeLCCA_X, TrDCCA_Z, TeDCCA_Z, TrDCCA_X, TeDCCA_X, lcca_prm, dcca_prm, t_cnt)

N                       = size(Dat.TrZ,1);
XX                      = [ones(N,1) Dat.TrZ];
B                       = Dat.TrX' * XX * (XX' * XX)^-1;
reg_hat0                = [ones(size(Dat.TeZ,1),1) Dat.TeZ] * B';

XX                      = [ones(N,1) TrPCA_Z];
B                       = Dat.TrX' * XX * (XX' * XX)^-1;
reg_hat1                = [ones(size(TePCA_Z,1),1) TePCA_Z] * B';

XX                      = [ones(N,1) TrFA_Z];
B                       = Dat.TrX' * XX * (XX' * XX)^-1;
reg_hat2                = [ones(size(TeFA_Z,1),1) TeFA_Z] * B';

XX                      = [ones(N,1) TrLCCA_Z];
B                       = Dat.TrX' * XX * (XX' * XX)^-1;
reg_hat3                = [ones(size(TeFA_Z,1),1) TeLCCA_Z] * B';


XX                      = [ones(N,1) TrDCCA_Z];
B                       = TrDCCA_X' * XX * (XX' * XX)^-1;
reg_hat4                = dcca_prm.fcn.invB([ones(size(TeFA_Z,1),1) TeDCCA_Z] * B');


vel_mae                 = [ mae(reg_hat0(:,1:2), Dat.TeX(:,1:2))
                            mae(reg_hat1(:,1:2), Dat.TeX(:,1:2))
                            mae(reg_hat2(:,1:2), Dat.TeX(:,1:2))
                            mae(reg_hat3(:,1:2), Dat.TeX(:,1:2))
                            mae(reg_hat4(:,1:2), Dat.TeX(:,1:2))];

vel_CC                  = [ diag(corr(reg_hat0(:,1:2), Dat.TeX))'
                            diag(corr(reg_hat1(:,1:2), Dat.TeX))'
                            diag(corr(reg_hat2(:,1:2), Dat.TeX))'
                            diag(corr(reg_hat3(:,1:2), Dat.TeX))'
                            diag(corr(reg_hat4(:,1:2), Dat.TeX))'
                            ];
                        
                        
fix_len                 = 15;
TU                      = unique(Dat.TeI);
pos_mae_                = zeros(size(TU,1),5);
for t = 1 : size(TU,1)
    ind                 = TU(t) == Dat.TeI;
    d_ind               = Dat.TargetIndex == Dat.TeG(t);
    t_cnt(d_ind)        = t_cnt(d_ind) + 1;
    
    ptrue               = [0 0 ; cumsum(Dat.TeX(ind,1:2))];
    z_pos               = [0 0 ; cumsum(reg_hat0(ind,1:2))];
    p_pos               = [0 0 ; cumsum(reg_hat1(ind,1:2))];
    f_pos               = [0 0 ; cumsum(reg_hat2(ind,1:2))];
    l_pos               = [0 0 ; cumsum(reg_hat3(ind,1:2))];
    d_pos               = [0 0 ; cumsum(reg_hat4(ind,1:2))];
    
    pos_mae_(t,:)       = [mae(ptrue, z_pos)
                            mae(ptrue, p_pos)
                            mae(ptrue, f_pos)
                            mae(ptrue, l_pos)
                            mae(ptrue, d_pos)];
    
    ptrue_tmp(:,:,t_cnt(d_ind),d_ind) = pchip(linspace(0, 1, sum(ind)+1), ptrue', linspace(0, 1, fix_len))';
    zphat_tmp(:,:,t_cnt(d_ind),d_ind) = pchip(linspace(0, 1, sum(ind)+1), z_pos', linspace(0, 1, fix_len))';
    pphat_tmp(:,:,t_cnt(d_ind),d_ind) = pchip(linspace(0, 1, sum(ind)+1), p_pos', linspace(0, 1, fix_len))';
    fphat_tmp(:,:,t_cnt(d_ind),d_ind) = pchip(linspace(0, 1, sum(ind)+1), f_pos', linspace(0, 1, fix_len))';
    lphat_tmp(:,:,t_cnt(d_ind),d_ind) = pchip(linspace(0, 1, sum(ind)+1), l_pos', linspace(0, 1, fix_len))';
    dphat_tmp(:,:,t_cnt(d_ind),d_ind) = pchip(linspace(0, 1, sum(ind)+1), d_pos', linspace(0, 1, fix_len))';
end
pos_mae                 = nanmean(pos_mae_,1);
traj                    = {ptrue_tmp zphat_tmp pphat_tmp fphat_tmp lphat_tmp dphat_tmp};



