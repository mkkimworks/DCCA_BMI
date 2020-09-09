function KF = InitKF(KF)

KF.x0 = zeros(size(KF.x0));
KF.P = KF.W;
% KF.K = zeros(size(KF.K));