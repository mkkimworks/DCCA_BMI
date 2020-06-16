function KF = InitLKF(KF)

KF.x0 = zeros(size(KF.x0));
KF.P = KF.W;