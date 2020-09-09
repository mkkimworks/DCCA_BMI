function [YC, VC]       = predictNeuralActivity(TestY, prm, dr_type)

Y                       = TestY';
zDim                    = size(TestY,2);
YC                      = zeros(size(TestY));
VC                      = zeros(size(TestY,2),1);
switch dr_type
    case 'pca'
        YC              = cosmoother_pca(Y, prm)';
    case 'fa'
        [YC, VC]        = cosmoother_fa(Y, prm);
    case 'lds'
%         C               = prm.C;
        Z               = kalmanSmoother(prm, Y);
%         YC              = (C*Z)';
        
        Amu             = prm.A*Z;
        YC              = (prm.C*Amu)';
        
        
%         C               = prm.C;
%         [yDim, xDim] = size(C);
%         Ycs = zeros(size(Y));
%         for i = 1:yDim
%             mi          = [1:(i-1) (i+1):yDim];
%             Xmi         = inv(C(mi,:)' * C(mi,:)) * C(mi,:)' * Y(mi,:);
%             Ycs(i,:)    = C(i,:) * Xmi;
%         end
%         YC              = Ycs';
end


