function Dat                = DataSegmentation(data, training_set_ratio,n)

UT                          = unique(data.IND);
IND                         = [];
for u = 1 : length(UT)
    len                     = sum(UT(u) == data.IND);
    IND                     = [IND ; u*ones(len,1)];
end
seg                         = segment_data(IND, training_set_ratio, 0);

data.Z                      = sqrt(data.Z);
Dat.TrI                     = IND(seg.trid);
Dat.TeI                     = IND(seg.teid);
Dat.TrZ                     = data.Z(seg.trid,:);
Dat.TeZ                     = data.Z(seg.teid,:);
Dat.TrX                     = [data.X(seg.trid,:) sqrt(sum(data.X(seg.trid,:).^2,2))];
Dat.TeX                     = [data.X(seg.teid,:) sqrt(sum(data.X(seg.teid,:).^2,2))];
if n == 1
    Dat.TrG                 = data.Target.deg(seg.trtri);
    Dat.TeG                 = data.Target.deg(seg.tetri);
else
    Dat.TrG                 = data.Target(seg.trtri);
    Dat.TeG                 = data.Target(seg.tetri);
end