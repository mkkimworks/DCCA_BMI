function output = segment_data(I, prc, isshuffle)

uI              = unique(I);
N               = size(uI,1);
M               = size(I,1);

trsz            = floor(N * prc);
tesz            = N - floor(N * prc);

if isshuffle
    ri          = randperm(N);
else
    ri          = 1 : N;
end

tr_tmp          = uI(ri(1 : trsz));
te_tmp          = uI(ri(trsz+1:end));

output.trid     = false(M,1);
for k = 1 : trsz
    output.trid(tr_tmp(k) == I) = 1;
end
output.teid     = false(M,1);
for k = 1 : tesz
    output.teid(te_tmp(k) == I) = 1;
end
output.trsz     = trsz;
output.tesz     = tesz;
output.trLen    = sum(output.trid);
output.teLen    = sum(output.teid);
output.tatal_length = M;

output.trtri    = unique(I(output.trid));
output.tetri    = unique(I(output.teid));