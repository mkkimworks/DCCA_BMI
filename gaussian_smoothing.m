function yOut                   = gaussian_smoothing(Z, bin_length, kernSD)
if nargin < 2
    kernSD                      = 200;
end
stepSize                        = bin_length * 1000;
fltHL                           = ceil(3 * kernSD / stepSize);
flt                             = normpdf(-fltHL*stepSize : stepSize : fltHL*stepSize, 0, kernSD);
flt(1:fltHL)                    = 0;

[T, yDim]                       = size(Z);
yOut                            = nan(T,yDim);

% Normalize by sum of filter taps actually used
nm = conv(flt, ones(1, T));
for i = 1 : yDim
    ys = conv(flt, Z(:,i)') ./ nm;
    % Cut off edges so that result of convolution is same length
    yOut(:,i)                   = ys(fltHL+1:end-fltHL);
end


