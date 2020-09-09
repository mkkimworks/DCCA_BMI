function h = fcnErrorbar2D(varargin)
% X, EX, col, mark, H
value                           = varargin{1};
vname                           = varargin(2:2:end);
variable                        = varargin(3:2:end);

data                            = value{1};
cnt                             = value{2};
[N,M_dim,~]                     = size(data);

Statistics                      = @ttest;
FaceColor                       = 'w';
XTickLabels                     = 1 : M_dim;
XLabel                          = '';
YLabel                          = '';
Title                           = '';
Gnd_i                           = [];
Fontsize                        = 10;
mark                            = '-';
col                             = 'k';
if sum(ismember(vname,'handles')) == 0
    Handle                      = axes;
end
for k = 1 : length(vname)
    switch vname{k}
        case 'handles'
            Handle              = variable{k};
        case 'linestyle'
            mark                = variable{k};
        case 'fontsize'
            Fontsize            = variable{k};
        case 'facecolor'
            FaceColor           = variable{k};
        case 'testtype'
            Statistics          = variable{k};
        case 'xticklabel'
            XTickLabels         = variable{k};
        case 'xlabel'
            XLabel              = variable{k};
        case 'ylabel'
            YLabel              = variable{k};
        case 'title'
            Title               = variable{k};
        case 'gnd'
            Gnd_i               = variable{k};
        case 'color'
            col                 = variable{k};
    end
end
hold(Handle, 'on');
TRJ                             = zeros(N, M_dim, length(cnt));
eTRJ                            = zeros(N, M_dim, length(cnt));
for d = 1 : length(cnt)
    TRJ(:,:,d)                  = mean(data(:,:,1:cnt(d),d), 3);
    eTRJ(:,:,d)                 = std(data(:,:,1:cnt(d),d), [],3) ./ sqrt(cnt(d));
end

for d = 1 : length(cnt)
    X                           = TRJ(:,:,d);
    EX                          = eTRJ(:,:,d);
    h0.main_line       = plot(X(:,1), X(:,2), 'linestyle',mark, 'color',col, 'linewidth',1.5,'parent',Handle);
    h0.edge_line1      = plot(X(:,1)+EX(:,1), X(:,2)+EX(:,2), 'linestyle','none', 'color',col, 'linewidth',0.5,'parent',Handle);
    h0.edge_line2      = plot(X(:,1)-EX(:,1), X(:,2)-EX(:,2), 'linestyle','none', 'color',col, 'linewidth',0.5,'parent',Handle);
    
    
    
    patchSaturation =0.15; %How de-saturated or transparent to make the patch
    faceAlpha=patchSaturation;
    patchColor=col;
    set(gcf,'renderer','openGL')
    
    %Calculate the y values at which we will place the error bars
    x_uE = X(:,1)+EX(:,1);
    x_lE = X(:,1)-EX(:,1);
    
    y_uE = X(:,2)+EX(:,2);
    y_lE = X(:,2)-EX(:,2);
    
    %Make the cordinats for the patch
    yP=[y_lE',fliplr(y_uE')];
    xP=[x_lE',fliplr(x_uE')];
    
    %remove any nans otherwise patch won't work
    xP(isnan(yP))=[];
    yP(isnan(yP))=[];
    
    h0.H_patch = patch(xP,yP,1,'facecolor',patchColor,...
        'edgecolor','none',...
        'facealpha',faceAlpha,'parent',Handle);
    
    title(Title,'parent',Handle);
    xlabel(XLabel,'parent',Handle);
    ylabel(YLabel,'parent',Handle);
    set(Handle, 'fontsize', Fontsize);
    H(d) = Handle;
end

