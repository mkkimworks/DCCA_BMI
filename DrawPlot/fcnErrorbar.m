%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   
%   Title: Decoding kinematic information from primary motor cortical 
%          ensemble activity using a deep canonical correlation analysis
%   Authors: M-K Kim; J-W Sohn; S-P Kim
%   E-mail: mkkim.works@gmail.com
%   Affiliation: Ulsan National Institute of Science and Technology (UNIST)
%
%   h = fcnErrorbar(varargin)
%   Inputs 
%       value: 2d input vectors
%       FaceColor            default 'w';
%   Outputs
%       h: handles
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function h = fcnErrorbar(varargin)


value                           = varargin{1};
vname                           = varargin(2:2:end);
variable                        = varargin(3:2:end);

[N,M_dim]                       = size(value);

FaceColor                       = 'w';
XTickLabels                     = 1 : M_dim;
XLabel                          = '';
YLabel                          = '';
Title                           = '';
if sum(ismember(vname,'handles')) == 0
    Handle                      = axes;
end
for k = 1 : length(vname)
    switch vname{k}
        case 'handles'
            Handle              = variable{k};
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
    end
end



mu                              = nanmean(value,1);
er                              = nanstd(value,[],1) ./ sqrt(N);

bar(1 : M_dim, mu, 'FaceColor', FaceColor, 'Parent',Handle); hold(Handle,'on');
errorbar(1 : M_dim, mu, er, 'LineStyle','none','color','k','Parent',Handle);

title(Title);
xlabel(XLabel);
ylabel(YLabel);
set(Handle,'XTickLabel',XTickLabels,'fontsize',13);
xtickangle(45);
box off;
h = Handle;