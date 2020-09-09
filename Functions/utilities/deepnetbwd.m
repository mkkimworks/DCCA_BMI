function [output1, output2] = deepnetbwd(X1, F1_opt, beta_)
numLayers           = length(F1_opt);

if nargin < 3
    OutX            = X1;
    OutX_buffer     = cell(numLayers,1);
    N               = size(X1,1);
    for l = 1 : numLayers
        switch F1_opt{l}.type
            case 'linear'
                OutX            = [OutX, ones(N,1)]*F1_opt{l}.W;
            case 'sigmoid'
                OutX            = [OutX, ones(N,1)]*F1_opt{l}.W;
                OutX            = 1./(1+exp(-OutX));
        end
        OutX_buffer(l)          = {OutX};
    end
    
    tmp                         = OutX_buffer{end};
    beta_                       = cell(numLayers-1,1);
    for l = numLayers : -1 : 1
        
        switch F1_opt{l}.type
            case 'linear'
                tmp                 = tmp / F1_opt{l}.W;
%                 tmp                 = F1_opt{l}.W * (F1_opt{l}.W' * F1_opt{l}.W)^-1 * tmp';
                beta                = [ones(N,1) tmp] \ OutX_buffer{l-1};
                tmp                 = [ones(N,1) tmp] * beta;
%                 beta                = [];
%                 for c = 1 : size(OutX_buffer{l-1},2)
%                     beta(:,c)       = glmfit(gather(tmp), gather(OutX_buffer{l-1}(:,c),'distribution','poisson','link','log'));
%                 end
%                 tmp                 = exp([ones(size(tmp,1),1) tmp] * beta);
                beta_(l)            = {beta};
                if strcmp(F1_opt{l-1}.type,'sigmoid')
                    tmp             = real(-log(complex(1./tmp - 1)));
                end
            case 'sigmoid'
                tmp                 = tmp / F1_opt{l}.W;
                if l > 1
                    beta            = [ones(size(tmp,1),1) tmp] \ OutX_buffer{l-1};
                    tmp             = [ones(size(tmp,1),1) tmp] * beta;
%                     beta                = [];
%                     for c = 1 : size(OutX_buffer{l-1},2)
%                         beta(:,c)       = glmfit(gather(tmp), gather(OutX_buffer{l-1}(:,c),'distribution','poisson','link','log'));
%                     end
%                     tmp             = exp([ones(size(tmp,1),1) tmp] * beta);
                    beta_(l)        = {beta};
                    if strcmp(F1_opt{l-1}.type,'sigmoid')
                        tmp         = real(-log(complex(1./tmp - 1)));
                    end
                end
        end
    end
    recon_X                     = gather(real(tmp));
    output1                     = beta_;
    output2                     = recon_X;
else
    
    tmp                         = X1;
    for l = numLayers : -1 : 1
        beta                        = beta_{l};
        switch F1_opt{l}.type
            case 'linear'
                tmp                 = tmp / F1_opt{l}.W;
                tmp                 = [ones(size(tmp,1),1) tmp] * beta;
%                 tmp                 = exp([ones(size(tmp,1),1) tmp] * beta);
                if strcmp(F1_opt{l-1}.type,'sigmoid')
                    tmp             = real(-log(complex(1./tmp - 1)));
                end
            case 'sigmoid'
                tmp                 = tmp / F1_opt{l}.W;
                if l > 1
%                     tmp             = exp([ones(size(tmp,1),1) tmp] * beta);
                    tmp             = [ones(size(tmp,1),1) tmp] * beta;
                    if strcmp(F1_opt{l-1}.type,'sigmoid')
                        tmp         = real(-log(complex(1./tmp - 1)));
                    end
                end
        end
    end
    recon_X                     = gather(real(tmp));
    output1                     = recon_X;
    output2                     = [];
end