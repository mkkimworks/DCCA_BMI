function [h0, h1, H] = nan_imagesc(x, y, map_pos, H)
if nargin < 4
    figure;
    H = axes;
end
colormap(jet); 
imAlpha=ones(size(map_pos)); imAlpha(isnan(map_pos))=0; 
h0 = imagesc(x,y,map_pos,'AlphaData',imAlpha,'parent', H); 
set(H,'ydir','normal','color',[1 1 1],'fontsize',12); 
h1 = colorbar; 
h1.FontSize = 13; 
% set(H, 'DataAspectRatio',[1 1 1]);