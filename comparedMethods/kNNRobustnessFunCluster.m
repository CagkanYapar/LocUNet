
function dif = kNNRobustnessFunCluster(k,mm,antList,rxx,rxy)

pathDataset = strcat(pwd,'/dataset');

noAnt = 5;


noTx = noAnt;


ind2 = mm - 1;

pathRadioMapsEst = strcat(pathDataset,'/DPM/estimate/');
pathRadioMapsTrue = strcat(pathDataset,'/IRT2cars/true/');



gainMapsEst = zeros(noAnt,256,256);
gainMapsTrue = zeros(noAnt,256,256);


for nn=1:noAnt
    ant = antList(nn);
    
    
    plTrue = imread(strcat(pathRadioMapsTrue,int2str(ind2),'_',int2str(ant-1),'.png'));
    gainMapsTrue(nn,:,:) = im2double(plTrue);
    
    plEst = imread(strcat(pathRadioMapsEst,int2str(ind2),'_',int2str(ant-1),'.png'));
    gainMapsEst(nn,:,:) = im2double(plEst);
    
end

%%% Determine k nearest neighbours

imsum = zeros(256,256);

for nn = 1:noTx
    
    gain = gainMapsTrue(nn,rxx,rxy);
    mapTemp2 = squeeze(gainMapsEst(nn,:,:));
    mapTemp = (mapTemp2 - gain).^2;
    
    imsum = imsum + mapTemp;
end

[sortedX, sortedInds] = sort(imsum(:),'ascend');
topk = sortedInds(1:k);
[XAll, YAll] = ind2sub(size(imsum), topk);

if ~isempty(XAll)
    %Center of Mass of the ks
    avX = sum(XAll)/length(XAll);
    avY = sum(YAll)/length(YAll);
    
    %%% find the difference between av and true
    diffX = abs(avX-rxx);
    diffY = abs(avY-rxy);
    avDiff = sqrt(diffX^2+diffY^2);
    
    
else%if found nothing, take the center...
    diffX = abs(128-rxx);
    diffY = abs(128-rxy);
    avDiff = sqrt(diffX^2+diffY^2);
    
end

dif = avDiff;




