function [difBis] =  delay_DBBisecMaxCorrFunCluster(mm,antList,antX,antY,rxx,rxy,matDelay,std,noAnt)
%W. Xiong, C. Schindelhauer, H. Cheung So, and Z. Wang, “Maximum correntropy criterion for robust TOA-based
%localization in NLOS environments,” arXiv e-prints, p. arXiv:2009.06032, Sep. 2020.

v = physconst('LightSpeed');
noTx = size(antList,1);
noTx = noAnt;
po = zeros(noTx,3);

warning off


pathDataset = strcat(pwd,'/dataset');
ind2 = mm - 1;
pathRadioMapsTrue = strcat(pathDataset,'/DPM/true/');

delDist = zeros(noTx,1);
for tx=1:noTx
    ant = antList(tx);
    
    plTrue = imread(strcat(pathRadioMapsTrue,int2str(ind2),'_',int2str(ant-1),'.png'));
    gainMapsTrue = im2double(plTrue);
    
    gain = gainMapsTrue(rxx,rxy);
    
    po(tx,1) = antX(tx);
    po(tx,2) = antY(tx);
    
    if gain == 0
        delTemp = 0;
    else
        delTemp = matDelay(rxx,rxy,ant);
    end
    delDist(tx) = (delTemp*v)/(10^9);
    po(tx,3) = delDist(tx) + std*normrnd(0,1);
    
    
end
%Check 0 distances

zeroTxs = find(~delDist);
zeroTxs = sort(zeroTxs, 'descend');
noTx2 = noTx;

if ~isempty(zeroTxs)
    noZeros = size(zeroTxs,1);
    noTx2 = noTx - noZeros;
    for z=1:noZeros
        
        po(zeroTxs(z),:) = [];
        delDist(zeroTxs(z)) = [];
    end
end
if ~isempty(po)
    pp = BisecCorr(po,noTx2);%
    
    pTrue = [rxx, rxy];

    diffRBis = pTrue(1)-pp(1);
    diffCBis = pTrue(2)-pp(2);
    difBis = sqrt(diffRBis^2+diffCBis^2);
    
else
    difBis = NaN;
end


% print('done')





end
