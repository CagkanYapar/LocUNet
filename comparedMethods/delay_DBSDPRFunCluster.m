function [difBis] =  delay_DBSDPRFunCluster(mm,antList,antX,antY,rxx,rxy,matDelay,std,noAnt,b)
%G. Wang, H. Chen, Y. Li, and N. Ansari, “NLOS error mitigation for TOA-based localization via convex relaxation,” IEEE
%Transactions on Wireless Communications, vol. 13, no. 8, pp. 4119–4131, 2014.

v = physconst('LightSpeed');

noTx = noAnt;
po = zeros(noTx,3);
sig = std;

if std == 0
    sig = 0.0001;
end

delDist = zeros(noTx,1);

pathDataset = strcat(pwd,'/dataset');
ind2 = mm - 1;
pathRadioMapsTrue = strcat(pathDataset,'/DPM/true/');


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
    bmax = b;
    pp = SDPR(po, sig, noTx2, bmax);
    pTrue = [rxx, rxy];
    diffRBis = pTrue(1)-pp(1);
    diffCBis = pTrue(2)-pp(2);
    difBis = sqrt(diffRBis^2+diffCBis^2);
    
else
    difBis = NaN;
end


end
