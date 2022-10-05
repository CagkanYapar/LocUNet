function [difBis] =  delay_DBBisecRobFunCluster(antList,antX,antY,rxx,rxy,matDelay,std,noAnt,bmax)
%S. Tomic, M. Beko, R. Dinis, and P. Montezuma, “A robust bisection-based estimator for TOA-based target localization
%in NLOS environments,” IEEE Communications Letters, vol. 21, no. 11, pp. 2488–2491, 2017.

v = physconst('LightSpeed');
noTx = noAnt;
po = zeros(noTx,3);

warning('off','MATLAB:singularMatrix')

delDist = zeros(noTx,1);

noBS = 0;
for tx=1:noTx
    ant = antList(tx);
    
%     plTrue = imread(strcat(pathRadioMapsTrue,int2str(ind2),'_',int2str(ant-1),'.png'));
%     gainMapsTrue = im2double(plTrue);
%     
%     gain = gainMapsTrue(rxx,rxy);
    
    po(tx,1) = antX(tx);
    po(tx,2) = antY(tx);
    
%     if gain == 0
%         delTemp = 0;
%     else
        delTemp = matDelay(rxx,rxy,ant);
%         noBS = noBS + 1;
%     end
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
    pp = BisecRob(po,noTx2,bmax);
    pTrue = [rxx, rxy];
    diffRBis = pTrue(1)-pp(1);
    diffCBis = pTrue(2)-pp(2);
    difBis = sqrt(diffRBis^2+diffCBis^2);
    
else
    difBis = NaN;
end


end
