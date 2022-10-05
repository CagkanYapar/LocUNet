function difPOCS =  delay_DBPOCSFunCluster(mm,antList,antX,antY,rxx,rxy,matDelay,std,noAnt)
%M. R. Gholami, H. Wymeersch, E. G. Strom, and M. Rydstr ¨ om, “Wireless network positioning as a convex feasibility ¨
%problem,” EURASIP Journal on Wireless Communications and Networking, vol. 2011, no. 1, p. 161, 2011

v = physconst('LightSpeed');
noTx = size(antList,1);
noTx = noAnt;

po = zeros(noTx,3);

% del = zeros(noTx,1);
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
noTx = noTx2;

pTrue = [rxx, rxy];

%Initialize for POCS
pp = [128, 128];

v0 = 10*noTx;
v1 = 20*noTx;
vSum = v0 + v1;

for tt=1:vSum
    if tt <= v0
        l = 1;
    else
        l = 1/(ceil((tt-v0+1)/noTx));
    end
    ttt = mod(tt-1,noTx) + 1;
    pp = POCSstep(po(ttt,:),pp,l);
end


%POCS error
diffRPocs = pTrue(1)-pp(1);
diffCPocs = pTrue(2)-pp(2);
difPOCS = sqrt(diffRPocs^2+diffCPocs^2);

end
