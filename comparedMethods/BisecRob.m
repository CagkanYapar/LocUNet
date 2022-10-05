function pp = BisecRob(po,noTx,bmax)
%S. Tomic, M. Beko, R. Dinis, and P. Montezuma, “A robust bisection-based estimator for TOA-based target localization
%in NLOS environments,” IEEE Communications Letters, vol. 21, no. 11, pp. 2488–2491, 2017.

warning('off','MATLAB:singularMatrix')

dists = po(:,3);
distTilde = dists - bmax/2;

satInd = find(distTilde <= bmax/2);
numSat = numel(satInd);

noTx2 = 2*noTx + numSat;

pp = zeros(2,1);
A = zeros(noTx2,3);
f = zeros(noTx2,1);
g = zeros(3,1);

W = zeros(noTx2);
D = zeros(3);


A(1:2*noTx,3) = -1;
if numSat > 0
    A(2*noTx+1:end,3) = 1;
end

g(3) = -0.5;
D(1,1) = 1;
D(2,2) = 1;


%for weights, get the sum dist
distSum = 0;
for nn=1:noTx
    distSum = distSum + po(nn,3);
end


for nn=1:2*noTx
        
    if nn <= noTx
        A(nn,1) = 2*po(nn,1);
        A(nn,2) = 2*po(nn,2);
        f(nn) = po(nn,1)^2 + po(nn,2)^2 - distTilde(nn)^2 - (bmax^2)/4;
        f(nn) = f(nn) + distTilde(nn)*bmax;
        w = 1 - po(nn,3)/distSum;
        w = sqrt(w)/2/po(nn,3);
        W(nn,nn) = w;
        
    else
        A(nn,1) = 2*po(nn-noTx,1);
        A(nn,2) = 2*po(nn-noTx,2);
        f(nn) = po(nn-noTx,1)^2 + po(nn-noTx,2)^2 - distTilde(nn-noTx)^2 - (bmax^2)/4;
        f(nn) = f(nn) - distTilde(nn-noTx)*bmax;
        w = 1 - po(nn-noTx,3)/distSum;
        w = sqrt(w)/2/po(nn-noTx,3);
        W(nn,nn) = w;
        
    end
    %weights
    
end

if numSat > 0
    for nn=1:numSat
        A(2*noTx+nn,1) = -2*po(satInd(nn),1);
        A(2*noTx+nn,2) = -2*po(satInd(nn),2);
        f(2*noTx+nn) = -po(satInd(nn),1)^2 - po(satInd(nn),2)^2;
    end
end

temp1 = ((A.')*(W.')*W*A)^(-0.5);
temp2 = temp1 * D * temp1;
eigMax = eigs(temp2,1);

lb = -1/eigMax;
ub = 100;

lb = real(lb);

funL = @(l) ((((A.')*(W.')*W*A+l.*D)\((A.')*(W.')*W*f-l.*g)).')*D*(((A.')*(W.')*W*A+l.*D)\((A.')*(W.')*W*f-l.*g)) + 2*(g.')*(((A.')*(W.')*W*A+l.*D)\((A.')*(W.')*W*f-l.*g));

lambda = bisection(funL,lb,ub);

yHat = inv((A.')*(W.')*W*A + lambda*D)*((A.')*(W.')*W*f - lambda*g);

pp(1) = yHat(1);
pp(2) = yHat(2);
end