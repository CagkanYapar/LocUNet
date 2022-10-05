function pp = BisecCorr(po,noTx)
%W. Xiong, C. Schindelhauer, H. Cheung So, and Z. Wang, “Maximum correntropy criterion for robust TOA-based
%localization in NLOS environments,” arXiv e-prints, p. arXiv:2009.06032, Sep. 2020.

gamma = 10^(-5);
N_max = 10;

pp1 = Inf*ones(2,2);

dists = po(:,3);

pp = zeros(2,1);
%try initializing in the middle to prevent NaNs later
pp(1) = 128;
pp(2) = 128;
A = zeros(noTx,3);
f = zeros(noTx,1);
g = zeros(3,1);

p_k = zeros(noTx,1);

W = zeros(noTx);
D = zeros(3);


A(1:noTx,3) = 1;

g(3) = -0.5;
D(1,1) = 1;
D(2,2) = 1;

%large initial sigma to prevent NaN
sig = 10000000;

k=1;
diff = Inf;

while k <= N_max && diff > gamma
    for nn=1:noTx
        
        p_k(nn) = -exp(-((dists(nn)^2 - (pp(1)-po(nn,1))^2 - (pp(2)-po(nn,2))^2)^2/2/(sig^2)));
        
        A(nn,1) = 2*po(nn,1);
        A(nn,2) = 2*po(nn,2);
        f(nn) = -po(nn,1)^2 - po(nn,2)^2 + dists(nn)^2;
        
        W(nn,nn) = sqrt(-p_k(nn));
        
    end
    if sum(W(:)) == 0
        break
    end
    
    temp1 = ((A.')*(W.')*W*A)^(-0.5);
    temp2 = temp1 * D * temp1;
    if sum(sum(isnan(temp1))) > 0 | sum(sum(isnan(temp2))) > 0 | sum(sum(isinf(temp1))) > 0 | sum(sum(isinf(temp2))) > 0
        break
    end
    eigMax = eigs(temp2,1);
    
    lb = -1/eigMax;
    ub = 100000;
    lb = real(lb);
    
    funL = @(l) ((((A.')*(W.')*W*A+l.*D)\((A.')*(W.')*W*f-l.*g)).')*D*(((A.')*(W.')*W*A+l.*D)\((A.')*(W.')*W*f-l.*g)) + 2*(g.')*(((A.')*(W.')*W*A+l.*D)\((A.')*(W.')*W*f-l.*g));
    
    lambda = bisection(funL,lb,ub);
    
    yHat = inv((A.')*(W.')*W*A + lambda*D)*((A.')*(W.')*W*f - lambda*g);
    
    pp(1) = -yHat(1);
    pp(2) = -yHat(2);
    
    pp1(1,1) = pp1(1,2);
    pp1(2,1) = pp1(2,2);
    pp1(1,2) = pp(1);
    pp1(2,2) = pp(2);
    
    if k > 1
        diff = sqrt((pp1(1,1) - pp1(1,2))^2 + (pp1(2,1) - pp1(2,2))^2);
    end
       
    aux = dists.^2 - (pp(1)-po(:,1)).^2 - (pp(2)-po(:,2)).^2;
    
    stdE = std(aux);
    R = iqr(aux);
    sig = 1.06*min(stdE,R/1.34)*(noTx^(-0.2));
    
    k = k + 1;
    

end
end