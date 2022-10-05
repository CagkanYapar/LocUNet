function x = SDPR(po, sig, noTx, bmax)
%G. Wang, H. Chen, Y. Li, and N. Ansari, “NLOS error mitigation for TOA-based localization via convex relaxation,” IEEE
%Transactions on Wireless Communications, vol. 13, no. 8, pp. 4119–4131, 2014.

d = po(:,3);
% Q = sig*eye(noTx);
Qinv = (1/sig)*eye(noTx);
S = po(:,1:2);

H = zeros(noTx,3);
H(:,1:2) = -2*S;
H(:,3) = 1;

f = d.^2 - S(:,1).^2 - S(:,2).^2;

%%%%%%%%%%%%%%%%%%%%%%%%%% cvx %%%%%%%%%%%%%%%%%%%%%%%%%%%

cvx_begin sdp quiet
cvx_solver sedumi

variable G(noTx,noTx)
variable g(noTx,1)
variable x(2,1)
variable r
variable eta
variable lambda

minimize eta

subject to

[lambda*eye(noTx) - Qinv, Qinv*(d - g) - (lambda*bmax/2)*ones(noTx,1);((d-g)')*Qinv - (lambda*bmax/2)*ones(1,noTx), -trace(Qinv*G) + 2*(d')*Qinv*g - (d')*Qinv*d + eta] >= 0;
[G, g; g', 1] >= 0;
[eye(2), x; x', r] >= 0;

for i=1:noTx
    G(i,i) == r - 2*S(i,:)*x + S(i,1)^2 + S(i,2)^2;
    for j=i-1:-1:1
        G(i,j) >= abs(r - (S(i,:)+S(j,:))*x + S(i,:)*(S(j,:)'));
    end
end

lambda >= 0;

H*([x', r]') <= f;

cvx_end;
%%%%%%%%%%%%%%%%%%%%%%%%%% end cvx %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%