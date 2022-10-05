function x = ImpSDPR(po, sig, noTx, bmax)
%H. Chen, G. Wang, and N. Ansari, “Improved robust TOA-based localization via NLOS balancing parameter estimation,”
%IEEE Transactions on Vehicular Technology, vol. 68, no. 6, pp. 6177–6181, 2019

B = [eye(noTx), ones(noTx,1)];

rho_hat = bmax*ones(noTx,1);


rho_bar = bmax/4;

d = po(:,3);
Q = sig*eye(noTx);
Qinv = (1/sig)*eye(noTx);
S = po(:,1:2);
% S = S';

H = zeros(noTx,3);
H(:,1:2) = -2*S;
H(:,3) = 1;

f = d.^2 - S(:,1).^2 - S(:,2).^2;

%%%%%%%%%%%%%%%%%%%%%%%%%% cvx %%%%%%%%%%%%%%%%%%%%%%%%%%%

cvx_begin sdp quiet
cvx_solver sedumi

variable G(noTx+1,noTx+1)
variable g(noTx+1,1)
variable x(2,1)
variable r
variable eta
variable lambda

minimize eta

subject to

[Qinv, Qinv*(B*g - d);((B*g-d)')*Qinv, trace(B'*Qinv*B*G) - 2*(d')*Qinv*B*g + (d')*Qinv*d - eta] <= lambda*[eye(noTx), rho_bar*ones(noTx,1)-rho_hat;...
    (rho_bar*ones(noTx,1)-rho_hat)', noTx*rho_bar^2 - 2*rho_bar*sum(rho_hat)]; %(8): rho_bar, rho_hat

[G, g; g', 1] >= 0;%(12a)
[eye(2), x; x', r] >= 0;%(12b)

for i=1:noTx
    G(i,i) == r - 2*S(i,:)*x + S(i,1)^2 + S(i,2)^2;%(10a)
    for j=i-1:-1:1
        G(i,j) >= abs(r - (S(i,:)+S(j,:))*x + S(i,:)*(S(j,:)'));%(13)
    end
end

g(noTx+1) >= 0;%(10d)
g(noTx+1) <= rho_bar;
G(noTx+1,noTx+1) >= 0;%(10e)
G(noTx+1,noTx+1) <= rho_bar^2;

%lambda >= 0;

H*([x', r]') <= f;%(14)


cvx_end;
%%%%%%%%%%%%%%%%%%%%%%%%%% end cvx %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%