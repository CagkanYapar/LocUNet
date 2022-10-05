function x = SDPNonCoop(po, sig, noTx)
%R. M. Vaghefi, J. Schloemann, and R. M. Buehrer, “NLOS mitigation in TOA-based localization using semidefinite
%programming,” in 2013 10th Workshop on Positioning, Navigation and Communication (WPNC), 2013, pp. 1–6.

d = po(:,3);
v = 1./((sig*d).^2);
u = 4*sig*d;

S = po(:,1:2);

%%%%%%%%%%%%%%%%%%%%%%%%%% cvx %%%%%%%%%%%%%%%%%%%%%%%%%%%

cvx_begin sdp quiet
cvx_solver sedumi

variable c(noTx,1)
variable h(noTx,1)
variable x(2,1)
variable z

minimize sum(v.* ((d.^2 - h - c).^2) + delta*(c.^2))

subject to

[eye(2), x; x', z] >= 0;

for i=1:noTx
    h(i) == [S(i,:), -1]*[eye(2), x;x', z]*[S(i,:), -1]';
    [eye(2), S(i,:)' - x;S(i,:) - x', d(i)^2 + u(i)] >= 0;
    c(i) >= 0;
   
end

cvx_end;
%%%%%%%%%%%%%%%%%%%%%%%%%% end cvx %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%