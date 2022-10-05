function ppnext = POCSstep(pDisc,pp,l)
ppnext = zeros(2,1);
rDisc = pDisc(1);
cDisc = pDisc(2);
radi = pDisc(3);

ppR = pp(1);
ppC = pp(2);

dist = sqrt((ppR-rDisc)^2 + (ppC-cDisc)^2);

if dist < radi
    r = ppR;
    c = ppC;
else
    
    diffR = ppR - rDisc;
    diffC = ppC - cDisc;
    diffR = diffR * radi / dist;
    diffC = diffC * radi / dist;
    r = rDisc + diffR;
    c = cDisc + diffC;   
end

ppnext(1) = ppR + l*(r-ppR);
ppnext(2) = ppC + l*(c-ppC);

end