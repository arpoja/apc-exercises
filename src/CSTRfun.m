function xdot = CSTRfun(t, x, u, p)

% params
F_in    = p(1);
T_in    = p(2);
C_A_in  = p(3);
C_B_in  = p(4);
r       = p(5);
k_0     = p(6);
E_per_R = p(7);
U       = p(8);
rho     = p(9);
C_p     = p(10);
DH      = p(11);
if length(p) == 13
    F   = p(12);
    T_c = p(13);
else
    F   = u(1);
    T_c = u(2);
end
% states
C_A  = x(1);
T    = x(2);
h    = x(3);
% precalc
A = pi*r^2;
E = exp(- E_per_R / T);
% funcs
dCA = F_in * (C_A_in - C_A) / (A * h) - ...
      k_0 * E * C_A;

dT  = F_in * (T_in - T) / (A * h) - ...
      DH / (rho * C_p) * k_0 * E * C_A + ...
      2*U / (r * rho * C_p) * (T_c - T);
  
dh  = (F_in - F) / (A);


xdot = [ dCA; dT; dh ];

end
