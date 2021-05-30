%% init
clearvars; clc; % close all
% addpath(...)
import casadi.*

%% Task 3
N = 8;
opti = casadi.Opti();
x = opti.variable(N + 1);
% build function and constraints
ys = {};cs = {};
for n = 1:N
    ys{end + 1} = 100*(x(n + 1) - x(n)^2)^2 + (1 - x(n))^2;
    cs{end + 1} = (x(n) - 1)^2;
end
cs{end + 1} = (x(N + 1) - 1)^2;
% optimize
opti.minimize(   sum([ys{:}]) );
opti.subject_to( sum([cs{:}]) <= 2 );
opti.solver('ipopt');
% sol = opti.solve();

%% Find optimal value
K = 25;
% pull initials from uniform N+1 dimensional hypersphere
% https://se.mathworks.com/matlabcentral/fileexchange/9443-random-points-in-an-n-dimensional-hypersphere?s_tid=answers_rc2-3_p6_MLT
x0 = randsphere(N+1,K,sqrt(2)) + 1;
xf = NaN(N+1,K);
for k = 1:K
    opti.set_initial(x,x0(:,k));
    sol = opti.solve();
    xf(:,k) = sol.value(x);
end
disp('All initials:')
disp(x0)
disp('All solutions:')
disp(xf)
