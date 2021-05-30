%% init
clearvars; clc; % close all
% addpath(...)
import casadi.*

%% Task 1
syms x real;
f = (x^2 - 5*x + 6)/(x^2 + 1);
df = diff(f,x);     % gradient
d2f = diff(df,x);   % hessian
% function handles
F  = matlabFunction(f);      
dF = matlabFunction(df);
d2F= matlabFunction(d2f);    
%% 1. visually inspected optimal value
figure(1); clf;
tl = tiledlayout('flow','TileSpacing','compact');
X_lims = { -1000:1:1000;
           -10:0.1:10;
           -1:0.01:5;
           2.35:.001:2.45 };
% visually inspected
xm = 2.41;
fm = F(xm);
tt = title(tl,['Visually inspected minimum with $x^* \approx ' num2str(xm) '$']);
tt.Interpreter = 'Latex'; 
for i = 1:4
    nexttile
    X = X_lims{i};
    plot(X,F(X));
    if i < 4
        ylim([-1 8])
    end
    hold on; grid on;
    % plot(X,dF(X));
    plot(xm,fm,'rx','MarkerSize',10,'LineWidth',2)
    ylabel('f(x)','Interpreter','latex');
    xlabel('x','Interpreter','latex');
    title(strcat(string(i),'.'));
end

%% 5. Use opti from CasADI to solve the optimization
opti = casadi.Opti();
x = opti.variable();
opti.minimize( (x^2 - 5*x + 6)/(x^2 + 1) );
opti.solver('ipopt');

sol_uncons = opti.solve();

opti.subject_to( x <= 4 );
opti.subject_to( x >= 0 );
sol_constr = opti.solve();

xstar_uncons = sol_uncons.value(x)
xstar_constr = sol_constr.value(x)

%% 5. Use provided simpleNewton function to solve for minimizer
K = 100; % max iter
% See possible trajectories for:
x_guess = [-0.5, -10, 0, xm, 100];
T = nan(length(x_guess),K + 1);
for  n = 1:length(x_guess)
    T(n,:) = simpleNewton(x_guess(n),dF,d2F,K);
end

figure(2); clf;
tl = tiledlayout('flow','TileSpacing','compact');
lineFormat = {'-o','-*','-s','-+','-x'};

nexttile;
X = -1:0.1:3;
plot(X,F(X),'b-','DisplayName','f(x)');
hold on; grid on;
for n = 1:length(x_guess)
    plot(T(n,:),F(T(n,:)),lineFormat{n},'DisplayName',['Newton x_0 = ' num2str(x_guess(n))]);
end
legend();
xlim([min(X),max(X)]);
ylim([-1, 8]);


