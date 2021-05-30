%% init
clearvars; clc; % close all
% addpath(...)
import casadi.*

%% Task 2
syms x y real;
f = -20*exp(-.2*(sqrt(0.5*(x.^2 + y.^2)))) - ...
    exp(0.5*(cos(2*pi*x) + cos(2*pi*y))) + ...
    exp(1) + 20;
df = jacobian(f);
d2f = hessian(f);

g = x^2 + y^2 - 3;

% function handles
F = matlabFunction(f);
dF = matlabFunction(df);
d2F = matlabFunction(d2f);
G = matlabFunction(g);
jac = @(x) dF(x(1),x(2))';
hes = @(x) d2F(x(1),x(2));
ff  = @(x) F(x(1),x(2));

%% visualize
[X,Y] = meshgrid(-2:0.05:2, -2:0.05:2);
% feasible set
[cX,cY,cZ] = cylinder(sqrt(3),101);

figure(1); clf;
tl = tiledlayout('flow','TileSpacing','compact');
tt = title(tl,['Different visualizations of the objective function and the feasible set']);
tt.Interpreter = 'Latex'; 

nexttile;
hold on; grid on;
surf(X,Y,F(X,Y));
surf(cX,cY,cZ*4 + 4)
plot3(0,0,5E-1,'rx','MarkerSize',10,'LineWidth',2)
view(2);
title('1. Top down');
xlabel('x','Interpreter','latex');
ylabel('y','Interpreter','latex');
zlabel('f(x)','Interpreter','latex');

nexttile;
hold on; grid on;
surf(X,Y,F(X,Y));
surf(cX,cY,cZ*4 + 4)
plot3(0,0,F(0,0),'rx','MarkerSize',10,'LineWidth',2)
view(3);
title('2. Constrained region');
xlabel('x','Interpreter','latex');
ylabel('y','Interpreter','latex');
zlabel('f(x)','Interpreter','latex');

filterindex = G(X,Y) >= 0;
X(filterindex) = NaN;
Y(filterindex) = NaN;

nexttile;
hold on; grid on;
surf(X,Y,F(X,Y));
plot3(0,0,F(0,0),'rx','MarkerSize',10,'LineWidth',2)
view(3);
title('3. Reduced region');
xlabel('x','Interpreter','latex');
ylabel('y','Interpreter','latex');
zlabel('f(x)','Interpreter','latex');


%% 2.1 Optimization using provided simpleNewton.m
N = 16; K = 10;
x0 = nan(N,2);
xt = nan(N,2,K+1); % trajectories
% pick initial points from uniform disk
r  = sqrt(3)*sqrt(rand(N,1));
th = 2*pi*rand(N,1);
for n = 1:N
    x0(n,:) = [r(n).*cos(th(n)); r(n).*sin(th(n))]; 
    xt(n,:,:) = simpleNewton(x0(n,:), jac, hes, K);
end
% visualize
figure(2); clf;
s = surf(X,Y,F(X,Y));
hold on; grid on;
for n = 1:N
    xx = squeeze(xt(n,1,:));
    yy = squeeze(xt(n,2,:));
    zz = F(xx,yy);
    p1 = plot3(xx,yy,zz,'x-k','LineWidth',1.5,'MarkerSize',10);
    p2 = plot3(xx(1),yy(1),F(xx(1),yy(1)),'r*','LineWidth',1.5,'MarkerSize',15);
    p3 = plot3(xx(end),yy(end),F(xx(end),yy(end)),'g^','LineWidth',1.5,'MarkerSize',15);
end

l = legend([s p1 p2 p3],'Objective function','Newton trajectory','Initial point','Final point');
l.FontSize = 15;
tt = title('Optimization using Newton''s method','Interpreter','latex');
tt.FontSize = 20;
xlabel('x','Interpreter','latex','FontSize',15);
ylabel('y','Interpreter','latex','FontSize',15);
zlabel('f(x)','Interpreter','latex','FontSize',15);


%% 2.2 Optimization using CasADI instead - ipopt does not seem to converge here
% opti = casadi.Opti();
% x = opti.variable();
% y = opti.variable();
% 
% fnc = -20*exp(-0.2*(sqrt(0.5*(x^2 + y^2)))) - ...
%       exp(0.5*(cos(2*pi*x) + cos(2*pi*y))) + ...
%       exp(1) + 20;
% con = x^2 + y^2 <= 3;
% 
% opti.minimize(fnc);
% opti.subject_to(con);
% p_opts = struct('expand',true);
% s_opts = struct('max_iter',1e6);
% opti.solver('ipopt',p_opts,s_opts);
% 
% for n = 1:N
%     opti.set_initial(x,x0(n,1));
%     opti.set_initial(y,x0(n,2));
%     %sol{n} = opti.solve();    
% end


