%% Exercise 3 - Optimal Control
% addpath('~/path_to/casadi_folder') % 
import casadi.*

clear; clc; % close all;
rng(769684,'twister');
%% Define initial value problem
% Integration parameters
Tf = 10;
K  = 250; 
deltaTime = Tf / K;
t = deltaTime*(0:K-1);
% RK4 "inner loop" integrator params
K_rk4 = 25;
dt = deltaTime / K_rk4;

% Initial value and references
xref = [0.878; 324.5; 0.659];
x0   = diag([0.9 1.1 1.0]) * xref;
uref = [0.1; 300]; 
umax = 1.15 * uref;
% umax = [0.115; 301];
UMAX = repmat(umax',1,K);
umin = 0.85 * uref;
% umin = [0.085; 299];
UMIN = repmat(umin',1,K);

% sizes
Nx = length(xref);
Nu = length(uref);

% Model parameter
flow_in   = 0.1;
temp_in   = 350;
cons_A_in = 1;
cons_B_in = 0;
radius    = 0.219;
k0        = 7.2E10;
E_per_R   = 8750;
U         = 54.94;
rho       = 1000;
cons_p    = 0.239;
deltaH    = -5E4;
px = [  flow_in; temp_in; cons_A_in; cons_B_in; radius;...
        k0; E_per_R; U; rho; cons_p; deltaH]; % treat control as a parameter

% LQR params
Q_sec  = eye(Nx);
Qf_sec = 1e6*Q_sec;
R_sec  = eye(Nu);

Q_sim  = diag([100 10 0]);
Qf_sim = 1e6*Q_sec;
R_sim  = 0.1*eye(Nu);



% load function into a handle to speed up simulation
CSTR = @(t,x,u,p) CSTRfun(t,x,u,p);



%% Sequential
% construction
x = MX.sym('x',Nx);
u = MX.sym('u',Nu);

xplus = x;
for i = 1:K_rk4
    xplus = rk4fnc(CSTR, dt, xplus, u, px);
end
F = Function('F', {x,u}, {xplus});

U = MX.sym('U',Nu,K);
X = F(x0, U(:,1));
g = X';
for k = 1:K - 1
    X = [X F(X(:,k), U(:,k + 1))];
    g = [g, X(:,k)'];
end

sumLx = 0; sumLu = 0;
for k = 1:K
    sumLx = sumLx + 1/2*(X(:,k) - xref)' * Q_sec * (X(:,k) - xref);
    sumLu = sumLu + 1/2*(U(:,k) - uref)' * R_sec * (U(:,k) - uref);
end
E = (X(:,end) - xref)' * Qf_sec * (X(:,end) - xref);
J = sumLx + sumLu + E;

% NLP
vecU = reshape(U, 1, Nu*K);
sequentialCSTR = struct('x', vecU, 'f', J, 'g', g);
solver = nlpsol('solver', 'ipopt', sequentialCSTR);
% U0 = repmat(uref',1,K) + rand(1,Nu*K);
U0 = rand(1,Nu*K);
solU = solver('x0',U0, 'lbx', UMIN, 'ubx', UMAX, ...
    'lbg', zeros(1,Nx*K), 'ubg', Inf(1,Nx*K));
starU = full(reshape(solU.x, Nu, K));
FX = Function('FX', {U}, {X});
starX = [x0, full(FX(starU))];

%% Plotting
% close all
fs = 15;
ms = 12;
lw = 02;

units = {'[m$^3$/min]','[K]','[m]','[m$^3$/min]','[K]'};

figure(1); clf
tl = tiledlayout(5,1,'tilespacing','none');
tt = title(tl,'Sequential solution');
tt.Interpreter = 'latex';
tt.FontSize = fs + 5;

for ix = 1:Nx
 % subplot(Nx + Nu,1,ix); 
 nexttile;
 hold on
    stairs(deltaTime*(1:K+1), starX(ix,:), '-', 'Linewidth', lw)
    line(deltaTime*[1 K+1],[xref(ix) xref(ix)],'LineStyle','--','Color','k'); 
    ylabel(strcat('$x_',num2str(ix),'\ $', units{ix}),'FontSize',fs,...
        'Interpreter','latex')
    if ix == Nx
        % xlabel('Time','FontSize',fs)
    end
 hold off; grid on;
 xlim([0, max(t)+ deltaTime]);
 set(gca,'FontSize',fs)
 end 
 
% figure(2); clf
for iu = 1:Nu
 % subplot(Nu + Nx,1,Nx + iu)
 nexttile;
 hold on
    l1 = stairs(deltaTime*(1:K), starU(iu,:), '.-', 'MarkerSize', ms);
    l2 = line(deltaTime*[1 K],[uref(iu) uref(iu)],'LineStyle','--','Color','k');
    l3 = line(deltaTime*[1 K],[umax(iu) umax(iu)],'LineStyle','--','Color','red');
    line(deltaTime*[1 K],[umin(iu) umin(iu)],'LineStyle','--','Color','red');
    ylabel(strcat('$u_',num2str(iu),'$ ', units{iu + Nx}),'FontSize',fs,...
        'Interpreter','latex')
    if iu == Nu
        xlabel('Time [min]','FontSize',fs, ...
            'Interpreter','latex')
    end
 hold off; grid on;
  xlim([0, max(t) + deltaTime]);
 set(gca,'FontSize',fs)
end 

legend([l1 l2 l3],'Variable','Reference','Min/Max',...
     'Interpreter','latex','Location','east')
 

%% Simultaneous
wx = [];
wu = [];
g  = [];

sumLx = 0;
sumLu = 0;
E = 0;
J = 0;

x_k = x0;
for k = 1:K
    sumLx = sumLx + 1/2*(x_k - xref)' * Q_sim * (x_k - xref);
    u_k = MX.sym(['u_', num2str(k)], Nu, 1);
    sumLu = sumLu + 1/2*(u_k - uref)' * R_sim * (u_k - uref);
    J = J + sumLx + sumLu;
    
    xplus_sim = F(x_k, u_k);
    
    x_k = MX.sym(['x_', num2str(k)], Nx, 1);
    wx = [wx, x_k];
    wu = [wu, u_k];
    
    g = [g; xplus_sim - x_k];
end
E = (x_k - xref)' * Qf_sim * (x_k - xref);
J = J + E;

% NLP
wVec = [reshape(wx, 1, Nx*K) reshape(wu, 1, Nu*K)];
simultaneousCSTR = struct('x', wVec, 'f', J, 'g', g);
solver = nlpsol('solver', 'ipopt', simultaneousCSTR);

w0 = rand(1,Nx*K + Nu*K);

LBX = [zeros(1,Nx*K) UMIN];
UBX = [Inf(1,Nx*K)   UMAX];

wSol = solver('x0', w0, 'lbg', 0, 'ubg', 0, 'lbx', LBX, 'ubx', UBX);
wxSol = full(reshape(wSol.x(1:Nx*K), Nx, K));
wuSol = full(reshape(wSol.x(1+Nx*K:end), Nu, K));

%% Plotting again
figure(2); clf
tl = tiledlayout(5,1,'tilespacing','none');
tt = title(tl,'Simultaneous solution');
tt.Interpreter = 'latex';
tt.FontSize = fs + 5;

 for ix = 1:Nx
 % subplot(Nx + Nu,1,ix); 
 nexttile;
 hold on
    stairs(deltaTime*(1:K), wxSol(ix,:), '-', 'Linewidth', lw)
    line(deltaTime*[1 K],[xref(ix) xref(ix)],'LineStyle','--','Color','k'); 
    ylabel(strcat('$x_',num2str(ix),'\ $', units{ix}),'FontSize',fs,...
        'Interpreter','latex')
    if ix == Nx
        % xlabel('Time','FontSize',fs)
    end
 hold off; grid on;
 xlim([0, max(t)+ deltaTime]);
 set(gca,'FontSize',fs)
 end 
 
% figure(4); clf
 for iu = 1:Nu
 % subplot(Nu + Nx,1,Nx + iu)
 nexttile;
 hold on
    l1 = stairs(deltaTime*(1:K), wuSol(iu,:), '.-', 'MarkerSize', ms);
    l2 = line(deltaTime*[1 K],[uref(iu) uref(iu)],'LineStyle','--','Color','k');
    l3 = line(deltaTime*[1 K],[umin(iu) umin(iu)],'LineStyle','--','Color','red');
    line(deltaTime*[1 K],[umax(iu) umax(iu)],'LineStyle','--','Color','red');
    ylabel(strcat('$u_',num2str(iu),'$ ', units{iu + Nx}),'FontSize',fs,...
        'Interpreter','latex')
    if iu == Nu
        xlabel('Time [min]','FontSize',fs, ...
            'Interpreter','latex')
    end
 hold off; grid on;
 xlim([0, max(t)+ deltaTime]);
 set(gca,'FontSize',fs)
 end 
 legend([l1 l2 l3],'Variable','Reference','Min/Max',...
     'Interpreter','latex','Location','east')
