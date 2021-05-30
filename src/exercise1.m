%% Exercise 1 - Explicit integrators
% addpath('~/path_to/casadi_folder') % 
import casadi.*

clear; clc; % close all;

%% Define initial value problem
% Integration parameters
Tf = 60;
K  = 120;
deltaTime = Tf / K;
t = deltaTime*(0:K-1);

% Initial value
x0 = [0.878; 324.5; 0.659];
u0 = [0.1; 300]; 

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
        k0; E_per_R; U; rho; cons_p; deltaH; u0 ]; % treat control as a parameter

% load function into a handle to speed up simulation
CSTR = @(t,x,u,p) CSTRfun(t,x,u,p);

%% ODE15s
% User friendly interface to integrate ODEs/DAEs of type xplus = f(t,x,u|px) 
% from initial conditon x(0), over [0,Tf], subjected to controls u(t)
tic
[~, ode15sX] = ode15s(@(t,x) CSTR(t,x,[],px), (0:K-1)*deltaTime, x0);
disp(strcat('ODE15s:',num2str(toc),'s'));

%% ODE45
% User friendly interface to integrate ODEs of type xplus = f(t,x,u|p) 
% from initial conditon x(0), over [0,Tf], subjected to controls u(t)
% Dormand-Prince version of the Runge-Kutta integrator
opt = odeset('RelTol',1e-9);
tic
[~, ode45X] = ode45(@(t,x) CSTR(t,x,[],px), (0:K-1)*deltaTime, x0, opt);
disp(strcat('ODE45:',num2str(toc),'s'));
    
%% Explicit Euler
eeX_plain = nan(3,K); eeX_plain(:,1) = x0;
tic
for k = 2:K
 eeX_plain(:,k) = eeX_plain(:,k-1) + ...
     CSTR([], eeX_plain(:,k-1), [], px) * deltaTime;
end
disp(strcat('EE_plain:',num2str(toc),'s'));
 
% Using a function handle
eeX_fnc = nan(3,K);   eeX_fnc(:,1) = x0;
tic
for k=2:K
 eeX_fnc(:,k) = eefnc(CSTR, deltaTime, eeX_fnc(:,k-1), [], px);
end
disp(strcat('EE_fnc:',num2str(toc),'s'));

    
%% Explicit Runge-Kutta order-4
rk4X_plain = nan(3,K); rk4X_plain(:,1) = x0;
tic
for k = 2:K
    
 kappa(:,1) = CSTR([], rk4X_plain(:,k-1), [], px);
 kappa(:,2) = CSTR([], rk4X_plain(:,k-1) + kappa(:,1) * deltaTime / 2, [], px);
 kappa(:,3) = CSTR([], rk4X_plain(:,k-1) + kappa(:,2) * deltaTime / 2, [], px);
 kappa(:,4) = CSTR([], rk4X_plain(:,k-1) + kappa(:,3) * deltaTime, [], px);
 
 rk4X_plain(:,k) = rk4X_plain(:,k-1) + ...
           deltaTime / 6 * (kappa(:,1) + 2*kappa(:,2) + 2*kappa(:,3) + kappa(:,4));
end
disp(strcat('RK4_plain:',num2str(toc),'s'));


% Using a function handle      
rk4X_fnc = nan(3,K);   rk4X_fnc(:,1) = x0;
tic
for k = 2:K
 rk4X_fnc(:,k) = rk4fnc(CSTR, deltaTime, rk4X_fnc(:,k-1), [], px);
end
disp(strcat('RK4_fnc:',num2str(toc),'s'));

    
%% CasADi

f = @(x) CSTR([], x, [], px);

x  = MX.sym('x', 3);
u  = MX.sym('u', 0);
px = MX.sym('px',3);
p  = [u; px];
    
ODE = struct('x', x, 'p', p, 'ode', f(x));

%%%%% RK4 by CasADi
rk4F = integrator('F', 'rk', ODE, struct('tf',deltaTime));
rk4X_casadi = nan(3,K); rk4X_casadi(:,1) = x0;
tic
for k=2:K
 rk4F_res = rk4F('x0', rk4X_casadi(:,k-1));
 rk4X_casadi(:,k) = full(rk4F_res.xf');     
end
disp(strcat('RK4_casadi:',num2str(toc),'s'));

%%%%% CVODES from CasADi
cvodesF = integrator('F', 'cvodes', ODE, struct('tf',deltaTime));
cvodesX_casadi = nan(3,K); cvodesX_casadi(:,1) = x0;
tic
for k=2:K
 cvodesF_res = cvodesF('x0', cvodesX_casadi(:,k-1));
 cvodesX_casadi(:,k) = full(cvodesF_res.xf');
end
disp(strcat('CVODES:',num2str(toc),'s'));

%%%%% IDAS from CasADi
idasF = integrator('F', 'idas', ODE, struct('tf',deltaTime));
idasX_casadi = nan(3,K); idasX_casadi(:,1) = x0;
tic; 
for k=2:K
 idasF_res = idasF('x0', idasX_casadi(:,k-1));
 idasX_casadi(:,k) = full(idasF_res.xf');
end
disp(strcat('IDAS:',num2str(toc),'s'));


%% Plotting
figure(1); 
clf;
tl = tiledlayout(3,1);
tl.TileSpacing = 'none';
tl.Padding = 'none';
tstr = strcat('Integrator comparisons: $u(k) = u = [',string(u0(1)),',',string(u0(2)),']$');
tt = title(tl,tstr);
tt.Interpreter = 'Latex'; 

ms = 4.0; lw = 1.0; fs = 12;
units = {'[m$^3$/min]','[K]','[m]'};
for i = 1:3
    nexttile;
    hold on;
      stairs(t,ode15sX(:,i),       '.-', 'DisplayName', 'ode15s',    'MarkerSize', ms);
      stairs(t,ode45X(:,i),        '.-', 'DisplayName', 'ode45',     'MarkerSize', ms);
      stairs(t,eeX_plain(i,:),     '.-', 'DisplayName', 'euler',     'MarkerSize', ms);
      stairs(t,rk4X_plain(i,:),    '.-', 'DisplayName', 'rk4',       'MarkerSize', ms);
      stairs(t,rk4X_casadi(i,:),   '.-', 'DisplayName', 'rk4 CasADI','MarkerSize', ms);
      stairs(t,cvodesX_casadi(i,:),'.-', 'DisplayName', 'CVODES',    'MarkerSize', ms);
      stairs(t,idasX_casadi(i,:),  '.-', 'DisplayName', 'IDAS',      'MarkerSize', ms);
    hold off;
    % title(strcat('$x_',string(i),'$'),'Interpreter','Latex')
    if i == 2
        legend('Location','eastoutside','Interpreter','Latex');
    end
    ylabel(strcat('$x_',string(i),'(t)$ ',units{i}),'Interpreter','Latex')
    if i == 3
        xlabel('$t$ [min]','Interpreter','Latex')
    end
    set(gca,'TickLabelInterpreter','Latex','FontSize',fs)
end
% 
% figure(2);
% subplot(1,4,1:3)
% hold on;
%  stairs(t,ode15sX(:,1), '.-', 'DisplayName', '$x_1$ ode15s', 'MarkerSize', ms)
%  stairs(t,ode15sX(:,2), '.-', 'DisplayName', '$x_2$ ode15s', 'MarkerSize', ms)
%  stairs(t,ode15sX(:,3), '.-', 'DisplayName', '$x_3$ ode15s', 'MarkerSize', ms)
% hold off; 
% legend('Location','southwest','Interpreter','Latex');
% ylabel('$x(t)$','Interpreter','Latex')
% xlabel('$t$','Interpreter','Latex')
% set(gca,'TickLabelInterpreter','Latex','FontSize',fs)
% 
% subplot(1,4,4)
% hold on
%  plot3(ode45X(:,1), ode45X(:,2), ode45X(:,3), 'k.-', 'MarkerSize', ms)
%  plot3(x0(1),x0(2),x0(3),'o','MarkerSize',2*ms)
%  %axis equal
% hold off
% view(3)
% zlabel('$x_3(t)$','Interpreter','Latex')
% ylabel('$x_2(t)$','Interpreter','Latex')
% xlabel('$x_1(t)$','Interpreter','Latex')
% set(gca,'TickLabelInterpreter','Latex','FontSize',fs)
% grid on