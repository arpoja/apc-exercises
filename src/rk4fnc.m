function xplus = rk4fnc(xdot,dt,x,u,p)

    k1 = xdot([], x            , u, p);
    k2 = xdot([], x + dt/2 * k1, u, p);
    k3 = xdot([], x + dt/2 * k2, u, p);
    k4 = xdot([], x + dt   * k3, u, p);

    xplus = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4);

end