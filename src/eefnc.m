function xplus = eefnc(xdot, dt, x, u, p)

    xplus = x + xdot([], x, u, p) * dt;

end
 