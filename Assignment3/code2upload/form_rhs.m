function F = form_rhs(a, b, m, fun, u)
    h = (b-a)/(m);
    # Creating A
    #A = poisson9(m)

    # Creating F
    x =a:h:(b-h);

    y = fliplr(x);

    [X, Y] = meshgrid(x, y);

    # Error correction
    F = fun(X, Y);
    
    # Adding boundary conditions
    for i = 1:m
        F(1,i)   = F(1,i)   - (4*u(x(i), b) + u(x(i)-h, b) + u(x(i)+h, b))/(h^2*6);
        F(m-1,i) = F(m-1,i) - (4*u(x(i), a) + u(x(i)-h, a) + u(x(i)+h, a))/(h^2*6);
        F(i,1)   = F(i,1)   - (4*u(a, y(i)) + u(a, y(i)-h) + u(a, y(i)+h))/(h^2*6);
        F(i,m-1) = F(i,m-1) - (4*u(b, y(i)) + u(b, y(i)-h) + u(b, y(i)+h))/(h^2*6);
    end
    # The corners had boundary conditions added twice, so one is removed
    F(1, 1)      = F(1, 1)     + u(a,b)/(h^2*6);
    F(m-1, 1)    = F(m-1, 1)   + u(a,a)/(h^2*6);
    F(m-1, m-1)  = F(m-1, m-1) + u(b,a)/(h^2*6);
    F(1, m-1)    = F(1, m-1)   + u(b,b)/(h^2*6);
    
    %F = reshape(F,1,[]);
    
end