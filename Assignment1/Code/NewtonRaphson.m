function root = NewtonRaphson(func, x0, tolerance)
    root = x0;
    xn = x0;
    
    iter = 10000;
    
    while abs(func(root)) > tolerance && iter > 0
        root = xn - func(xn)/FDM(func, 0.03, xn);
        xn = root;
        iter = iter - 1;
    end
end