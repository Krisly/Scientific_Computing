function root = NewtonRaphson(func, x0, tolerance)
% Newton-Raphson root finding algorithm
% func : name of the function
% x0 : Initial point to search around
% tolerance : desired accuracy

    root = x0;
    xn = x0;
    
    iter = 10000;
    
    while abs(func(root)) > tolerance && iter > 0
        root = xn - func(xn)/FDM(func, 0.03, xn);
        xn = root;
        iter = iter - 1;
    end
end