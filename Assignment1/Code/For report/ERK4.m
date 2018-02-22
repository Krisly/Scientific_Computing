function [tout,yout] = ERK4( func, tspan, nsteps, y0, param )
% Explicit fourth order Runge-Kutta for one step
% func : name of the function
% tn : current point in time
% h : step
% yn : current solution
% param : a vector array of parameters that may be transferred and used in func
% tout : time at the step taken
% yout : solution at the step taken
    c = [0 1/2 1/2 1];
    a = [0 0 0 0; 1/2 0 0 0; 0 1/2 0 0; 0 0 1 0];
    b = [1/6 1/3 1/3 1/6];
    q = length(c);
    
    tout = linspace(tspan(1),tspan(2),nsteps);
    h = (tspan(2) - tspan(1))/(nsteps-1);
    
    yout = y0;
    
    for n = 1:(nsteps-1)
        tn = tout(n);
        yn = yout(n);
        
        k = 0;
        for i = 1:1:q
            k(i) = func( tn + c(i)*h, (yn + h*sum(a(i,1:(i-1)).*k)), param);
        end
        
        yout(n + 1) = yn + h*sum(b.*k);
    end
end