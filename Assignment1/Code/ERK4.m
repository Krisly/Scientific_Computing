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
    
    %k(1) = func(tn, yn, param);
    %k(2) = func(tn + c(2)*h, yn + h*a(2,1)*k(1), param);
    %k(3) = func(tn + c(3)*h, yn + h*(a(3,1)*k(1) + a(3,2)*k(2)), param);
    %k(4) = func(tn + c(4)*h, yn + h*(a(4,1)*k(1) + a(4,2)*k(2) + a(4,3)*k(3)), param)

    yout = y0;
    
    for n = 1:(nsteps-1)
        tn = tout(n);
        yn = yout(n);
        
        k = 0;
        for i = 1:1:q
            %DGB [i,yn,tn + c(i)*h, yn + h*sum(a(i,1:(i-1)).*k)]
            k(i) = func( tn + c(i)*h, (yn + h*sum(a(i,1:(i-1)).*k)), param);
        end
        
        %k(1) = func(tn, yn, param);
        %k(2) = func(tn + c(2)*h, yn + h*a(2,1)*k(1), param);
        %k(3) = func(tn + c(3)*h, yn + h*(a(3,1)*k(1) + a(3,2)*k(2)), param);
        %k(4) = func(tn + c(4)*h, yn + h*(a(4,1)*k(1) + a(4,2)*k(2) + a(4,3)*k(3)), param);
        
        yout(n + 1) = yn + h*sum(b.*k);
        %DBG yn + h*sum(b.*k)
        %DBG k
        %DBG if ~isreal(yn + h*sum(b.*k))
        %DBG     break
        %DBG end
        %yout(n + 1) = yn + (1/6)*(k(1)+2*k(2)+2*k(3)+k(4))*h;
    end
end

%[t,y] = ERK4(@(t,y,param) 4*t*sqrt(y) , 0, 1.6, 1, 0)