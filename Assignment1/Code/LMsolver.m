function [tout,yout] = LMsolver( func, tspan, nsteps, y0, param ) 
% func : name of the function for computing the right hand side. 
% tspan : [start, end] times. 
% n : number of steps to take starting form tspan(1) to finishing at tspan(2) 
% y0 : initial solution 
% param : a vector array of parameters that may be transferred and used in func. 
% tout : vector of time values corresponding to steps taken. 
% yout : vector of solution values corresponding to steps taken.

    h = (tspan(2) - tspan(1))/(nsteps-1);
    %h2 = (tspan(2) - tspan(1))/(nsteps);
    tout = linspace(tspan(1),tspan(2),nsteps);
    yout = zeros(1,nsteps);
    yout(1) = y0;

    options = optimset('FunValCheck','off','TolX',10^-4);
    for n = 1:(nsteps-1)
        tn = tout(n);
        yn = yout(n);
        
        if n-2 < 1
        % first steps, need to use another method to start
            [t,y] = ERK4(func, [tn tn+h], 2, yn, param);
            yout(n+1) = y(2);
        else
        % subsequent steps, normal operation
        
            %% take 1, our method
            lmm = @(u) (( yout(n-1) + 1/3*h*func(tout(n-1),...
                       yout(n-1), param) + 4/3*h*func(tout(n),...
                       yout(n), param) + 1/3*h*func(tout(n+1), u,...
                       param)) - (u));
                   
            %sll = fzero(lmm,[1 10^300],options);
            sll  = NewtonRaphson(lmm,yn,10^-7);
            %sll = fmincon(lmm, yn, [], [], [], [], 0, Inf);
            %disp((sll))
            yout(n+1) = sll;
            
            %yout(n+1) = yout(n-1) + 1/3*h*func(tout(n-1), yout(n-1), 
            %param) + 4/3*h*func(tout(n), yout(n), param) + 
            %1/3*h* func(tout(n+1), tout(n+1).^4 + 
            %2.75681*tout(n+1).^2 + 1.9, param);
            %tout2.^4 + 2.75681*tout2.^2 + 1.9
            
            %% take 1.5
            
            %% take 2, trapezoidal for verification
            %lmm = @(u) (  u - yout(n) - h/2*func(tout(n+1), u, param)
            %- h/2*func(tout(n), yout(n), param)  );
            %yout(n+1) = NewtonRaphson(lmm, yn, 1e-20);
            
            %% take 3, r=2 adams-moulton
            %lmm = @(u) (  u - yout(n) - h/2*( func(tout(n), yout(n),...
            %param) + func(tout(n+1), u, param) )  );
            %yout(n+1) = NewtonRaphson(lmm, yn, 1e-20);
            
            %% take 4, r=2 adams-bashforth (explicit)
            %yout(n+1) = yout(n) + h/2*( -func(tout(n-1), yout(n-1),...
            %param) + 3*func(tout(n), yout(n), param) );
            
            %% take 5, r=1 adams-bashforth (explicit)
            %yout(n+1) = yout(n) + h*func(tout(n), yout(n), param);
            
            
            % uncomment to check if NewtonRaphson finds the root
            %disp([lmm(yout(n+1)), yout(n+1)])
        end
        
    end
end