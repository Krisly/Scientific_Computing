function out = FDM(func, step, x0)
% optimal step = 0.03
    coefficients = [-1/60 3/20 -3/4 0 3/4 -3/20 1/60]./step^1;
	steps = [-3,-2,-1,0,1,2,3].*step;
    
    out = sum(coefficients .* arrayfun(func,(x0 + steps))); 
end