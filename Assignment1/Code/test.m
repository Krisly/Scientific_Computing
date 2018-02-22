close all, clear all, clc

lambda = 1;
fun = @(t, y, param) 4*t*sqrt(y); % y; %  %(4*t*sqrt(y));
true = @(t) (t.^2+1).^2; %t.^4; %2*exp(t); %(t.^2+1).^2; %8*exp(t); %(t.^2+1).^2;%(t.^4);%(t.^4 + 2.75681*t.^2 + 1.9);
y0 = 1;

steps = 1e-2:1e-3:1e-1;
timespan = [0 1];

maxErrorsERK4 = zeros(size(steps));
maxErrorsLMsolver = zeros(size(steps));

i = 1;
for h = steps
    nsteps = round((timespan(2) - timespan(1))/h);
    
    [tout2,yout2] = ERK4( fun, timespan, nsteps, y0, 0 );
    [tout1,yout1] = LMsolver2( fun, timespan, nsteps, y0, 0 );

    maxErrorsERK4(i) = max(abs(true(tout2)-yout2));
    maxErrorsLMsolver(i) = max(abs(true(tout1)-yout1));
    i = i + 1;
end

[tout2,yout2] = ERK4( fun, timespan, 100, y0, 0 );
[tout1,yout1] = LMsolver2( fun, timespan, 100, y0, 0 );

set(0,'defaultfigurecolor',[1 1 1])

figure
plot(tout1,yout1,tout2, true(tout2),'--', 'LineWidth', 1.5)
grid
xlim([timespan(1) timespan(2)])
legend('LMsolver','analytical', "Location", "NorthWest")
xlabel('t')
ylabel('y(t)')

figure
loglog(steps, maxErrorsLMsolver, steps, (steps.^4), '--', steps, (steps.^5), '--', steps, (steps.^6), '--', 'LineWidth', 1.5)
grid
xlim([min(steps) max(steps)])
legend('log|u_{LMS}(x)-u(x)|','O(h^4)','O(h^5)','O(h^6)', "Location", "NorthWest")
xlabel("log(h)")
ylabel("Error")