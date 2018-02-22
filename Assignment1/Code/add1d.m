close all

stepRange = (1e-2: 1e-5 :1e-1);
%s = 2:1:5;
%stepRange = 1./2.^s;

stencil1Output = zeros(size(stepRange));
stencil2Output = zeros(size(stepRange));

x = pi/2;

for i = 1:length(stepRange)
  h = stepRange(i);
  coefficients1 = [35/12 -26/3 19/2 -14/3 11/12]./h^2;
  pointLocations1 = [0 1 2 3 4].*h;
  
  stencil1Output(i) = sum(coefficients1 .* F(x + pointLocations1));
  
  coefficients2 = [-1/12 4/3 -5/2 4/3 -1/12]./h^2;
  pointLocations2 = [-2 -1 0 1 2].*h;
  
  stencil2Output(i) = sum(coefficients2 .* F(x + pointLocations2));
end

set(0,'defaultfigurecolor',[1 1 1])

figure
plot(stepRange, stencil1Output, stepRange, ...
    ones(size(stepRange)).*F2d(x), 'LineWidth', 1.5)
ylabel("f''(pi/2)")
xlabel("step size")
legend("FDM", "Analytical", "Location", "NorthWest")
axis([1e-2 0.1 -Inf Inf])
grid

figure
loglog(stepRange, abs(ones(size(stepRange)).* F2d(x) - stencil1Output),...
    stepRange, stepRange.^3, 'LineWidth', 1.5)
xlabel("log(h)")
ylabel("Error")
legend("log|D_4u(x)-u(x)|", "O(h^3)", "Location", "NorthWest")
grid

figure
plot(stepRange,stencil2Output, stepRange, ones(size(stepRange)).* F2d(x),...
    'LineWidth', 1.5)
ylabel("f''(pi/2)")
xlabel("step size")
legend("FDM", "Analytical", "Location", "NorthWest")
axis([1e-2 0.1 -Inf Inf])
grid

figure
loglog(stepRange, abs(ones(size(stepRange)).* F2d(x) - stencil2Output),...
    stepRange, stepRange.^4, 'LineWidth', 1.5)
xlabel("log(h)")
ylabel("Error")
legend("log|D_4u(x)-u(x)|", "O(h^4)", "Location", "NorthWest")
grid

function y = F (x)
  y = exp(-x.^2);%sin(x);
end

function y = F2d (x)
  y = -2.*exp(-x.^2)+4.*x.^2.*exp(-x.^2);%-sin(x);
end