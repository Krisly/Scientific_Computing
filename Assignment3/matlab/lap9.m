function [A, f] = lap9(a, b, m, fun, funlap, u)
 h = (b-a)/(m + 1);
 % Creating A 
 A =  0; %lap_kron(m);
 % Creating F 
 x = (a+h):h:(b-h);
 y = fliplr(x);
 [X, Y] = meshgrid(x, y);
 F = fun(X, Y);% + funlap(X, Y)*h^2/12;
 for i = 1:m
 F(1,i) = F(1,i) - (4*u(x(i), b) + u(x(i)-h, b) + u(x(i)+h, b))/(h^2*6);
 F(m,i) = F(m,i) - (4*u(x(i), a) + u(x(i)-h, a) + u(x(i)+h, a))/(h^2*6);
 F(i,1) = F(i,1) - (4*u(a, y(i)) + u(a, y(i)-h) + u(a, y(i)+h))/(h^2*6);
 F(i,m) = F(i,m) - (4*u(b, y(i)) + u(b, y(i)-h) + u(b, y(i)+h))/(h^2*6);
 end
 F(1, 1) = F(1, 1) + u(a,b)/(h^2*6);
 F(m, 1) = F(m, 1) + u(a,a)/(h^2*6);
 F(m, m) = F(m, m) + u(b,a)/(h^2*6);
 F(1, m) = F(1, m) + u(b,b)/(h^2*6);
 f = F;%(:);
 end