close all
u=@(x,y) exp(pi*x).*sin(pi*y)+0.5*(x.*y).^2;
f=@(x,y) x.^2+y.^2;
m=2^6+1;
U =zeros(m,m);

F =form_rhs(0,1,m,f,u); %% TODO: Form the right-hand side
%[A, F] = lap9(0, 1, m, f, 0, u);

QQ = nma_V_cycle(U,F,3,3,2);

surf(QQ)

figure

 h = (b-a)/(m + 1);
 % Creating A 
 A =  0; %lap_kron(m);
 % Creating F 
 x = (a+h):h:(b-h);
 y = fliplr(x);
 [X, Y] = meshgrid(x, y);
 
 surf(u(X,Y))