close all
u=@(x,y) exp(pi*x).*sin(pi*y)+0.5*(x.*y).^2;
f=@(x,y) x.^2+y.^2;
m=2^6+1;
a = 0;
b = 1;
U =zeros(m,m);

F =form_rhs(a,b,m,f,u); %% TODO: Form the right-hand side

QQ = Vcycle(U,F,3,2);



 h = (b-a)/(m + 1);
 % Creating A 
 A =  0; %lap_kron(m);
 % Creating F 
 x = (a+h):h:(b-h);
 y = fliplr(x);
 [X, Y] = meshgrid(x, y);
 
 figure('Position',[0,0,1000,1000])
 surf(X,Y,QQ)
 set(gca, "fontsize", 20)
 title('V-cycle solution', "fontsize", 20)

 figure('Position',[0,0,1000,1000])
 surf(X,Y,u(X,Y))
 set(gca, "fontsize", 20)
 title('Real solution', "fontsize", 20)
