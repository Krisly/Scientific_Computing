u=@(x,y) exp(pi*x).*sin(pi*y)+0.5*(x.*y).^2;
f=@(x,y) x.^2+y.^2;
m=2^6+1;
U =zeros(m,m);
F =form_rhs(0,1,m,f,u); %% TODO: Form the right-hand side

QQ = nma_V_cycle(U,F,3,3,2);