function [u] = smooth2d(u,f)

n = size(u,1);
h = 1/(n-1);

i = 2:n-1;
j = 2:n-1;
u(i,j) = (1/4)*( u(i-1,j) + u(i+1,j) + u(i,j-1) + u(i,j+1) - h^2 * f(i,j) );

end