function [ f ] = nma_c2f( c )
%implements coarse to fine grid bilinear interpolation mapping

%--------------------------------------------------------
% This function is the prologation operator
% which implements coarse to fine grid bilinear
% interpolation mapping
%
% INPUT:
%   c the coarse 2D grid, spacing 2h
% OUTPUT:
%   f the fine 2D grid, spacing h
%
% EXAMPLE
% nma_c2f([0 0 0;0 1 0;0 0 0])
%          0         0         0         0         0
%          0    0.2500    0.5000    0.2500         0
%          0    0.5000    1.0000    0.5000         0
%          0    0.2500    0.5000    0.2500         0
%          0         0         0         0         0
% by Nasser M. Abbasi
% Math 228a, UC Davis, Fall 2010
%

nma_validate_dimensions_1(c);

[n,~]=size(c);
if n < 3
   error('nma_c2f:: inconsistent matrix size detected < 3');
end

n = 2*(n-1)+1;  % alocate space for fine grid
f = zeros(n);

i = 3:2:n-2;
j = 3:2:n-2;
f(i-1,j-1) = f(i-1,j-1) + (1/4)*c(ceil(i/2),ceil(i/2));
f(i-1,j)   = f(i-1,j)   + (1/2)*c(ceil(i/2),ceil(i/2));
f(i-1,j+1) = f(i-1,j+1) + (1/4)*c(ceil(i/2),ceil(i/2));

f(i,j-1)   = f(i,j-1)   + (1/2)*c(ceil(i/2),ceil(i/2));
f(i,j)     = f(i,j)     +    c(ceil(i/2),ceil(i/2));
f(i,j+1)   = f(i,j+1)   + (1/2)* c(ceil(i/2),ceil(i/2));

f(i+1,j-1) = f(i+1,j-1) + (1/4)* c(ceil(i/2),ceil(i/2));
f(i+1,j)   = f(i+1,j)   + (1/2)* c(ceil(i/2),ceil(i/2));
f(i+1,j+1) = f(i+1,j+1) + (1/4)* c(ceil(i/2),ceil(i/2));

end
