function r = nma_find_residue( u , f)
%calculates residue

%------------------------------------
% This function is called from number of places to obtain
% the residue
%
% INPUT:
%   u: 2D grid represent the current solution
%   f: 2D grid represent the RHS, the force field
% OUTPUT:
%   r: 2D grid which represent the residual, which is f-Au
%
% by Nasser M. Abbasi
% Math 228a, UC Davis, Fall 2010
%

%nma_validate_dimensions(u,f);

[n,~]=size(u);
h = 1/(n-1);

i=2:n-1;
j=2:n-1;

r = zeros(n);

r(i,j) = f(i,j) -  (1/h^2)*( u(i,j-1) + u(i,j+1) + ...
   u(i+1,j) + u(i-1,j) - 4*u(i,j) ) ;

end


