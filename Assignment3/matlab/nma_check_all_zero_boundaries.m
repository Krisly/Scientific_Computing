function nma_check_all_zero_boundaries(grid)
%auxiliary function to validate boundary conditions

%----------------------------------------------
% An auxiliary function used by number of other function
% to validate that input is consistent.
%
% INPUT:
%   grid: 2D matrix
% OUTPUT:
%   no output. An exception is thrown on invalid dimension
%
% Verifies that grid is power of 2 plus one size. Verfies that
% grid is 2D and square, and also verifies that boundary conditions
% are zero.
%
% by Nasser M. Abbasi
% Math 228a, UC Davis, Fall 2010
%

if any(grid(:,1)~=0)
   error(':: inconsistency, grid(:,1) not all zero');
elseif  any(grid(:,end)~=0)
   error(':: inconsistency, f(:,end) not all zero');
elseif  any(grid(1,:)~=0)
   error(':: inconsistency, f(1,:) not all zero');
elseif  any(grid(end,:)~=0)
   error(':: inconsistency, f(end,:) not all zero');
end

end
