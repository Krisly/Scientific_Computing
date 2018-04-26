function nma_validate_dimensions_1(grid)
% auxiliary function validates input dimensions consistent for 1D only

% An auxiliary function used by number of other function
% to validate that input dimensions are consistent for
% one grid only
%
% INPUT:
%   grid: 2D matrix
% OUTPUT:
%   no output. An exception is thrown on invalid dimension
%
% Verifies that grid is power of 2 plus one size. Verfies that
% grid is 2D and square
%
% by Nasser M. Abbasi
% Math 228a, UC Davis, Fall 2010
%


if ndims(grid) > 2
   error('::input number of dimensions too large');
end

[n,nCol]  = size(grid);
if n ~= nCol
   error('::input matrix u must be square');
end

valid_grid_points = log2(n-1);
if  round(valid_grid_points) ~= valid_grid_points
   error(':: invalid number of grid points value');
end
end