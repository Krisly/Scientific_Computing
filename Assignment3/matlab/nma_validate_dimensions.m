function  nma_validate_dimensions(vh,fh)
% auxiliary function used by other function to validate input dimensions are consistent.

% INPUT:
%   vh: 2D matrix, the solution grid
%   fh: 2D matrix, the RHS force field
% OUTPUT:
%   no output. An exception is thrown on invalid dimension
%
% Verifies that grid is power of 2 plus one size. Verfies that
% grids are 2D and square, and that dimensions match.
%
% by Nasser M. Abbasi
% Math 228a, UC Davis, Fall 2010
%


if ndims(vh) > 2
   error('::input number of dimensions too large');
end

[n,nCol]  = size(vh);
if n ~= nCol
   error('::input matrix u must be square');
end

[nRowf,nColf]  = size(fh);
if nRowf  ~= nColf
   error('::input matrix f must be square');
end

if nRowf ~= n
   error('::input matrix u,f must be same size');
end

valid_grid_points = log2(n-1);
if  round(valid_grid_points) ~= valid_grid_points
   error(':: invalid number of grid points value');
end

end