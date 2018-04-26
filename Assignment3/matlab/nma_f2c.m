function [ c ] = nma_f2c( f )
% restriction operator for fine grid to a coarse grid full weight mapping on 2D

%--------------------------------------------------------
% This function is the restriction operator nma_f2c()
% which implements the fine grid to a coarse grid
% full weight mapping on 2D
%
% INPUT:
%   f the fine 2D grid, spacing h
% OUTPUT:
%   c the corase 2D grid, spacing 2h
%
% Example   % ---- EXAMPLE 1
% f=[0 0 0 0 0;0 1 1 3 0;0 3 1 2 0;0 2 2 2 0;0 0 0 0 0];
% nma_f2c(f)
% ans =
%          0         0         0
%          0    4.6667         0
%          0         0         0
% by Nasser M. Abbasi
% Math 228a, UC Davis, Fall 2010
%

nma_validate_dimensions_1(f);
[n,~]=size(f);
if n <= 3
   error('nma_f2c::input matrix too small to restrict');
end

mask = (1/16)*[1 2 1;2 4 2;1 2 1]; % mapping mask, bilinear
for i=3:2:n-2
   for j=3:2:n-2
      f(i,j)=sum(sum(f(i-1:i+1,j-1:j+1).*mask));
   end
end

newn = ceil(n/2);
c    = zeros(newn);
c(2:end-1,2:end-1) = f(3:2:n-2,3:2:n-2);
end
