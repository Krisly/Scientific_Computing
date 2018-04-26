function [u] = nma_relax(u,f,method)
% does one iteration relaxation, called by V cycle for multigrid solver 


%--------------------------------------------------------
% This function does one iteration relaxation, called by
% V cycle for multigrid solver of the 2D Poisson PDE On unit square.
%
% INPUT:
%   u: 2D square grid that represent current solution
%   f: 2D square grid of same size, represent the force
%   method: a numeric code which represent which smoother to use
%         see nma_V_cycle.m for allowed values
% OUTPUT:
%   u: updated u after one relaxation sweep over the whole grid
%
% EXAMPLE
% u=zeros(5); f=rand(5);
% u=nma_relax(u,f)
% u =
%          0         0         0         0         0
%          0   -0.0142   -0.0218   -0.0097         0
%          0   -0.0097   -0.0091   -0.0122         0
%          0   -0.0041   -0.0139   -0.0080         0
%          0         0         0         0         0
% Nasser M. Abbasi
% Math 228a, UC Davis, Fall 2010


USE_GSRB   = 1;
USE_JACOBI = 2;
USE_GSLEX  = 3;
USE_SOR    = 4;
USE_GSRB_PRE = 5;  % Gauss-Seidel red/black, but as preconditioner

if nargin < 3
   method = USE_GSRB;
else
   if method ~=1 && method ~=2 && method ~=3 && method ~=4 && method ~=5
      error('nma_relax:: invlaid method code');
   end
end

nma_validate_dimensions(u,f);
[n,~] = size(u);
h     = 1/(n-1);

switch method
   
   case USE_GSRB_PRE
      
      for i = 2:n-1
         for j = 2:n-1
            if bitget(i+j,1) == 0  % red squares
               u(i,j) = (1/4)*( u(i,j-1) + u(i,j+1) + ...
                  u(i-1,j) + u(i+1,j) - h^2 * f(i,j));
            end
         end
      end
      
      for i = 2:n-1
         for j = 2:n-1
            if bitget(i+j,1) ~=0  % black squares
               u(i,j) = (1/4)*( u(i,j-1) + u(i,j+1) + ...
                  u(i+1,j) + u(i-1,j) - h^2 * f(i,j));
            end
         end
      end
      
      for i = n-1:-1:2
         for j = n-1:-1:2
            if bitget(i+j,1) == 0  % red squares
               u(i,j) = (1/4)*( u(i,j-1) + u(i,j+1) + ...
                  u(i-1,j) + u(i+1,j) - h^2 * f(i,j));
            end
         end
      end
      
      for i = n-1:-1:2
         for j = n-1:-1:2
            if bitget(i+j,1) ~=0  % black squares
               u(i,j) = (1/4)*( u(i,j-1) + u(i,j+1) + ...
                  u(i+1,j) + u(i-1,j) - h^2 * f(i,j));
            end
         end
      end
      
   case USE_JACOBI
      i      = 2:n-1;
      j      = 2:n-1;
      u(i,j) = (1/4)*( u(i-1,j) + u(i+1,j) + u(i,j-1) + u(i,j+1) - h^2 * f(i,j) );
      
   case USE_GSRB
      for i = 2:n-1
         for j = 2:n-1
            if bitget(i+j,1) == 0  % red squares
               u(i,j) = (1/4)*( u(i,j-1) + u(i,j+1) + ...
                  u(i-1,j) + u(i+1,j) - h^2 * f(i,j));
            end
         end
      end
      
      for i = 2:n-1
         for j = 2:n-1
            if bitget(i+j,1) ~=0  % black squares
               u(i,j) = (1/4)*( u(i,j-1) + u(i,j+1) + ...
                  u(i+1,j) + u(i-1,j) - h^2 * f(i,j));
            end
         end
      end
      
   case USE_GSLEX
      
      unew = zeros(size(u));
      for i = 2:n-1
         for j = 2:n-1
            unew(i,j) = (1/4)*(unew(i-1,j)+u(i+1,j)...
               +unew(i,j-1)+u(i,j+1)-h^2*f(i,j));
         end
      end
      u = unew;
      
   case USE_SOR
      
      unew = zeros(size(u));
      w    = 2*(1-pi*h);  %optimal omega
      for i = 2 : n-1
         for j = 2 : n-1
            unew(i,j) = w/4* ( unew(i-1,j) + u(i+1,j) + unew(i,j-1)+...
               u(i,j+1) - h^2*f(i,j)) + (1-w)*u(i,j);
         end
      end
      u = unew;
end
end