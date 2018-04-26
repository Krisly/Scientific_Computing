function u = nma_V_cycle(u,f,mu1,mu2,smoother)
%implement multigrid V Cycle

%---------------------------------------------------
% Implement multigrid V Cycle
%
% INPUT:  u: approximate solution
%         f: RHS
%         mu1: number of pre smooth
%         mu2: number of post smooth
%         method: a numeric code which represent which smoother to use
%         these are the allowed code:
%         [1] Gauss-Seidel red/black method, [2] means use Jacobi
%         [3] means use Gauss-Seidel LEX     [4] means use SOR
%         [5] means use GS red/black but as a preconditioner, which
%             means to do one forward sweep and one backward sweep.
%
% OUTPUT:
%         u: a more accurate solution than the input solution
%
% by Nasser M. Abbasi, Math 228A, UC Davis, Fall 2010


nma_validate_dimensions(u,f);  % asserts all dimensions makes sense
[n,~] = size(u);  % number of grid points on one dimension

if n == 3  % this is most coarse grid, 3 by 3, so solve directly
   h      = 1/(n-1);
   u(2,2) = -0.25*h^2*f(2,2); %since boundaries are all zero
else
   for i = 1:mu1   % PRE SMOOTH
      u = nma_relax(u,f,smoother);
   end
   
   %find residue, map to coarse
   the_residue = nma_f2c(nma_find_residue(u,f));
   
   % now, recusrively solve the error equation  A error = residue, use
   % zero for initial guess for the error, make residue the RHS
   
   error_correction = nma_V_cycle(zeros(size(the_residue)),...
      the_residue, mu1, mu2, smoother );
   
   error_correction = nma_c2f(error_correction); % coarse to fine mapping
   u                = u + error_correction;      % add correction
   
   for i = 1:mu2  % POST SMOOTH
      u = nma_relax(u,f,smoother);
   end
end

nma_check_all_zero_boundaries(u);  % asserts boundaries still zero
end