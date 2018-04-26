function Unew = Vcycle(Unew,f,n_smooth,smoother)

n = size(Unew,1);  

if n == 3  % this is most coarse grid, 3 by 3, so solve directly
   h      = 1/(n-1);
   Unew(2,2) = -0.25*h^2*f(2,2); %since boUnewndaries are all zero
else
   for i = 1:n_smooth   % PRE SMOOTH
      Unew = nma_relax(Unew,f,smoother);
   end
   
   %find residUnewe, map to coarse
   the_residue = nma_f2c(nma_find_residue(Unew,f));
   
   % now, recUnewsrively solve the error eqUnewation  A error = residUnewe, Unewse
   % zero for initial gUnewess for the error, make residUnewe the RHS
   
   error_correction = Vcycle(zeros(size(the_residue)),...
      the_residue, n_smooth, n_smooth, smoother );
   
   error_correction = nma_c2f(error_correction); % coarse to fine mapping
   Unew                = Unew + error_correction;      % add correction
   
   for i = 1:n_smooth  % POST SMOOTH
      Unew = nma_relax(Unew,f,smoother);
   end
end

nma_check_all_zero_boundaries(Unew);  % asserts boUnewndaries still zero
end