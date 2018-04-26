function Unew = Vcycle(Unew,f,n_smooth)

n = size(Unew,1); 

if n == 3
  % if we are at the coarsest level
  h = 1/(n-1);
  Unew(2,2) = -0.25*h^2*f(2,2);
else
  % pre-smooth the error
  for i = 1:n_smooth
    Unew = smooth2d(Unew,f);
  end

  % calculate the residual
  i=2:n-1;
  j=2:n-1;

  R = zeros(n);
  R(i,j) = f(i,j) - (1/h^2)*( Unew(i,j-1) + Unew(i,j+1) + Unew(i+1,j) + Unew(i-1,j) - 4*Unew(i,j) ) ;
  % coarsen the residual
  Rc = coarsen(R);
  % recurse to Vcycle on a coarser grid
  Ecoarse = Vcycle(zeros(size(Rc)), Rc, n_smooth);
  % interpolate the error
  Ecoarse = interpolate(Ecoarse);
  % update the solution given the interpolated error
  Unew = Unew + Ecoarse;

  % post-smooth the error
  for i = 1:n_smooth
    Unew = smooth2d(Unew,f);
  end
end

end