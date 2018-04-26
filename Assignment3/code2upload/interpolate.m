function [ R ] = interpolate( Rc )

n = size(Rc,1);
n = 2*(n-1)+1;
R = zeros(n);
i = 3:2:n-2;
j = 3:2:n-2;

R(i+1,j-1) = R(i+1,j-1) + (1/4)* Rc(ceil(i/2),ceil(i/2));
R(i+1,j) = R(i+1,j) + (1/2)* Rc(ceil(i/2),ceil(i/2));
R(i+1,j+1) = R(i+1,j+1) + (1/4)* Rc(ceil(i/2),ceil(i/2));

R(i-1,j-1) = R(i-1,j-1) + (1/4)*Rc(ceil(i/2),ceil(i/2));
R(i-1,j) = R(i-1,j) + (1/2)*Rc(ceil(i/2),ceil(i/2));
R(i-1,j+1) = R(i-1,j+1) + (1/4)*Rc(ceil(i/2),ceil(i/2));

R(i,j-1) = R(i,j-1) + (1/2)*Rc(ceil(i/2),ceil(i/2));
R(i,j) = R(i,j) + Rc(ceil(i/2),ceil(i/2));
R(i,j+1) = R(i,j+1) + (1/2)* Rc(ceil(i/2),ceil(i/2));

end
