clc,close all,clear all

lambda = [0,1,20];
a = [2,10];
h = [0.025,0.05,0.1];
timespan = [0 10];

fun = @(t, y, param) ((4.*t.*sqrt(y))) -(param.*(y-(t.^2+1).^2));
sol = zeros(18,floor((timespan(2) - timespan(1))/min(h)));
tsol = zeros(18,floor((timespan(2) - timespan(1))/min(h)));
dlen = zeros(1,18);

% legendStrings = string.empty;
colororder = [0.00  0.00  1.00;0.00  0.50  0.00;1.00  0.00  0.00;0.00  0.75  0.75;0.75  0.00  0.75;0.75  0.75  0.00;0.25  0.25  0.25;0.75  0.25  0.25;0.95  0.95  0.00;0.25  0.25  0.75;0.75  0.75  0.75;0.00  1.00  0.00;0.76  0.57  0.17;0.54  0.63  0.22;0.34  0.57  0.92;1.00  0.10  0.60;0.88  0.75  0.73;0.10  0.49  0.47;0.66  0.34  0.65;0.99  0.41  0.23];

figure(1)
set(gca, 'ColorOrder', colororder);
xlabel('time (t)')
ylabel('Value')
hold on

l = 1;
for i = 1:length(h)
    for k = 1:length(lambda)
        for j = 1:length(a)
            disp([h(i),lambda(k),a(j)])
%            legendStrings(end+1) ='h=' + num2str(h(i)) + ', l=' + num2str(lambda(k)) + ', a=' + num2str(a(j))
            nsteps = floor((timespan(2) - timespan(1))/h(i));
%            [tK4,yK4] = ERK4( fun, timespan, nsteps, a(j), lambda(k));
            [tLM,yLM] = LMsolver( fun, timespan, nsteps, a(j), lambda(k));
            dlen(l) = length(yLM);
            plot(tLM,yLM)
            sol(l,1:length(tLM)) = yLM;
            tsol(l,1:length(tLM)) = tLM;
            l = l + 1;
        end
    end
end


%%
figure(2)
plot(tsol(1,1:dlen(1)),sol(1,1:dlen(1)),...
     tsol(2,1:dlen(2)),sol(2,1:dlen(2)),...
     tsol(3,1:dlen(3)),sol(3,1:dlen(3)),...
     tsol(4,1:dlen(4)),sol(4,1:dlen(4)),...
     tsol(5,1:dlen(5)),sol(5,1:dlen(5)),...
     tsol(6,1:dlen(6)),sol(6,1:dlen(6)))
grid
xlim([timespan(1) timespan(2)])
%%
figure(3)
plot(tsol(7,1:dlen(7)),sol(7,1:dlen(7)),...
     tsol(8,1:dlen(8)),sol(8,1:dlen(8)),...
     tsol(9,1:dlen(9)),sol(9,1:dlen(9)),...
     tsol(10,1:dlen(10)),sol(10,1:dlen(10)),...
     tsol(11,1:dlen(11)),sol(11,1:dlen(11)),...
     tsol(12,1:dlen(12)),sol(12,1:dlen(12)))
grid
xlim([timespan(1) timespan(2)])

figure(4)
plot(tsol(13,1:dlen(13)),sol(13,1:dlen(13)),...
     tsol(14,1:dlen(14)),sol(14,1:dlen(14)),...
     tsol(15,1:dlen(15)),sol(15,1:dlen(15)),...
     tsol(16,1:dlen(16)),sol(16,1:dlen(16)),...
     tsol(17,1:dlen(17)),sol(17,1:dlen(17)),...
     tsol(18,1:dlen(18)),sol(18,1:dlen(18)))
grid
xlim([timespan(1) timespan(2)])















