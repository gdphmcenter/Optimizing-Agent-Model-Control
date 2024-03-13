%% Modeling due to data confidentiality, only part of the code is provided, using the water heater as an example, %%
%% collecting the water heater openness data as input and the water discharge temperature data as output%%
data = xlsread('C:surveying.xlsx');

t = data(:, 1);
u = data(:, 2);
y = data(:, 3);

sys_func = @(x, t) (x(1) * (1 - exp(-t/x(2)))) .* u;


x0 = [1; 1]; 


options = optimoptions('lsqcurvefit','Display','iter');
[x,resnorm] = lsqcurvefit(sys_func, x0, t, y, [], [], options);


K = x(1);
tau = x(2);

y_est = sys_func(x, t);


plot(t, y, 'bo', t, y_est, 'r-');
legend('acture', 'pre');
xlabel('time (s)');
ylabel('temp (°C)');
title('result');