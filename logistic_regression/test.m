%% Machine Learning Online Class - Exercise 2: Logistic Regression
%
%  Instructions
%  ------------
%
%  This file contains code that helps you get started on the logistic
%  regression exercise. You will need to complete the following functions
%  in this exericse:
%
%     sigmoid.m
%     costFunction.m
%     predict.m
%     costFunctionReg.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%

%% Initialization
clear ; close all; clc
rng('shuffle');
true_theta = [2;3];
%{
test_X = [-10:0.01:10];
for i = 1:2001
    prob(i) = 1/(1+exp(1)^(-[test_X(i) 1]*true_theta));
end
plot(test_X,prob,'b');
%}
NUM_LOOP = 100;
unpurified_error = zeros(1,NUM_LOOP);
purified_error = zeros(1,NUM_LOOP);
svm_error = zeros(1,NUM_LOOP);

%% Begin Loop
for kk = 1: 1: NUM_LOOP

if mod(kk,10) == 0
    kk
end

n = 1000;

%% Generate Data and label
xx = -10 + 20*rand(n,1);
x = [xx ones(n,1)];
y = zeros(n,1);
y_filter = zeros(n,1);


for i = 1 : n
    threshold = 1/(1+exp(1)^(-x(i,:)*true_theta));
    coin = rand(1,1);
    if threshold > coin
        y(i) = 1;
    else
        y(i) = 0;
    end

    if threshold > 0.5
        y_filter(i) = 1;
    else
        y_filter(i) = 0;
    end
end

if isempty(xx(y_filter == 0)) || isempty(xx(y_filter == 1))
    kk = kk -1;
    continue;
end

initial_theta = zeros(2,1);

%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost
options = optimset('GradObj', 'on', 'MaxIter', 1000, 'Display', 'off','TolFun',0,'TolX',0,'MaxFunEvals',inf);

[theta_filter, cost_filter] = ...
	fminunc(@(t)(costFunction(t, x, y_filter)), initial_theta, options);

[theta, cost] = ...
	fminunc(@(t)(costFunction(t, x, y)), initial_theta, options);

%% Plot the logistic function
%{
test_X = [-10:0.01:10];
for i = 1:2001
    prob(i) = 1/(1+exp(1)^(-[1 test_X(i)]*true_theta));
end
plot(test_X,prob,'b');

hold on;
for i = 1:2001
    prob(i) = 1/(1+exp(1)^(-[1 test_X(i)]*theta_filter));
end
plot(test_X,prob,'m');

hold on;
for i = 1:2001
    prob(i) = 1/(1+exp(1)^(-[1 test_X(i)]*theta));
end
plot(test_X,prob,'r');

legend('True Model','Filter Model','Learned Model');
fprintf('Expected Error\n');
fprintf('Unpurified: %f\n',expected_error(true_theta,boundary_unpurified));
fprintf('Purified: %f\n',expected_error(true_theta,boundary_purified));
fprintf('SVM: %f\n',expected_error(true_theta,boundary_svm));
%}
%% Calculate expected error
[xx(:,1) y];
boundary_unpurified = -theta(2)/theta(1);
boundary_purified = -theta_filter(2)/theta_filter(1);
boundary_svm = (max(xx(y_filter == 0)) + min(xx(y_filter == 1)))/2;

if isempty(xx(y_filter == 0))
    boundary_svm = min(xx(y_filter == 1));
else if isempty(xx(y_filter == 1))
    boundary_svm = max(xx(y_filter == 0));
    end
end



unpurified_error(kk) = expected_error(true_theta,boundary_unpurified);
purified_error(kk) = expected_error(true_theta,boundary_purified);
svm_error(kk) = expected_error(true_theta,boundary_svm);
end


file = fopen('result.txt','a');
fprintf(file,'\n--------Theta:[%f.%f], N = %d\n',true_theta(1),true_theta(2),n);
fprintf(file,'True error:%f\n',expected_error(true_theta,-true_theta(2)/true_theta(1)));
fprintf(file,'unpurified error %f\n',mean(unpurified_error));
fprintf(file,'purified error %f\n',mean(purified_error));
fprintf(file,'svm error %f\n',mean(svm_error));
fprintf(file,'unpurified VS purified & svm wins %d\n',sum(unpurified_error < purified_error & unpurified_error < svm_error));
fprintf(file,'purified VS unpurified wins %d\n',sum(purified_error < unpurified_error));
fprintf(file,'svm VS unpurified wins %d\n',sum(svm_error < unpurified_error));
fprintf(file,'svm VS purified wins %d\n',sum(svm_error < purified_error));

fprintf(file,'purified VS unpurified sum error %f\n', sum(-purified_error+unpurified_error)/length(purified_error));
fprintf(file,'svm VS unpurified sum error %f\n', sum(-svm_error+unpurified_error)/length(purified_error));
fprintf(file,'SVM VS purified sum error %f\n', sum(-svm_error+purified_error)/length(purified_error));
fclose(file);


