function error = expected_error(theta,boundary)
 
a = theta(1);
b = theta(2);

%error = (2/a)*log(1+exp(-a*boundary - b))+ boundary + b/a;
error = (2/a)*log(1+exp(a*boundary+b)) - boundary - b/a;