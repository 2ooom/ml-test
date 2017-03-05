function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta);
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    h = X*theta;
    thetaTemp = zeros(n, 1);
    for i = 1:n
      thetaTemp(i) = theta(i) - alpha/m*sum((h - y).*X(:,i));
    end

    theta = thetaTemp;

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
