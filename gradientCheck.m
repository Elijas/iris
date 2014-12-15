function [grad_est] = gradientCheck( \
%%%% Configuration
nn_params, nn_lsizes, X, y, lambda)

par = nn_params;
costFunction = @(p) nnCostFunction(p, nn_lsizes, X, y, lambda);
epsilon = 0.001;
%%%%

% Grad Estimation
grad_est = zeros(length(par), 1);
for i=1:length(nn_params)
    par_PLUS = par_MINUS = par;

    par_PLUS(i) += epsilon;
    par_MINUS(i) -= epsilon;

    cost_PLUS  = costFunction(par_PLUS);
    cost_MINUS = costFunction(par_MINUS);
    
    grad_est(i) = ( cost_PLUS - cost_MINUS ) / (2*epsilon);
end

endfunction

