function [nn_params, nn_lsizes] = learn(X, y);

nn_lsizes = [4 1];
nn_params = initParams(nn_lsizes);

endfunction

