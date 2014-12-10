function nn_params = initParams(nn_lsizes)
nn_params=[];
for i=1:size(nn_lsizes,2)-1
    param_count = nn_lsizes(i+1) * (nn_lsizes(i)+1);             % Parameter (i.e. weight) matrix size for layer i
    epsilon = 4*sqrt(6)/sqrt((nn_lsizes(i)+1) + nn_lsizes(i+1)); %

    nn_params = [nn_params; (rand(param_count,1)*2-1)*epsilon];
end
endfunction

