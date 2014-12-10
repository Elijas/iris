%TODO is initParams epsilon formula allright?
function main

% Get Data
load("irisData.txt");
[X, y, X_cv, y_cv, X_test, y_test, mu, s] = prepareData(irisData);
y=(y==3); %TODO predict other classes. (to nnCostFunction(if needed):  y = eye(nn_lsizes(end))(y,:);  )

% Set Learning Parameters
nn_lsizes = [4 1];
nn_lambda = 0.1;
nn_options = optimset('MaxIter', 5000);

% Learn
[nn_params, cost] = fmincg(@(p) nnCostFunction(p, nn_lsizes, X, y, nn_lambda), nnInitParams(nn_lsizes), nn_options);

% Measure performance
prediction = round(nnFeedForward(nn_params, nn_lsizes, X));
score = mean(double(  y == prediction ))*100;
fprintf('\nCV Set Accuracy: %f\n', score);

endfunction

