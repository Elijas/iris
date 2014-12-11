function main

% Acquire Data
load('irisData.txt');
[X, y, X_test, y_test, mu, s] = prepareData(irisData);


% Experiment 1: Single Layer Perceptron
disp(" ### Experiment No.1: Using a Single Layer Perceptrons to classify the class 2")
disp(" ## Learning..");
nn_lsizes = [4 1];
nn_lambda = 0;
nn_options = optimset('MaxIter', 200);
[nn_params, cost] = fmincg(@(p) nnCostFunction(p, nn_lsizes, X, y==2, nn_lambda), nnInitParams(nn_lsizes), nn_options);
predictions = round(nnFeedForward(nn_params, nn_lsizes, X_test));
fprintf(" ## Done. Accuracy of predicting class 2: %f\n", mean((y_test==2) == predictions)*100);
disp(" ## Press Enter to continue ..."); pause;


% Experiment 2: Multi-layered Neural Networks
disp("\n ### Experiment No.2: Using a Multi-layered Neural Networks")
sFirst = size(X,2);     % Units in the first layer
sLast = max(y);         % Units in the last layer
yExp = eye(sLast)(y,:);    % Expand yExp to binaryExp vectors (to prepare for NN learning)
yExp_test = eye(sLast)(y_test,:);
nn_lambda = 1;
nn_options = optimset('MaxIter', 400);

fprintf(" ## Learning.. (lambda = %f)\n", nn_lambda);
for s2=0:7
    nn_lsizes = [sFirst s2 sLast];
    nn_lsizes = nn_lsizes(find(nn_lsizes)); %eliminates zeros in the vector
    
    nn_params = fmincg(@(p) nnCostFunction(p, nn_lsizes, X, yExp, nn_lambda), nnInitParams(nn_lsizes), nn_options);
    
    accuracyExp = mean(((yExp) ==  (round(nnFeedForward(nn_params, nn_lsizes, X)))  )(:));
    accuracyExp_test = mean(((yExp_test) ==  (round(nnFeedForward(nn_params, nn_lsizes, X_test)))  )(:));
    fprintf(" ## Accuracies: %f (tr.set) %f (test set) | Layers:", accuracyExp*100, accuracyExp_test*100);
    disp(nn_lsizes);
end
disp(" ## Press Enter to continue ..."); pause;


% Experiment 3: Separate Multi-layered Neural Networks for each class
disp("\n ### Experiment No.3: Using separate Multi-layered Neural Networks for separate classes")
sFirst = size(X,2);     % Units in the first layer
sLast = max(y);         % Units in the last layer (class count)
yExp = eye(sLast)(y,:);    % Expand yExp to binaryExp vectors (to prepare for NN learning)
yExp_test = eye(sLast)(y_test,:);
nn_lambda = 1;
nn_options = optimset('MaxIter', 400);

fprintf(" ## Learning.. (lambda = %f)\n", nn_lambda);
for s2=0:7
    nn_lsizes = [sFirst s2 1];
    nn_lsizes = nn_lsizes(find(nn_lsizes)); %eliminates zeros in the vector
    
    predictions = [];
    predictions_test = [];
    for n=1:sLast
        nn_params = fmincg(@(p) nnCostFunction(p, nn_lsizes, X, yExp(:,n), nn_lambda), nnInitParams(nn_lsizes), nn_options);
        predictions = [predictions nnFeedForward(nn_params, nn_lsizes, X)];
        predictions_test = [predictions_test nnFeedForward(nn_params, nn_lsizes, X_test)];
    end
    
    [predictions, predictions] = max(predictions,[],2);
    [predictions_test, predictions_test] = max(predictions_test,[],2);
    fprintf(" ## Accuracies: %f (tr.set) %f (test set) | Layers:", mean(y == predictions)*100, mean(y_test == predictions_test)*100);
    disp(nn_lsizes);
end

endfunction

