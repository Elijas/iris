function nnTraining
LAMBDA = 2.2;
% Acquire Data
load('irisDataRaw.txt');
[X, y, X_test, y_test, data_mu, data_s] = prepareData(irisDataRaw);
load('irisDataPrepared');

X = polyFeatures(X);
X_test = polyFeatures(X_test);

% Experiment 1: Single Layer Perceptron
printf("# Experiment No.1: Training a Single Layer Perceptron\n")
nn_lsizes = [size(X,2) 1];
nn_lambda = 3;
nn_options = optimset('MaxIter', 50);
predict_class = 2;

printf("Training...  (lambda = %g, iteration limit = %d)\n", nn_lambda, nn_options.MaxIter);
warning('off', 'Octave:possible-matlab-short-circuit-operator');
[nn_params, cost] = fmincg(@(p) nnCostFunction(p, nn_lsizes, X, y==predict_class, nn_lambda), nnInitParams(nn_lsizes), nn_options);
predictions = round(nnFeedForward(nn_params, nn_lsizes, X_test));
printf("Accuracy: %g%% (predicting class %d, test set)\n", mean((y_test==predict_class) == predictions)*100, predict_class);
printf("[Press Enter]"); pause; printf("\r             \n");

% Experiment 2: Multi-layered Neural Networks
printf("# Experiment No.2: Training Multi-layered Neural Networks\n")
sFirst = size(X,2);     % Units in the first layer
sLast = max(y);         % Units in the last layer
yExp = eye(sLast)(y,:); % Expand yExp to binaryExp vectors (to prepare for NN learning)
yExp_test = eye(sLast)(y_test,:);

nn_lambda = LAMBDA;
nn_options = optimset('MaxIter', 600);

printf("Training...  (lambda = %g, iteration limit = %d)\n", nn_lambda, nn_options.MaxIter);
printf("-----------------------------------\n");
printf("Layers  | Acc. (trn.) | Acc. (test)\n");
for s2=50 %0:7
    nn_lsizes = [sFirst s2 sLast];
    nn_lsizes = nn_lsizes(find(nn_lsizes)); % Eliminates zeros in the vector
    
    nn_params = fmincg(@(p) nnCostFunction(p, nn_lsizes, X, yExp, nn_lambda), nnInitParams(nn_lsizes), nn_options);
    
    accuracyExp = mean(  (  yExp==round(nnFeedForward(nn_params, nn_lsizes, X))  )(:)  );
    accuracyExp_test = mean(  (  yExp_test==round(nnFeedForward(nn_params, nn_lsizes, X_test))  )(:)  );
    printf(" %d", nn_lsizes), printf("\t| %10.2f%% | %10.2f%%\n", accuracyExp*100, accuracyExp_test*100);
end
printf("-----------------------------------\n");
printf("[Press Enter]"); pause; printf("\r             \n");


% Experiment 3: Separate Multi-layered Neural Networks for each class
printf("# Experiment No.3: Training Multi-layered NNs for separate classes\n")
sFirst = size(X,2);     % Units in the first layer
sLast = max(y);         % Units in the last layer (class count)
yExp = eye(sLast)(y,:); % Expand yExp to binaryExp vectors (to prepare for NN learning)
yExp_test = eye(sLast)(y_test,:);

nn_lambda = LAMBDA;
nn_options = optimset('MaxIter', 200);

printf("Training...  (lambda = %g, iteration limit = %d)\n", nn_lambda, nn_options.MaxIter);
printf("-----------------------------------\n");
printf("Layers  | Acc. (trn.) | Acc. (test)\n");
for s2=50 %0:7
    nn_lsizes = [sFirst s2 1];
    nn_lsizes = nn_lsizes(find(nn_lsizes)); % Eliminates zeros in the vector
    
    predictions = []; 
    predictions_test = []; % These are used to contatenate outputs, to make them effectively the same as with a Neural Network with multiple units in output layer
    for n=1:sLast
        nn_params = fmincg(@(p) nnCostFunction(p, nn_lsizes, X, yExp(:,n), nn_lambda), nnInitParams(nn_lsizes), nn_options);
        predictions = [predictions nnFeedForward(nn_params, nn_lsizes, X)];
        predictions_test = [predictions_test nnFeedForward(nn_params, nn_lsizes, X_test)];
    end
    
    [predictions, predictions] = max(predictions,[],2);
    [predictions_test, predictions_test] = max(predictions_test,[],2);
    printf(" %d", nn_lsizes), printf("\t| %10.2f%% | %10.2f%%\n", mean(y==predictions)*100, mean(y_test==predictions_test)*100);
end
printf("-----------------------------------\n");
printf("[Press Enter]"); pause; printf("\r             \n");
%{
% Experiment 4: Large Neural Network
sFirst = size(X,2);     % Units in the first layer
sLast = max(y);         % Units in the last layer
yExp = eye(sLast)(y,:); % Expand yExp to binaryExp vectors (to prepare for NN learning)
yExp_test = eye(sLast)(y_test,:);

nn_lambda = .5;
nn_options = optimset('MaxIter', 1000);
printf("# Experiment No.4: Training a Large NN\n")
printf("Training...  (lambda = %g, iteration limit = %d)\n", nn_lambda, nn_options.MaxIter);
nn_lsizes = [sFirst 150 sLast];
nn_lsizes = nn_lsizes(find(nn_lsizes)); % Eliminates zeros in the vector
    
nn_params = fmincg(@(p) nnCostFunction(p, nn_lsizes, X, yExp, nn_lambda), nnInitParams(nn_lsizes), nn_options);
    
accuracyExp = mean(  (  yExp==round(nnFeedForward(nn_params, nn_lsizes, X))  )(:)  );
accuracyExp_test = mean(  (  yExp_test==round(nnFeedForward(nn_params, nn_lsizes, X_test))  )(:)  );
printf(" %d", nn_lsizes), printf("\t| %10.2f%% | %10.2f%%\n", accuracyExp*100, accuracyExp_test*100);
%}
endfunction
