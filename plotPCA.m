function plotPCA
% Load Data
load('irisDataRaw.txt');
[X, y, X_test, y_test, data_mu, data_s] = prepareData(irisDataRaw);
load('irisDataPrepared');
X_all = [X; X_test];
y_all = [y; y_test];



% Run PCA (i.e. Principal Component Analysis) (reduce to 2 dimensions)
printf("Reducing to 2 dimensions\n");
Z_all = X_all*diag(svd(1/size(X_all,1)*(X_all'*X_all)))(:,1:    2    );

close all;hold on;
plot(Z_all(find(y==1),1), Z_all(find(y==1),2), 'ko', 'MarkerFaceColor', 'r', 'MarkerSize', 7);
plot(Z_all(find(y==2),1), Z_all(find(y==2),2), 'ko', 'MarkerFaceColor', 'b', 'MarkerSize', 7);
plot(Z_all(find(y==3),1), Z_all(find(y==3),2), 'ko', 'MarkerFaceColor', 'g', 'MarkerSize', 7);
printf("[Press Enter]"); pause; printf("\r             \n");
% NOT REALLY USEFUL SINCE DATA IS OBVIOUSLY NOT SEPARABLE WITH 2 DIMENSIONS (except class 1): 
% Train a NN for one of the reduced class and plot NN output in 3D
%{
prediction_class = 1;

sFirst = size(Z_all,2);     % Units in the first layer
sLast = max(y);             % Units in the last layer
yExp = eye(sLast)(y,:);     % Expand yExp to binaryExp vectors (to prepare for NN learning)
yExp_test = eye(sLast)(y_test,:);
nn_lambda = .1;
nn_options = optimset('MaxIter', 10000);
printf("Training...  (lambda = %g, iteration limit = %d)\n", nn_lambda, nn_options.MaxIter);
nn_lsizes = [sFirst 1];
nn_params = fmincg(@(p) nnCostFunction(p, nn_lsizes, Z_all, y_all==prediction_class, nn_lambda), nnInitParams(nn_lsizes), nn_options);
accuracyExp = mean(  (  (y_all==prediction_class)==round(nnFeedForward(nn_params, nn_lsizes, Z_all))  )(:)  );
%accuracyExp_test = mean(  (  yExp_test==round(nnFeedForward(nn_params, nn_lsizes, X_test))  )(:)  );
printf(" %d", nn_lsizes), printf("\t| %10.2f%%\n", accuracyExp*100);
dim1_linspace = linspace(min(Z_all(:,1)), max(Z_all(:,1)), 50);
dim2_linspace = linspace(min(Z_all(:,2)), max(Z_all(:,2)), 50);
[dimdim1, dimdim2] = meshgrid(dim1_linspace, dim2_linspace);
predictions = zeros(length(dim1_linspace), length(dim2_linspace));
for i = 1:length(dim1_linspace)
    for j = 1:length(dim2_linspace)
        predictions(i,j) = nnFeedForward(nn_params, nn_lsizes, [dimdim1(i,j), dimdim2(i,j)]) - .5;
    end
end
%3D plot
mesh(dim1_linspace, dim2_linspace, predictions);
hold on;
mesh(linspace(min(Z_all(:,1)),max(Z_all(:,1)), 2), linspace(min(Z_all(:,1)),max(Z_all(:,1)), 2), zeros(2, 2));
%Contour plot
contour(dim1_linspace, dim2_linspace, predictions, [0 0], 'LineWidth', 2);
%}



% Run PCA (i.e. Principal Component Analysis) (reduce to 3 dimensions)
printf("Reducing to 3 dimensions\n");
Z_all = X_all * diag(svd(1/size(X_all,1)*(X_all'*X_all)))(:,1:    3    );

close all;hold on;
scatter3(Z_all(find(y==1),1), Z_all(find(y==1),2), Z_all(find(y==1),3), 'r');
scatter3(Z_all(find(y==2),1), Z_all(find(y==2),2), Z_all(find(y==2),3), 'b');
scatter3(Z_all(find(y==3),1), Z_all(find(y==3),2), Z_all(find(y==3),3), 'g');
% Train a NN for one of the reduced class and plot NN output in 3D
prediction_class = 2;

sFirst = size(Z_all,2);         % Units in the first layer
sLast = max(y_all);             % Units in the last layer
y_allExp = eye(sLast)(y_all,:); % Expand yExp to binaryExp vectors (to prepare for NN learning)
nn_lambda = .5;
nn_options = optimset('MaxIter', 10000);
printf("Training...  (lambda = %g, iteration limit = %d)\n", nn_lambda, nn_options.MaxIter);
nn_lsizes = [sFirst 4 sLast];
nn_params = fmincg(@(p) nnCostFunction(p, nn_lsizes, Z_all, y_allExp, nn_lambda), nnInitParams(nn_lsizes), nn_options);
accuracyExp = mean(  (  y_allExp==round(nnFeedForward(nn_params, nn_lsizes, Z_all))  )(:)  );
printf(" %d", nn_lsizes), printf("\t| %10.2f%%\n", accuracyExp*100);
dim1_linspace = linspace(min(Z_all(:,1)), max(Z_all(:,1)), 10);
dim2_linspace = linspace(min(Z_all(:,2)), max(Z_all(:,2)), 10);
dim3_linspace = linspace(min(Z_all(:,3)), max(Z_all(:,3)), 10);
[dimdim1, dimdim2, dimdim3] = meshgrid(dim1_linspace, dim2_linspace, dim3_linspace);
predictions = zeros(length(dim1_linspace), length(dim2_linspace), length(dim3_linspace));
for i = 1:length(dim1_linspace)
    for j = 1:length(dim2_linspace)
        for u = 1:length(dim3_linspace)
            predictions(i,j,u) = -0.5 + nnFeedForward(nn_params, nn_lsizes, [dimdim1(i,j,u) dimdim2(i,j,u) dimdim3(i,j,u)]) (prediction_class);
        end
    end
end

cont=predictions>0;
scatter3(dimdim1(:)(find(cont(:))), dimdim2(:)(find(cont(:))), dimdim3(:)(find(cont(:))), [], 'y');
%scatter3(dimdim1(:)(find(cont(:))), dimdim2(:)(find(cont(:))), dimdim3(:)(find(cont(:))), [], predictions(:)(find(cont(:))));

endfunction

