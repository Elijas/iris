function plotFeaturePairs
% Load Data
load('irisDataRaw.txt');
[X, y, X_test, y_test, data_mu, data_s] = prepareData(irisDataRaw);
load('irisDataPrepared');


% Train a NN
sFirst = size(X,2);     % Units in the first layer
sLast = max(y);         % Units in the last layer
yExp = eye(sLast)(y,:); % Expand yExp to binaryExp vectors (to prepare for NN learning)
yExp_test = eye(sLast)(y_test,:);

nn_lambda = .1;
nn_options = optimset('MaxIter', 10000);
printf("Training...  (lambda = %g, iteration limit = %d)\n", nn_lambda, nn_options.MaxIter);
nn_lsizes = [sFirst 2 sLast];
nn_lsizes = nn_lsizes(find(nn_lsizes)); % Eliminates zeros in the vector
    
nn_params = fmincg(@(p) nnCostFunction(p, nn_lsizes, X, yExp, nn_lambda), nnInitParams(nn_lsizes), nn_options);
    
accuracyExp = mean(  (  yExp==round(nnFeedForward(nn_params, nn_lsizes, X))  )(:)  );
accuracyExp_test = mean(  (  yExp_test==round(nnFeedForward(nn_params, nn_lsizes, X_test))  )(:)  );
printf(" %d", nn_lsizes), printf("\t| %10.2f%% | %10.2f%%\n", accuracyExp*100, accuracyExp_test*100);


% plot 4D by plotting each pair of dimentions in 2D
X_all = [X; X_test];
y_all = [y; y_test];
close all;
fig = 0;
for dim1=1:4
    for dim2=2:4
        if dim1>=dim2, continue, end
        figure(++fig);
        hold on;

        plot(X(find(y==1),dim1), X(find(y==1),dim2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);
        plot(X(find(y==2),dim1), X(find(y==2),dim2), 'ko', 'MarkerFaceColor', 'b', 'MarkerSize', 7);
        plot(X(find(y==3),dim1), X(find(y==3),dim2), 'ko', 'MarkerFaceColor', 'g', 'MarkerSize', 7);
        plot(X_test(find(y_test==1),dim1), X_test(find(y_test==1),dim2), 'ro', 'MarkerFaceColor', 'y', 'MarkerSize', 7);
        plot(X_test(find(y_test==2),dim1), X_test(find(y_test==2),dim2), 'ro', 'MarkerFaceColor', 'b', 'MarkerSize', 7);
        plot(X_test(find(y_test==3),dim1), X_test(find(y_test==3),dim2), 'ro', 'MarkerFaceColor', 'g', 'MarkerSize', 7);

        prediction_class = 2;
        dim1_linspace = linspace(min(X_all(:,dim1)), max(X_all(:,dim1)), 50);
        dim2_linspace = linspace(min(X_all(:,dim2)), max(X_all(:,dim2)), 50);
        [dimdim1, dimdim2] = meshgrid(dim1_linspace, dim2_linspace);
        predictions = zeros(length(dim1_linspace), length(dim2_linspace));
        for i = 1:length(dim1_linspace)
            for j = 1:length(dim2_linspace)
                input = mean(X_all(find(y==prediction_class),:),1); input(dim1) = dimdim1(i,j); input(dim2) = dimdim2(i,j);
                predictions(i,j) = nnFeedForward(nn_params, nn_lsizes, input) (prediction_class) -.5;
            end
        end

        %3D plot
        mesh(dim1_linspace, dim2_linspace, predictions);
        hold on;
        mesh(linspace(min(X_all(:,dim1)),max(X_all(:,dim1)), 2), linspace(min(X_all(:,dim1)),max(X_all(:,dim1)), 2), zeros(2, 2));

        %Contour plot
        contour(dim1_linspace, dim2_linspace, predictions, [0 0], 'LineWidth', 2);
    end
end
endfunction

