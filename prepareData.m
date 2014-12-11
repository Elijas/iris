%CV - Cross-validation data split is skipped (currently redundant)
function [X, y, \
          X_test, y_test, mu, s] = prepareData(data) %CV X_cv, y_cv,

% Scales features
features = data(:,1:end-1);
mu = mean(features, 1);
s = std(features, [], 1);
features -= (ones(size(data,1),1) * mu);
features ./= (ones(size(data,1),1) * s);
data(:,1:end-1) = features;

% Randomly shuffles dataset
data = data(randperm(size(data,1)), :);

% Splits dataset into training/(cross-validation)/test sets
size_train = round(size(data,1) * .7); %CV should be .6
data_train = data((1:size_train), :);
data = data(size_train+1:end, :);

%CV
%{
size_cv = round(size(data,1) * .5);
data_cv = data((1:size_cv), :);
data = data(size_cv+1:end, :);
%}

data_test = data;

X = data_train(:,1:end-1);
y = data_train(:,end);
%CV X_cv = data_cv(:,1:end-1);
%CV y_cv = data_cv(:,end);
X_test = data_test(:,1:end-1);
y_test = data_test(:,end);
endfunction

