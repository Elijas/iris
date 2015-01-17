% Based on https://chrisjmccormick.wordpress.com/2013/08/15/radial-basis-function-network-rbfn-tutorial/
function nn
addpath('dataset')
addpath('kMeans')
addpath('RBFN')

disp('# Loading dataset...')
load('irisDataShuffled')
X = irisDataShuffled(:, 1:4);
y = irisDataShuffled(:, 5);

m = size(X, 1);

% ==========================
% Train RBF Network
% ==========================
disp('# Training the RBFN...')
[Centers, betas, Theta] = trainRBFN(X, y, 15, false);

% ==========================
% Evaluate performance
% ==========================
disp("# Measuring training accuracy...")

numRight = 0;

% For each training sample...
for (i = 1 : m)
    % Compute the scores for both categories.
    scores = evaluateRBFN(Centers, betas, Theta, X(i, :));
    
    [maxScore, category] = max(scores);

    % Validate the result.
    if (category == y(i))
        numRight++;
    end
    
end

accuracy = numRight / m * 100;
fprintf("Training accuracy: %d / %d, %.1f%%\n", numRight, m, accuracy);
if exist('OCTAVE_VERSION') fflush(stdout); end;

endfunction

