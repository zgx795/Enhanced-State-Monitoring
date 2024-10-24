function finetune_model = finetunemd(md, datastr, newdatastr, selectfeature, x_mean, x_sig, y_mean, y_sig)
% Fine-tunes a pre-trained neural network using new data

% Load and standardize new data
opts = detectImportOptions(newdatastr, 'VariableNamingRule', 'preserve');
opts.SelectedVariableNames = opts.SelectedVariableNames(1:14);
df = readtable(newdatastr, opts);
df = df(:, 2:end);
newData = df(1:end, :);
newRows = height(df);
X_new = newData(:, selectfeature);
y_new = newData(:, end-3:end);

% Standardization
X_new = (X_new - x_mean) ./ x_sig;
y_new = (y_new - y_mean) ./ y_sig;

% Load pre-trained model
net = md.model;

% Load and standardize old data
opts = detectImportOptions(datastr, 'VariableNamingRule', 'preserve');
opts.SelectedVariableNames = opts.SelectedVariableNames(1:14);
df = readtable(datastr, opts);
df = df(:, 2:end);
allData = df(1:end, :);
X = allData(:, selectfeature);
y = allData(:, end-3:end);
X = (X - x_mean) ./ x_sig;
y = (y - y_mean) ./ y_sig;

% Data conversion
X = table2array(X);
X_new = table2array(X_new);
y = table2array(y);

% Select old data samples
[~, X_old, ~, y_old] = split_dataset(X, y, newRows);
y_old = array2table(y_old);
y_old.Properties.VariableNames = y_new.Properties.VariableNames;

% Reshape input data to 4-D format
X = reshape(X', [size(X, 2), 1, 1, size(X, 1)]);
X_new = reshape(X_new', [size(X_new, 2), 1, 1, size(X_new, 1)]);
X_old = reshape(X_old', [size(X_old, 2), 1, 1, size(X_old, 1)]);

% Concatenate data
X_mix = cat(4, X_old, X_new);
y_mix = cat(1, y_old, y_new);
y_mix = table2array(y_mix);

% Split into training and validation sets
numElements = size(X_mix, 4);
numLastElements = round(0.2 * numElements);
X_val = X_mix(:, :, :, end - numLastElements + 1:end);
X_train = X_mix(:, :, :, 1:numElements);
y_train = y_mix(1:numElements, :);
y_val = y_mix(end - numLastElements + 1:end, :);

% Define training options
numEpochs = 5;
miniBatchSize = 128;
validationFrequency = floor((1 / 0.1) * numEpochs);
performance = ModelPerformance();
outputFcn = @(info) performance.update(info);
options = trainingOptions('adam', ...
    'MaxEpochs', numEpochs, ...
    'InitialLearnRate', 0.001, ...
    'MiniBatchSize', miniBatchSize, ...
    'ValidationData', {X_val, y_val}, ...
    'ValidationFrequency', validationFrequency, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'OutputFcn', outputFcn);

% Train the model
[model, traininfo] = trainNetwork(X_train, y_train, net.Layers, options);

% Model prediction and evaluation
y_pred1 = predict(model, X_old);
y_pred2 = predict(model, X_new);
y_pred3 = predict(model, X);
y_p1 = predict(net, X_old);
y_p2 = predict(net, X_new);
y_p3 = predict(net, X);

% Reverse standardization
y_pred1_orig = y_pred1 .* y_sig + y_mean;
y_pred2_orig = y_pred2 .* y_sig + y_mean;
y_pred3_orig = y_pred3 .* y_sig + y_mean;
y_p1_orig = y_p1 .* y_sig + y_mean;
y_p2_orig = y_p2 .* y_sig + y_mean;
y_p3_orig = y_p3 .* y_sig + y_mean;

% Calculate MSE
mse1 = immse(y_old, double(y_pred1));
mse2 = immse(y_new, double(y_pred2));
mse3 = immse(y, double(y_pred3));
mse11 = immse(y_old, double(y_p1));
mse22 = immse(y_new, double(y_p2));
mse33 = immse(y, double(y_p3));

% Print MSE results
fprintf('MSE for y_old: %.4f\n', mse1);
fprintf('MSE for y_new: %.4f\n', mse2);
fprintf('MSE for y: %.4f\n', mse3);
fprintf('Old Model MSE for y_old: %.4f\n', mse11);
fprintf('Old Model MSE for y_new: %.4f\n', mse22);
fprintf('Old Model MSE for y: %.4f\n', mse33);

% Split dataset function
function [X_train, X_test, y_train, y_test] = split_dataset(X, y, test_size)
    N = size(X, 1);
    idx = randperm(N);
    test_idx = idx(1:test_size);
    train_idx = idx(test_size+1:end);
    X_test = X(test_idx, :);
    y_test = y(test_idx, :);
    X_train = X(train_idx, :);
    y_train = y(train_idx, :);
end

% Return fine-tuned model and results
finetune_model.model = model;
finetune_model.traininfo = traininfo;
finetune_model.y_p1 = y_p1;
finetune_model.y_p2 = y_p2;
finetune_model.y_p3 = y_p3;
finetune_model.y_pred1 = y_pred1;
finetune_model.y_pred2 = y_pred2;
finetune_model.y_pred3 = y_pred3;
finetune_model.y_p1_orig = y_p1_orig;
finetune_model.y_p2_orig = y_p2_orig;
finetune_model.y_p3_orig = y_p3_orig;
finetune_model.y_pred1_orig = y_pred1_orig;
finetune_model.y_pred2_orig = y_pred2_orig;
finetune_model.y_pred3_orig = y_pred3_orig;

end
