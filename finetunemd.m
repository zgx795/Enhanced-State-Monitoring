function finetune_model = finetunemd(md, datastr, newdatastr, selectfeature, x_mean, x_sig, y_mean, y_sig)
% FINETUNEMD Fine-tunes a pre-trained neural network model using new data.
%
% This function fine-tunes an existing neural network model by using both old 
% and new datasets. The function standardizes the data based on the original 
% model's training data, retrains the model, and provides performance 
% evaluation metrics.
%
% Parameters:
%   md: Struct containing the pre-trained model.
%   datastr: File path to the old dataset.
%   newdatastr: File path to the new dataset.
%   selectfeature: Indices of the features used for training.
%   x_mean, x_sig: Mean and standard deviation of the input data.
%   y_mean, y_sig: Mean and standard deviation of the output data.
%
% Returns:
%   finetune_model: A structure containing the fine-tuned model, predictions, 
%   and training information.

%% Step 1: Load and standardize new data
opts = detectImportOptions(newdatastr, 'VariableNamingRule', 'preserve');
opts.SelectedVariableNames = opts.SelectedVariableNames(1:14); % Select the first 14 columns
df = readtable(newdatastr, opts); % Load new data
df = df(:, 2:end); % Exclude the first column (e.g., timestamp)
newData = df(1:end, :);
newRows = height(df); % Get number of new data rows
X_new = newData(:, selectfeature); % Extract selected features for inputs
y_new = newData(:, end-3:end); % Extract the last 4 columns as output

% Standardize new data using the original training mean and standard deviation
X_new = (X_new - x_mean) ./ x_sig;
y_new = (y_new - y_mean) ./ y_sig;

%% Step 2: Load pre-trained model
net = md.model;

%% Step 3: Load and standardize old data
opts = detectImportOptions(datastr, 'VariableNamingRule', 'preserve');
opts.SelectedVariableNames = opts.SelectedVariableNames(1:14); % Select the first 14 columns
df = readtable(datastr, opts); % Load old data
df = df(:, 2:end); % Exclude the first column (e.g., timestamp)
allData = df(1:end, :);
X = allData(:, selectfeature); % Extract selected features for inputs
y = allData(:, end-3:end); % Extract the last 4 columns as output

% Standardize old data using the original training mean and standard deviation
X = (X - x_mean) ./ x_sig;
y = (y - y_mean) ./ y_sig;

% Convert data to array format for processing
X = table2array(X);
X_new = table2array(X_new);
y = table2array(y);

% Randomly select the same number of old data samples as the new data
[~, X_old, ~, y_old] = split_dataset(X, y, newRows);
y_old = array2table(y_old);
y_old.Properties.VariableNames = y_new.Properties.VariableNames; % Match column names

%% Step 4: Reshape input data to 4-D format
X = reshape(X', [size(X, 2), 1, 1, size(X, 1)]);
X_new = reshape(X_new', [size(X_new, 2), 1, 1, size(X_new, 1)]);
X_old = reshape(X_old', [size(X_old, 2), 1, 1, size(X_old, 1)]);

% Concatenate old and new data for training
X_mix = cat(4, X_old, X_new);
y_mix = cat(1, y_old, y_new);

y_mix = table2array(y_mix); % Convert to array format

%% Step 5: Split the data into training and validation sets
numElements = size(X_mix, 4);
numLastElements = round(0.2 * numElements); % 20% for validation

% Split data
X_val = X_mix(:, :, :, end - numLastElements + 1:end);
X_train = X_mix(:, :, :, 1:numElements);
y_train = y_mix(1:numElements, :);
y_val = y_mix(end - numLastElements + 1:end, :);

%% Step 6: Define training options
numEpochs = 5;
miniBatchSize = 128;
validationFrequency = floor((1 / 0.1) * numEpochs); % Validation frequency

% Initialize performance tracking
performance = ModelPerformance();

% Custom function to update and save the best model during training
outputFcn = @(info) performance.update(info);

% Training options using 'adam' optimizer
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

%% Step 7: Train the model
[model, traininfo] = trainNetwork(X_train, y_train, net.Layers, options);

%% Step 8: Model prediction and evaluation
% Predictions using the fine-tuned model
y_pred1 = predict(model, X_old);
y_pred2 = predict(model, X_new); 
y_pred3 = predict(model, X);

% Predictions using the old model
y_p1 = predict(net, X_old);
y_p2 = predict(net, X_new);
y_p3 = predict(net, X);

% Reverse standardization for model outputs
y_pred1_orig = y_pred1 .* y_sig + y_mean;
y_pred2_orig = y_pred2 .* y_sig + y_mean;
y_pred3_orig = y_pred3 .* y_sig + y_mean;

% Reverse standardization for old model outputs
y_p1_orig = y_p1 .* y_sig + y_mean;
y_p2_orig = y_p2 .* y_sig + y_mean;
y_p3_orig = y_p3 .* y_sig + y_mean;

% Calculate mean square error (MSE)
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

%% Callback function for data splitting
function [X_train, X_test, y_train, y_test] = split_dataset(X, y, test_size)
    % SPLIT_DATASET Randomly splits data into training and testing sets
    %
    % Parameters:
    %   X: Input features
    %   y: Output labels
    %   test_size: Number of test samples
    %
    % Returns:
    %   X_train, X_test: Training and testing sets for inputs
    %   y_train, y_test: Training and testing sets for outputs

    assert(size(X, 1) == length(y)); % Ensure data sizes match
    N = size(X, 1); % Get dataset size

    % Shuffle data
    idx = randperm(N);

    % Split based on test_size
    test_idx = idx(1:test_size);
    train_idx = idx(test_size+1:end);

    % Create training and test sets
    X_test = X(test_idx, :);
    y_test = y(test_idx, :);
    X_train = X(train_idx, :);
    y_train = y(train_idx, :);
end

%% Return the fine-tuned model and related information
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
