%% Main Function for Model Training and Fine-tuning

clc;
clear;
close all;

%% Step 1: Data Loading and Model Training
% Replace with relative path or user input for flexibility
data_str = 'data/TEST01.xlsx'; % Path to training data

% Train the model using the initial dataset
model_output = train_model(data_str);

%% Step 2: Predict on New Data
% Load test data for model evaluation
data_str1 = 'data/test_data.csv'; % Path to testing data
test_data = readtable(data_str1, 'VariableNamingRule', 'preserve'); % Load test data

% Remove timestamp column
test_data(:,1) = [];

% Feature selection and normalization
test_data_norm = test_data(:, model_output.feature_need_last); % Select relevant features
test_data_norm = (test_data_norm - model_output.x_mu) ./ model_output.x_sig; % Normalize features
test_data_norm = table2array(test_data_norm); % Convert table to array

% Reshape data for model input
reshape_testdata = reshape(test_data_norm', size(test_data_norm, 2), 1, 1, size(test_data_norm, 1));

% Predict using the trained model
y_pre_norm = predict(model_output.Mdl, reshape_testdata);

% Reverse normalization (denormalize)
y_pre_norm = y_pre_norm .* model_output.y_sig + model_output.y_mu;

%% Step 3: Model Fine-tuning on New Data
% Fine-tune the model using new datasets
data_str = 'data/original_training_data.csv'; % Path to original dataset
data_str2 = 'data/new_data_updates/'; % Directory containing new data updates

% Initialize variables to track best model
best_model_performance = inf;
best_model = model_output.model;

% Table to store model performance
model_performance_table = table();

% List all update data files in the directory
files = dir(fullfile(data_str2, '*.csv'));

% Sort files based on their file name number
file_numbers = cellfun(@(x) str2double(regexp(x, '\d*', 'match')), {files.name});
[~, sorted_indices] = sort(file_numbers);
files = files(sorted_indices);

% Iterate over each new data file to fine-tune the model
for k = 1:length(files)
    file_name = fullfile(files(k).folder, files(k).name);
    
    % Fine-tune the model using new data
    finetune_model = finetunemd1(model_output, data_str, file_name, ...
        model_output.feature_need_last, model_output.x_mu, model_output.x_sig, model_output.y_mu, model_output.y_sig);
    
    % Save the fine-tuned model
    model_file_name = fullfile(files(k).folder, ['model_', num2str(k), '.mat']);
    save(model_file_name, 'finetune_model');
    
    % Evaluate the old model performance on new validation data
    if k >= 1
        y_pred_old = predict(best_model, finetune_model.X_val);
        y_pred_old_prig = y_pred_old .* model_output.y_sig + model_output.y_mu;
        old_model_mse = immse(finetune_model.y_val, double(y_pred_old_prig));
    else
        old_model_mse = inf;
    end
    
    % Record model performance
    model_performance_table = [model_performance_table; table(k, old_model_mse, finetune_model.mse, 'VariableNames', {'ModelNumber', 'OldMSE', 'NewMSE'})];
    
    % Update the best model if new model is better
    if finetune_model.mse < old_model_mse
        best_model_performance = finetune_model.mse;
        best_model = finetune_model.model;
        
        % Save the best model
        best_model_file_name = fullfile(files(k).folder, ['best_model_', num2str(k), '.mat']);
        save(best_model_file_name, 'best_model');
    end
end

% Save model performance table as CSV
writetable(model_performance_table, fullfile(files(1).folder, 'model_performance.csv'));

%% Step 4: Model Prediction and Comparison
% Load and prepare new test data
data_str3 = 'data/test_data.csv'; % Path to new test data
test_data = readtable(data_str3, 'VariableNamingRule', 'preserve');
test_data(:,1) = []; % Remove timestamp column
y_test = test_data(:, [10, 11, 12, 13]); % Extract actual values

% Normalize test data
test_data_norm = test_data(:, model_output.feature_need_last);
test_data_norm = (test_data_norm - model_output.x_mu) ./ model_output.x_sig;
test_data_norm = table2array(test_data_norm);

% Reshape data for model input
reshape_testdata = reshape(test_data_norm', size(test_data_norm, 2), 1, 1, size(test_data_norm, 1));

% Predict using the old and new models
y_pre_norm = predict(model_output.model, reshape_testdata);
y_pre_norm = y_pre_norm .* model_output.y_sig + model_output.y_mu;

y_pre_norm1 = predict(best_model, reshape_testdata);
y_pre_norm1 = y_pre_norm1 .* model_output.y_sig + model_output.y_mu;

% Compare model performance
column_to_compare = 1; % Index of the column to compare
y_test_column = table2array(y_test(:, column_to_compare));
y_pre_norm_column = y_pre_norm(:, column_to_compare);
y_pre_norm1_column = y_pre_norm1(:, column_to_compare);

mse_old_model = mean((y_test_column - y_pre_norm_column).^2); % Old model MSE
mse_new_model = mean((y_test_column - y_pre_norm1_column).^2); % New model MSE

% Plot comparison
figure;
subplot(1, 2, 1);
plot(y_test_column, 'k', 'DisplayName', 'Actual');
hold on;
plot(y_pre_norm_column, 'r--', 'DisplayName', 'Old Model Prediction');
plot(y_pre_norm1_column, 'b-.', 'DisplayName', 'New Model Prediction');
hold off;
title('Actual vs. Predicted');
xlabel('Sample Index');
ylabel('Value');
legend('show');

subplot(1, 2, 2);
plot(y_test_column - y_pre_norm_column, 'r--', 'DisplayName', ['Old Model Error, MSE: ', num2str(mse_old_model)]);
hold on;
plot(y_test_column - y_pre_norm1_column, 'b-.', 'DisplayName', ['New Model Error, MSE: ', num2str(mse_new_model)]);
hold off;
title('Prediction Error');
xlabel('Sample Index');
ylabel('Error');
legend('show');
