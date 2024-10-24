function model_output = train_model(data_str)
% train_model - Train a regression model using VMD, Random Forest, and CNN-BiLSTM
%
% Inputs:
%    data_str - Path to the file containing the dataset
%
% Outputs:
%    model_output - Struct containing the trained model and feature statistics

    %% Initialize environment and load configuration
    clc; clear; close all;
    rng(3); % Set random seed for reproducibility
 
    % Read the dataset from the specified path
    dataO1 = readtable(data_str, 'VariableNamingRule', 'preserve');
    data1 = dataO1(:, 2:end); % Exclude the first column
    test_data = table2cell(dataO1(1, 2:end)); % Convert the first row to cell array

    %% Determine data types of columns
    index_la = determine_data_types(test_data);
    index_char = find(index_la == 1); % Indices of char columns
    index_double = find(index_la == 2); % Indices of double columns

    %% Handle numeric and categorical data
    [data_numshuju2, data_shuju] = process_data_types(data1, index_double, index_char);

    %% Combine numeric and categorical data
    data_all_last = [data_shuju, data_numshuju2];
    label_all_last = [index_char, index_double];
    data = data_all_last;
    data_biao_all = data1.Properties.VariableNames;
    data_biao = data_biao_all(label_all_last);

    %% Interpolate missing data
    dataO = interpolate_missing_data(data);

    %% Perform feature selection using Random Forest
    [data_select, feature_need_last, print_index_name] = feature_selection(dataO, data_biao, 1);

    %% Apply VMD to the target variable for noise filtering
    data_select = apply_vmd(data_select);

    %% Split and normalize data
    [train_x, train_y, valid_x, valid_y, test_x, test_y, x_mu, x_sig, y_mu, y_sig] = ...
        split_and_normalize(data_select, 2, [8,1,1]);

    %% Set up model options and data structure
    [opt, data_struct1] = define_model_options(data_select(:, 1:end-2), ...
                                               data_select(:, end-2+1:end), ...
                                               train_x, train_y, valid_x, valid_y, ...
                                               test_x, test_y, 70, ...
                                               26784, 30, ...
                                               2, 0, ...
                                               [1,2]);

    %% Train the CNN-BiLSTM model using Bayesian optimization
    [opt, data_struct1] = OptimizeBaye_CNNS1(opt, data_struct1);

    %% Evaluate the trained model
    [~, data_struct1, ~, Loss] = EvaluationData2(opt, data_struct1);

    %% Retrieve the trained model
    Mdl = data_struct1.BiLSTM.Net;

    %% Model Output
    model_output = struct();
    model_output.model = Mdl;
    model_output.loss = Loss;
    model_output.feature_need_last = feature_need_last;
    model_output.print_index_name = print_index_name;
    model_output.x_mu = x_mu;
    model_output.x_sig = x_sig;
    model_output.y_mu = y_mu;
    model_output.y_sig = y_sig;
end

function index_la = determine_data_types(test_data)
% determine_data_types - Determine data types of columns


    index_la = zeros(1, length(test_data)); % Initialize index array
    for i = 1:length(test_data)
        if ischar(test_data{1, i})
            index_la(i) = 1; % char type
        elseif isnumeric(test_data{1, i})
            index_la(i) = 2; % double type
        else
            index_la(i) = 0; % other types
        end
    end
end

function [data_numshuju2, data_shuju] = process_data_types(data1, index_double, index_char)
% process_data_types - Process numeric and categorical data


    % Handle numeric data
    if ~isempty(index_double)
        data_numshuju2 = table2array(data1(:, index_double)); % Extract numeric data
    else
        data_numshuju2 = [];
    end

    % Handle categorical data
    data_shuju = [];
    if ~isempty(index_char)
        for j = 1:length(index_char)
            data_get = table2array(data1(:, index_char(j)));
            data_label = unique(data_get);
            for NN = 1:length(data_label)
                idx = find(ismember(data_get, data_label{NN}));
                data_shuju(idx, j) = NN; % Convert categorical to numeric labels
            end
        end
    end
end

function dataO = interpolate_missing_data(data)
% interpolate_missing_data - Interpolate missing data using spline interpolation


    dataO = zeros(size(data));
    for NN = 1:size(data, 2)
        data_test = data(:, NN);
        index = isnan(data_test); % Identify missing values
        data_test1 = data_test(~index); % Exclude NaN values
        index_label = 1:length(data_test);
        index_label1 = index_label(~index);
        data_all = interp1(index_label1, data_test1, index_label, 'spline'); % Spline interpolation
        dataO(:, NN) = data_all;
    end
end

function [data_select, feature_need_last, print_index_name] = feature_selection(dataO, data_biao, ~)
% feature_selection - Perform feature selection using Random Forest


    select_feature_num =6; %G_out_data.select_feature_num; % Number of features to select
    predict_num = 2; % Number of points to predict
    index_name = data_biao;

    % Train Random Forest model to evaluate feature importance
    RF_Model = TreeBagger(50, dataO(:, 1:end - predict_num), dataO(:, end - predict_num + 1), ...
                          'Method', 'regression', 'OOBPredictorImportance', 'on');
    imp = RF_Model.OOBPermutedPredictorDeltaError; % Get feature importance scores

    % Sort and select the top features
    [~, sort_feature] = sort(imp, 'descend');
    feature_need_last = sort_feature(1:select_feature_num);

    % Display selected features
    print_index_name = index_name(feature_need_last);
    disp('Selected Features:');
    disp(print_index_name);

    % Prepare data with selected features
    data_select = [dataO(:, feature_need_last), dataO(:, end - predict_num + 1:end)];
end

function data_select = apply_vmd(data_select)
% apply_vmd - Apply VMD (Variational Mode Decomposition) for noise filtering


    num_IMFs = 9; % Number of IMFs for decomposition
    [imf, ~] = vmd(data_select(:, end), 'NumIMF', num_IMFs); % Apply VMD on the target variable
    y_denoise = sum(imf(:, 2:num_IMFs), 2);

    % Plot original vs denoised data
    figure;
    plot(y_denoise); hold on;
    plot(data_select(:, end), 'r');
    legend('Denoised', 'Original');
    title('VMD Denoising');
    set(gca, 'FontSize', 11, 'LineWidth', 1);
    box off;

    % Replace original data with denoised data
    data_select(:, end) = y_denoise;
end

function [train_x, train_y, valid_x, valid_y, test_x, test_y, x_mu, x_sig, y_mu, y_sig] = ...
    split_and_normalize(data_select, select_predict_num, spilt_ri)
% split_and_normalize - Split the data into training, validation, and test sets, and normalize


    % Split the data
    x_feature_label = data_select(:, 1:end - select_predict_num);
    y_feature_label = data_select(:, end - select_predict_num + 1:end);
    train_num = round(spilt_ri(1) / sum(spilt_ri) * size(x_feature_label, 1));
    valid_num = round((spilt_ri(1) + spilt_ri(2)) / sum(spilt_ri) * size(x_feature_label, 1));

    train_x = x_feature_label(1:train_num, :);
    train_y = y_feature_label(1:train_num, :);
    valid_x = x_feature_label(train_num + 1:valid_num, :);
    valid_y = y_feature_label(train_num + 1:valid_num, :);
    test_x = x_feature_label(valid_num + 1:end, :);
    test_y = y_feature_label(valid_num + 1:end, :);

    % Z-score normalization
    x_mu = mean(train_x);
    x_sig = std(train_x);
    train_x = (train_x - x_mu) ./ x_sig;
    valid_x = (valid_x - x_mu) ./ x_sig;
    test_x = (test_x - x_mu) ./ x_sig;

    y_mu = mean(train_y);
    y_sig = std(train_y);
    train_y = (train_y - y_mu) ./ y_sig;
    valid_y = (valid_y - y_mu) ./ y_sig;
    test_y = (test_y - y_mu) ./ y_sig;
end

function [opt, data_struct1] = define_model_options(x_feature_label, y_feature_label, ...
                                                    train_x, train_y, valid_x, valid_y, ...
                                                    test_x, test_y, max_epoch_LC, ...
                                                    min_batchsize, num_BO_iter, ...
                                                    roll_num_in, attention_label, attention_head)
% define_model_options - Set up model options and data structure for CNN-BiLSTM training


    %% Set up model options for CNN-BiLSTM
    opt = struct();
    opt.methods = 'CNN-BiLSTM';  % Model type
    opt.maxEpochs = max_epoch_LC;  % Maximum number of training epochs
    opt.miniBatchSize = min_batchsize;  % Minimum batch size
    opt.executionEnvironment = 'auto';  % Execution environment ('cpu', 'gpu', 'auto')
    opt.LR = 'adam';  % Learning rate algorithm ('sgdm', 'rmsprop', 'adam')
    opt.trainingProgress = 'none';  % Training progress display ('training-progress', 'none')
    opt.isUseBiLSTMLayer = true;  % Use BiLSTM layer
    opt.isUseDropoutLayer = true;  % Use dropout layer
    opt.DropoutValue = 0.2;  % Dropout value
    opt.roll_num_in = roll_num_in;  % Rolling prediction time length

    % Define optimization variables for Bayesian optimization
    opt.optimVars = [
        optimizableVariable('NumOfLayer', [1 2], 'Type', 'integer'), ...
        optimizableVariable('NumOfUnits', [50 200], 'Type', 'integer'), ...
        optimizableVariable('isUseBiLSTMLayer', [1 2], 'Type', 'integer'), ...
        optimizableVariable('InitialLearnRate', [1e-2 1], 'Transform', 'log'), ...
        optimizableVariable('L2Regularization', [1e-10 1e-2], 'Transform', 'log') ...
    ];

    % Additional options for Bayesian optimization
    opt.isUseOptimizer = true;
    opt.isSaveOptimizedValue = false;
    opt.isSaveBestOptimizedValue = true;
    opt.MaxOptimizationTime = 14 * 60 * 60;  % Maximum optimization time (14 hours)
    opt.MaxItrationNumber = num_BO_iter;  % Maximum number of iterations for optimization
    opt.isDispOptimizationLog = true;  % Display optimization log

    %% Set up data structure for training
    data_struct1 = struct();
    data_struct1.X = x_feature_label;
    data_struct1.Y = y_feature_label;
    data_struct1.XTr = reshape(train_x', size(train_x, 2), 1, 1, size(train_x, 1));
    data_struct1.YTr = train_y;
    data_struct1.XVl = reshape(valid_x', size(valid_x, 2), 1, 1, size(valid_x, 1));
    data_struct1.YVl = valid_y;
    data_struct1.XTs = reshape(test_x', size(test_x, 2), 1, 1, size(test_x, 1));
    data_struct1.YTs = test_y;
    data_struct1.attention_label = attention_label;
    data_struct1.attention_head = attention_head;
end
