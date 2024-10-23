function model_output = train_model(data_str)
% train_model1 - Train a regression model based on provided data and return model output.
%
% Inputs:
%    data_str - Path to the CSV file containing the dataset
%
% Outputs:
%    model_output - Struct containing the trained model and relevant feature statistics

    % Load the pre-saved configuration file
    load('MT_13_Mar_2024_19_31_32.mat');
    
    % Read the dataset from CSV
    opts = detectImportOptions(data_str, 'VariableNamingRule', 'preserve');
    opts.SelectedVariableNames = opts.SelectedVariableNames(1:7); % Select first 7 columns
    dataO1 = readtable(data_str, opts); % Load data
    data1 = dataO1(:, 2:end); % Exclude the first column (usually time or index)

    % Determine data types of the columns (char or numeric)
    test_data = table2cell(dataO1(1, 2:end)); % Convert the first row to cell for type checking
    index_char = find(cellfun(@ischar, test_data)); % Identify columns with char type
    index_double = find(cellfun(@isnumeric, test_data)); % Identify columns with numeric type
    
    %% Handle Numeric Data
    if ~isempty(index_double)
        data_numshuju2 = table2array(data1(:, index_double)); % Extract numeric columns
    else
        data_numshuju2 = [];
    end

    %% Handle Categorical Data (Text)
    data_shuju = [];
    if ~isempty(index_char)
        for j = 1:length(index_char)
            data_get = table2array(data1(:, index_char(j)));
            data_label = unique(data_get); % Get unique labels for categorical data
            for NN = 1:length(data_label)
                idx = find(ismember(data_get, data_label{NN})); % Convert categorical to numeric labels
                data_shuju(idx, j) = NN;
            end
        end
    end

    % Combine numeric and categorical data
    data_all_last = [data_shuju, data_numshuju2];
    label_all_last = [index_char, index_double];
    data = data_all_last;
    data_biao_all = data1.Properties.VariableNames;

    % Extract feature names
    data_biao = data_biao_all(label_all_last);

    %% Interpolate Missing Data
    data_numshuju = data;
    for NN = 1:size(data_numshuju, 2)
        data_test = data_numshuju(:, NN);
        index = isnan(data_test); % Identify missing values
        data_test1 = data_test(~index); % Exclude NaN values
        index_label = 1:length(data_test);
        index_label1 = index_label(~index); % Indices without missing values
        data_all = interp1(index_label1, data_test1, index_label, 'spline'); % Spline interpolation
        dataO(:, NN) = data_all; % Store interpolated data
    end

    %% Feature Selection using Lasso
    A_data1 = dataO;
    select_feature_num = G_out_data.select_feature_num; % Number of features to select
    predict_num = G_out_data.predict_num_set; % Number of predicted points
    index_name = data_biao;

    % Perform Lasso regression to select features
    [B, ~] = lasso(A_data1(:, 1:end - predict_num), A_data1(:, end - predict_num + 1), 'Alpha', 1);
    L_B = (B ~= 0); % Identify non-zero coefficients
    SL_B = sum(L_B);
    [~, index_L1] = min(abs(SL_B - select_feature_num)); % Find index with desired number of features
    feature_need_last = find(L_B(:, index_L1) == 1); % Select the important features

    % Display selected features
    for NN = 1:length(feature_need_last)
        print_index_name{1, NN} = index_name{1, feature_need_last(NN)};
    end
    disp('Selected Features:');
    disp(print_index_name);

    % Prepare data with selected features for training
    data_select = [A_data1(:, feature_need_last), A_data1(:, end - predict_num + 1:end)];

    %% Model Training Parameters
    select_predict_num = G_out_data.select_predict_num; % Number of points to predict
    min_batchsize = G_out_data.min_batchsize; % Minimum batch size for training
    max_epoch_LC = G_out_data.max_epoch_LC; % Maximum epochs for training
    num_BO_iter = G_out_data.num_BO_iter; % Bayesian optimization iterations
    list_cell = G_out_data.list_cell; % List of selected features for training

    %% Model Training Process
    x_mu_all = []; x_sig_all = []; y_mu_all = []; y_sig_all = [];
    for NUM_all = 1:size(data_select, 1)
        data_process = data_select;
        x_feature_label = data_process(:, 1:end - select_predict_num);
        y_feature_label = data_process(:, end - select_predict_num + 1:end);

        % Split data into training, validation, and test sets
        spilt_ri = G_out_data.spilt_ri;
        train_num = round(spilt_ri(1) / sum(spilt_ri) * size(x_feature_label, 1)); % Training set size
        vaild_num = round((spilt_ri(1) + spilt_ri(2)) / sum(spilt_ri) * size(x_feature_label, 1)); % Validation set size
        train_x_feature_label = x_feature_label(1:train_num, :);
        train_y_feature_label = y_feature_label(1:train_num, :);
        vaild_x_feature_label = x_feature_label(train_num + 1:vaild_num, :);
        vaild_y_feature_label = y_feature_label(train_num + 1:vaild_num, :);
        test_x_feature_label = x_feature_label(vaild_num + 1:end, :);
        test_y_feature_label = y_feature_label(vaild_num + 1:end, :);

        % Normalize the data (Z-score normalization)
        x_mu = mean(train_x_feature_label);  x_sig = std(train_x_feature_label);
        train_x_feature_label_norm = (train_x_feature_label - x_mu) ./ x_sig;
        y_mu = mean(train_y_feature_label);  y_sig = std(train_y_feature_label);
        train_y_feature_label_norm = (train_y_feature_label - y_mu) ./ y_sig;
        vaild_x_feature_label_norm = (vaild_x_feature_label - x_mu) ./ x_sig;
        vaild_y_feature_label_norm = (vaild_y_feature_label - y_mu) ./ y_sig;
        test_x_feature_label_norm = (test_x_feature_label - x_mu) ./ x_sig;
        test_y_feature_label_norm = (test_y_feature_label - y_mu) ./ y_sig;

        x_mu_all(NUM_all, :) = x_mu;
        x_sig_all(NUM_all, :) = x_sig;
        y_mu_all(NUM_all, :) = y_mu;
        y_sig_all(NUM_all, :) = y_sig;

        % Reshape data for input to CNN-BiLSTM model
        p_train1 = reshape(train_x_feature_label_norm', size(train_x_feature_label_norm, 2), 1, 1, size(train_x_feature_label_norm, 1));
        p_vaild1 = reshape(vaild_x_feature_label_norm', size(vaild_x_feature_label_norm, 2), 1, 1, size(vaild_x_feature_label_norm, 1));
        p_test1 = reshape(test_x_feature_label_norm', size(test_x_feature_label_norm, 2), 1, 1, size(test_x_feature_label_norm, 1));

        % Define model options
        opt.methods = 'CNN-BiLSTM';
        opt.maxEpochs = max_epoch_LC;
        opt.miniBatchSize = min_batchsize;
        opt.LR = 'adam'; % Learning rate algorithm
        opt.isUseBiLSTMLayer = true;
        opt.isUseDropoutLayer = true;
        opt.DropoutValue = 0.2;
        opt.optimVars = [optimizableVariable('NumOfLayer', [1 2], 'Type', 'integer'),
                         optimizableVariable('NumOfUnits', [50 200], 'Type', 'integer'),
                         optimizableVariable('isUseBiLSTMLayer', [1 2], 'Type', 'integer'),
                         optimizableVariable('InitialLearnRate', [1e-2 1], 'Transform', 'log'),
                         optimizableVariable('L2Regularization', [1e-10 1e-2], 'Transform', 'log')];

        % Train the model
        data_struct1.XTr = p_train1; data_struct1.YTr = train_y_feature_label_norm(:, list_cell{1, NUM_all});
        data_struct1.XTs = p_test1; data_struct1.YTs = test_y_feature_label_norm(:, list_cell{1, NUM_all});
        data_struct1.XVl = p_vaild1; data_struct1.YVl = vaild_y_feature_label_norm(:, list_cell{1, NUM_all});

        [opt, data_struct1] = Optimize_CNNS(opt, data_struct1); % Bayesian optimization
        [opt, data_struct1, ~] = EvaluationData(opt, data_struct1); % Model evaluation

        % Predict and store results
        Mdl = data_struct1.BiLSTM.Net;
        y_train_predict_norm_roll = predict(Mdl, p_train1, 'MiniBatchSize', opt.miniBatchSize);
        y_vaild_predict_norm_roll = predict(Mdl, p_vaild1, 'MiniBatchSize', opt.miniBatchSize);
        y_test_predict_norm_roll = predict(Mdl, p_test1, 'MiniBatchSize', opt.miniBatchSize);
    end

    %% Model Output
    model_output.model = Mdl;
    model_output.feature_need_last = feature_need_last;
    model_output.x_mu = x_mu;
    model_output.x_sig = x_sig;
    model_output.y_mu = y_mu;
    model_output.y_sig = y_sig;

end
