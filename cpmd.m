%% Algorithm Comparison Plot
% This script compares the performance of multiple regression and time series prediction algorithms.
% The data consists of predicted and actual values from various models (exported directly from the toolbox).
% The key modification is to ensure the loaded models and their corresponding `str` names are aligned.
% The script needs to be run in the directory where the model files are located.

% Models:
% GRU: '25_Jan_10_46_14 train_result_train_vaild_test.mat'
% LSTM: '25_Jan_10_55_33 train_result_train_vaild_test.mat'
% BiLSTM: '25_Jan_11_05_22 train_result_train_vaild_test.mat'
% CNN: '25_Jan_11_09_15 train_result_train_vaild_test.mat'
% CNN-BiLSTM: '25_Jan_11_13_27 train_result_train_vaild_test.mat'
% Bayesian Optimized CNN-BiLSTM: '25_Jan_12_13_03 train_result_train_vaild_test.mat'

%% Step-by-Step Execution: Load models, plot predictions, and compare errors

clc;
clear;
close all;

% Initialize variables
data_pre_all = [];  % Store predicted data
selectnumber = 3;  % Select which column of predictions to use

% Load model results and aggregate predicted data
% Correspond the loaded models with their respective names in `str`
load('bayescnnbistm.mat');
data1 = data_Oriny_prey.y_test_predict(:, selectnumber); 
data_pre_all = [data_pre_all, data1];
data_true = data_Oriny_prey.test_y;

load('cnnbilstm.mat');
data2 = data_Oriny_prey.y_test_predict(:, selectnumber);
data_pre_all = [data_pre_all, data2];

load('bilstm.mat');
data3 = data_Oriny_prey.y_test_predict(:, selectnumber);
data_pre_all = [data_pre_all, data3];

load('lstm.mat');
data4 = data_Oriny_prey.y_test_predict(:, selectnumber);
data_pre_all = [data_pre_all, data4];

load('gru.mat');
data5 = data_Oriny_prey.y_test_predict(:, selectnumber);
data_pre_all = [data_pre_all, data5];

load('cnn.mat');
data6 = data_Oriny_prey.y_test_predict(:, selectnumber);
data_pre_all = [data_pre_all, data6];

% Define legend for the plots
str = {'Actual Value', 'BO-CNN-BiLSTM', 'CNN-BiLSTM', 'BiLSTM', 'LSTM', 'GRU', 'CNN'};

%% Plot actual and predicted values

figure('Units', 'pixels', 'Position', [300 300 860 375]);

% Generate time ticks for the x-axis
start_time = datetime('2022-12-28 21:36:00');
end_time = datetime('2022-12-31 23:59:50');
num_intervals = 5;

% Create time intervals for the plot
time_ticks = linspace(datenum(start_time), datenum(end_time), num_intervals + 1);
time_ticks = datetime(time_ticks, 'ConvertFrom', 'datenum');

% Plot actual values (dashed line)
plot(data_true(:, selectnumber), '--');
hold on;

% Plot predicted values for each model
for i = 1:size(data_pre_all, 2)
    plot(data_pre_all(:, i));
    hold on;
end

% Add legend and formatting
legend(str);
set(gca, 'FontSize', 10, 'LineWidth', 1.2);
box off;
legend box off;

% Set x-axis ticks and labels as time points
xticks(linspace(1, 26784, num_intervals + 1));  % Assume 26784 total data points
xticklabels(datestr(time_ticks, 'mmm-dd HH:MM'));

% Add year label at the bottom right corner
text(0.9, -0.15, '2022', 'Units', 'normalized', 'HorizontalAlignment', 'right', 'VerticalAlignment', 'bottom');

%% Plot error metrics comparison (MAE, MAPE, RMSE) for each model

Test_all = [];
for j = 1:size(data_pre_all, 2)
    y_test_predict = data_pre_all(:, j);
    test_y = data_true(:, selectnumber);
    
    test_MAE = sum(abs(y_test_predict - test_y)) / length(test_y);
    test_MAPE = sum(abs((y_test_predict - test_y) ./ test_y)) / length(test_y);
    test_MSE = (sum((y_test_predict - test_y).^2) / length(test_y));
    test_RMSE = sqrt(sum((y_test_predict - test_y).^2) / length(test_y));
    test_R2 = 1 - (norm(test_y - y_test_predict)^2 / norm(test_y - mean(test_y))^2);
    
    Test_all = [Test_all; test_MAE, test_MAPE, test_MSE, test_RMSE, test_R2];
end

% Convert results to a table for better display
str1 = str(2:end);
str2 = {'MAE', 'MAPE', 'MSE', 'RMSE', 'R2'};
data_out = array2table(Test_all, 'VariableNames', str2, 'RowNames', str1);
disp(data_out);

%% Plot bar graph for error metrics

color = [0.1569, 0.4706, 0.7098; 
         0.6039, 0.7882, 0.8588; 
         0.9725, 0.6745, 0.5490; 
         0.8549, 0.9373, 0.8275; 
         0.7451, 0.7216, 0.8627; 
         0.7843, 0.1412, 0.1373; 
         1.0000, 0.5333, 0.5176];

figure('Units', 'pixels', 'Position', [300 300 660 375]);

% Select specific error metrics to plot
plot_data_t = Test_all(:, [1, 2, 4])';  % Select MAE, MAPE, RMSE
b = bar(plot_data_t, 0.8);  % Plot bar graph
hold on;

% Customize colors for each model
for i = 1:size(plot_data_t, 2)
    b(i).FaceColor = color(i, :);
    b(i).EdgeColor = [0.6353, 0.6314, 0.6431];
    b(i).LineWidth = 1.2;
end

% Add separation lines between bars
for i = 1:size(plot_data_t, 1) - 1
    xilnk = (x_data(i, end) + x_data(i + 1, 1)) / 2;
    xline(xilnk, '--', 'LineWidth', 1.2);
    hold on;
end

% Finalize plot
ax = gca;
legend(b, str1, 'Location', 'best');
ax.XTickLabels = {'MAE', 'MAPE', 'RMSE'};
set(gca, "FontSize", 12, "LineWidth", 2);
box off;
legend box off;

%% Error density scatter plot

figure('Units', 'pixels', 'Position', [150 150 920 500]);
for i = 1:5
    subplot(2, 3, i);
    n = 50;
    X = double(data_true(:, selectnumber));
    Y = double(data_pre_all(:, i));
    M = polyfit(X, Y, 1);  % Linear fit
    Y1 = polyval(M, X);
    
    % Kernel density estimation
    XList = linspace(min(X), max(X), n);
    YList = linspace(min(Y), max(Y), n);
    [XMesh, YMesh] = meshgrid(XList, YList);
    F = ksdensity([X, Y], [XMesh(:), YMesh(:)]);
    ZMesh = reshape(F, size(XMesh));
    H = interp2(double(XMesh), double(YMesh), double(ZMesh), X, Y);
    
    % Scatter plot with density color
    scatter(data_true(:, selectnumber), data_pre_all(:, i), 35, 'filled', 'CData', H, 'MarkerFaceAlpha', .5);
    hold on;
    
    % Plot regression line
    plot(X(1:10:end), Y1(1:10:end), '--', 'LineWidth', 1.2);
    
    % Customize plot
    str_label = [str1{1, i}, ' ', 'R2=', num2str(Test_all(i, end))];
    title(str_label);
    set(gca, "FontSize", 10, "LineWidth", 1.5);
    xlabel('True');
    ylabel('Predicted');
end

%% Radar chart for error comparison

figure('Units', 'pixels', 'Position', [150 150 520 500]);
Test_all1 = Test_all ./ sum(Test_all);  % Normalize all metrics
Test_all1(:, end) = 1 - Test_all(:, end);  % Adjust for R2
RC = radarChart(Test_all1);
str3 = {'A-MAE', 'A-MAPE', 'A-MSE', 'A-RMSE', '1-R2'};
RC.PropName = str3;
RC.ClassName = str1;
RC = RC.draw(); 
RC.legend();

% Set colors for radar chart
colorList = [78 101 155;
             138 140 191;
             184 168 207;
             231 188 198;
             253 207 158;
             239 164 132;
             182 118 108] / 255;

for n = 1:RC.ClassNum
    RC.setPatchN(n, 'Color', colorList(n, :), 'MarkerFaceColor', colorList(n, :));
end

%% Compass plot for error comparison

figure('Units', 'pixels', 'Position', [150 150 920 600]);
t = tiledlayout('flow', 'TileSpacing', 'compact');
for i = 1:length(Test_all(:, 1))
    nexttile;
    th1 = linspace(2*pi/length(Test_all(:, 1))/2, 2*pi-2*pi/length(Test_all(:, 1))/2, length(Test_all(:, 1)));
    r1 = Test_all(:, i)';
    [u1, v1] = pol2cart(th1, r1);
    M = compass(u1, v1);
    
    % Customize plot
    for j = 1:length(Test_all(:, 1))
        M(j).LineWidth = 2;
        M(j).Color = colorList(j, :);
    end
    
    title(str2{i});
    set(gca, "FontSize", 10, "LineWidth", 1);
end

legend(M, str1, "FontSize", 10, "LineWidth", 1, 'Box', 'off', 'Location', 'southoutside');
