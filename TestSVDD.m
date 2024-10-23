%% This script uses a trained SVDD model to perform anomaly detection and calculate the health index of equipment over time.

clc;
clear;
close all;

%% Configuration Section
% Define file paths for data and trained model
dataFilePath = 'data/full_data.csv';  % Configured file path for data
modelFilePath = 'models/md1.mat';  % Configured file path for the SVDD model

% Load trained SVDD model
if isfile(modelFilePath)
    load(modelFilePath);  % Load the trained model
else
    error('Model file not found: %s', modelFilePath);  % Error handling if the model is not found
end

% Read CSV file
if isfile(dataFilePath)
    tbl = readtable(dataFilePath);  % Load data from CSV file
else
    error('Data file not found: %s', dataFilePath);  % Error handling if data file is not found
end

% Extract specific columns for analysis
try
    datat = table2array(tbl(:,{'HANCHUAN_DOMAIN003_ATR1106', 'HANCHUAN_DOMAIN003_ATR2508'}));  % Extract specified columns
catch
    error('Error in data column extraction. Check column names.');  % Error handling for incorrect column names
end

%% Data Normalization
% Normalize the data to the range [-1, 1]
datat = 2 * (datat - min(datat)) ./ (max(datat) - min(datat)) - 1;

% Use the entire dataset as test data by default
testDatat = datat(1:end, :);

% Provide test labels, assuming all data is labeled as normal (1)
testLabell = ones(size(testDatat, 1), 1);

%% Perform Testing with the Trained Model
% Test the SVDD model with the test data
results1 = svdd.test(testDatat, testLabell);

%% Mahalanobis Distance-Based Health Index Calculation
% Extract the normal data points from the trained model results
ndata = results.data(results.predictedLabel == 1, :);  % Extract the data points labeled as normal

% Calculate mean vector and covariance matrix for normal data
[mean_vector, cov_matrix] = calculateMahalanobisParams(ndata);

% Calculate Mahalanobis distances for the test data
MD = calculateMahalanobisDistance(testDatat, mean_vector, cov_matrix);

%% Health Index Calculation
% Define coefficient b based on predefined threshold
b = 0.0267;  % Coefficient used to calculate health index based on Mahalanobis distance

% Calculate health degree (HD) using an exponential decay function
HD = exp(-b * MD);

% Plot the health degree over time
figure;
plot(HD);
title('Health Degree Over Time');
xlabel('Time');
ylabel('Health Degree');

%% Continuous Anomaly Detection and Alarm
% Detect continuous anomalies over a specified window size
alarmPoints = detectContinuousAnomalies(results1, 5);

% Visualize the anomalies and the distance to the decision boundary
plotAnomalies(results1, alarmPoints);

%% Helper Functions

% Function to calculate mean vector and covariance matrix for Mahalanobis distance
function [mean_vector, cov_matrix] = calculateMahalanobisParams(data)
    % Calculate the mean vector of the data
    mean_vector = mean(data);
    
    % Calculate the covariance matrix of the data
    cov_matrix = cov(data);
end

% Function to calculate Mahalanobis distance for each sample
function MD = calculateMahalanobisDistance(samples, mean_vector, cov_matrix)
    % Initialize the Mahalanobis distance array
    MD = zeros(size(samples, 1), 1);
    
    % Loop through each sample and calculate Mahalanobis distance
    for i = 1:size(samples, 1)
        MD(i) = sqrt((samples(i,:) - mean_vector) / cov_matrix * (samples(i,:) - mean_vector)');
    end
end

% Function to detect continuous anomalies in the results based on a window size
function alarmPoints = detectContinuousAnomalies(results, windowSize)
    % Initialize an array to store anomaly points
    alarmPoints = zeros(length(results.distance), 1);
    
    % Loop through distance values and detect continuous anomalies
    for i = 1:length(results.distance) - windowSize + 1
        % Check if all points in the window exceed the decision boundary (radius)
        if all(results.distance(i:i + windowSize - 1) > results.radius)
            alarmPoints(i:i + windowSize - 1) = 1;  % Mark these points as alarms
        end
    end
end

% Function to visualize anomalies and distance to the decision boundary
function plotAnomalies(results, alarmPoints)
    % Plot the distance values
    figure;
    plot(results.distance, 'b');  % Plot distance with blue line
    hold on;
    
    % Plot the decision boundary (radius)
    plot(results.radius * ones(length(results.distance), 1), 'r--');  % Red dashed line for radius
    
    % Mark anomaly points with red circles
    plot(find(alarmPoints == 1), results.distance(alarmPoints == 1), 'ro');  
    
    % Find and annotate the start and end of each anomaly sequence
    startIdx = find(diff([0; alarmPoints; 0]) == 1);
    endIdx = find(diff([0; alarmPoints; 0]) == -1) - 1;
    
    % Annotate each anomaly sequence with text
    for i = 1:length(startIdx)
        text(startIdx(i), results.distance(startIdx(i)), ...
            sprintf('Alarm from %d to %d', startIdx(i), endIdx(i)), 'Color', 'red');
    end

    % Add title and labels to the plot
    title('Distance and Alarm Points');
    xlabel('Index');
    ylabel('Distance');
    legend('Distance', 'Radius', 'Alarm Points');
end
