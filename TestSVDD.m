clc; clear; close all;

% Configuration
dataFilePath = 'data/full_data.csv';
modelFilePath = 'models/md1.mat';

% Load SVDD model
if isfile(modelFilePath)
    load(modelFilePath);
else
    error('Model file not found: %s', modelFilePath);
end

% Load data from CSV
if isfile(dataFilePath)
    tbl = readtable(dataFilePath);
else
    error('Data file not found: %s', dataFilePath);
end

% Extract specified columns
try
    datat = table2array(tbl(:, {'HANCHUAN_DOMAIN003_ATR1106', 'HANCHUAN_DOMAIN003_ATR2508'}));
catch
    error('Error in data column extraction. Check column names.');
end

% Data Normalization
datat = 2 * (datat - min(datat)) ./ (max(datat) - min(datat)) - 1;

% Test data and labels
testDatat = datat;
testLabell = ones(size(testDatat, 1), 1);

% Test SVDD model
results1 = svdd.test(testDatat, testLabell);

% Mahalanobis Distance for normal data
ndata = results.data(results.predictedLabel == 1, :);
[mean_vector, cov_matrix] = calculateMahalanobisParams(ndata);
MD = calculateMahalanobisDistance(testDatat, mean_vector, cov_matrix);

% Health Index Calculation
b = 0.0267;
HD = exp(-b * MD);

% Plot Health Degree
figure;
plot(HD);
title('Health Degree Over Time');
xlabel('Time');
ylabel('Health Degree');

% Continuous Anomaly Detection
alarmPoints = detectContinuousAnomalies(results1, 5);

% Visualize Anomalies
plotAnomalies(results1, alarmPoints);

% Helper Functions
function [mean_vector, cov_matrix] = calculateMahalanobisParams(data)
    mean_vector = mean(data);
    cov_matrix = cov(data);
end

function MD = calculateMahalanobisDistance(samples, mean_vector, cov_matrix)
    MD = zeros(size(samples, 1), 1);
    for i = 1:size(samples, 1)
        MD(i) = sqrt((samples(i,:) - mean_vector) / cov_matrix * (samples(i,:) - mean_vector)');
    end
end

function alarmPoints = detectContinuousAnomalies(results, windowSize)
    alarmPoints = zeros(length(results.distance), 1);
    for i = 1:length(results.distance) - windowSize + 1
        if all(results.distance(i:i + windowSize - 1) > results.radius)
            alarmPoints(i:i + windowSize - 1) = 1;
        end
    end
end

function plotAnomalies(results, alarmPoints)
    figure;
    plot(results.distance, 'b'); hold on;
    plot(results.radius * ones(length(results.distance), 1), 'r--');
    plot(find(alarmPoints == 1), results.distance(alarmPoints == 1), 'ro');
    
    startIdx = find(diff([0; alarmPoints; 0]) == 1);
    endIdx = find(diff([0; alarmPoints; 0]) == -1) - 1;
    
    for i = 1:length(startIdx)
        text(startIdx(i), results.distance(startIdx(i)), ...
            sprintf('Alarm from %d to %d', startIdx(i), endIdx(i)), 'Color', 'red');
    end

    title('Distance and Alarm Points');
    xlabel('Index');
    ylabel('Distance');
    legend('Distance', 'Radius', 'Alarm Points');
end
