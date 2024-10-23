%{
    Demonstration of SVDD parameter optimization.
    This script performs the SVDD model training and testing, including
    parameter optimization using Bayesian optimization.
%}

clc;
clear;
close all;
addpath(genpath(pwd));

% ---------- Configuration Section ----------
% Load data from external source
dataFile = fullfile('data', 'GJOUT_interval_30.xlsx'); % Parameterized file path
if ~isfile(dataFile)
    error('Data file not found: %s', dataFile);
end

% Optimization settings
cost = 0.9;
kernelType = 'gaussian';
gammaValue = 1.5;
optMethod = 'bayes'; % bayes, ga, pso
maxIterations = 40;
optPoints = 3;
kFold = 5; % KFold cross-validation setting

% Parameter bounds for optimization
opt.variableName = {'cost', 'gamma'};
opt.variableType = {'real', 'real'};
opt.lowerBound = [10^-2, 2^-6];
opt.upperBound = [10^0, 2^6];

% ---------- Data Loading and Preprocessing ----------
% Read data from Excel file
tbl = readtable(dataFile);
data = [tbl(:,1), tbl(:,2)]; % Extract first two columns

% Convert table to array
data = table2array(data);

% Normalize the data to range [-1, 1]
data = 2 * (data - min(data)) ./ (max(data) - min(data)) - 1;

% Split data into training (80%) and testing (20%) sets
trainData = data(1:floor(0.8 * size(data, 1)), :);
testData = data(floor(0.2 * size(data, 1)) + 1:end, :);

% Assume all labels are 1 (normal); customize for different datasets
trainLabel = ones(size(trainData, 1), 1);
testLabel = ones(size(testData, 1), 1);

% ---------- SVDD Model Setup and Optimization ----------
% Define kernel function
kernel = BaseKernel('type', kernelType, 'gamma', gammaValue);

% Set optimization parameters
opt.method = optMethod;
opt.maxIteration = maxIterations;
opt.points = optPoints;
opt.display = 'on'; % Display optimization process

% SVDD parameter structure
svddParameter = struct('cost', cost, ...
                       'kernelFunc', kernel, ...
                       'optimization', opt, ...
                       'KFold', kFold); 

% Create an SVDD object
svdd = BaseSVDD(svddParameter);

% ---------- Model Training and Testing ----------
% Train the SVDD model
svdd.train(trainData, trainLabel);

% Test the SVDD model
results = svdd.test(testData, testLabel);

% ---------- Visualization ----------
% Create visualization object
svplot = SvddVisualization();

% Plot decision boundary
svplot.boundary(svdd);

% Plot distance of test data to boundary
svplot.distance(svdd, results);

% ---------- Logging and Documentation ----------
% Save the model, results, and figures if needed
% You can add saving functionality here if required
