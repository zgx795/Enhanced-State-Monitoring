%% SVDD Parameter Optimization Demonstration
% This script demonstrates how to perform parameter optimization for SVDD (Support Vector Data Description).
% It includes loading data, preprocessing, setting up SVDD parameters, and performing optimization and testing.

clc;
clear;
close all;

% Add all paths related to the current folder
addpath(genpath(pwd));

%% Load Data
% Load the data from an Excel file.
filename = 'GJOUT_interval.xlsx'; % Specify the path to your xlsx file
tbl = readtable(filename); % Read the entire file into a table
data = [tbl(:,1), tbl(:,2)]; % Extract the first and second columns of data

% Convert the table to an array
data = table2array(data);

%% Data Normalization
% Normalize the data to the range [-1, 1] using min-max normalization.
data = 2 * (data - min(data)) ./ (max(data) - min(data)) - 1;

%% Split Data into Training and Testing Sets
% Split the dataset into training (80%) and testing (20%) sets.
trainData = data(1:floor(0.8 * size(data, 1)), :);
testData = data(floor(0.2 * size(data, 1)) + 1:end, :);

%% Define Labels for Training and Testing Data
% Assume that all historical data is labeled as 'normal' with a value of 1.
% Adjust the labels according to your use case for anomaly detection.
trainLabel = ones(size(trainData, 1), 1); % Labels for training data
testLabel = ones(size(testData, 1), 1);   % Labels for testing data

%% Set SVDD Parameters
% Define the SVDD model parameters, including the kernel type and cost parameter.
cost = 0.9; % Penalty cost parameter for SVDD
kernel = BaseKernel('type', 'gaussian', 'gamma', 1.5); % Gaussian kernel with gamma value of 1.5

%% Set Optimization Settings
% Configure the optimization method and the parameters to be optimized.
opt.method = 'bayes'; % Optimization method: 'bayes', 'ga', 'pso' (Bayesian optimization here)
opt.variableName = {'cost', 'gamma'}; % Variables to optimize
opt.variableType = {'real', 'real'};  % Variable types (real numbers)
opt.lowerBound = [10^-2, 2^-6]; % Lower bounds for the variables
opt.upperBound = [10^0, 2^6];   % Upper bounds for the variables
opt.maxIteration = 40;          % Maximum number of optimization iterations
opt.points = 3;                 % Number of initial points for Bayesian optimization
opt.display = 'on';             % Display optimization progress

%% SVDD Parameter Struct
% Create an SVDD parameter structure, including the cost, kernel, and optimization settings.
svddParameter = struct('cost', cost, ...
                       'kernelFunc', kernel, ...
                       'optimization', opt, ...
                       'KFold', 5); % 5-fold cross-validation is used here

%% Create and Train the SVDD Model
% Instantiate the SVDD model with the defined parameters and train it using the training data and labels.
svdd = BaseSVDD(svddParameter);
svdd.train(trainData, trainLabel);

%% Test the SVDD Model
% Test the SVDD model on the test dataset and return the results.
results = svdd.test(testData, testLabel);

%% Visualization of SVDD Results
% Plot the results of the SVDD model, including the boundary and distance of the test points.
svplot = SvddVisualization(); % Instantiate visualization object
svplot.boundary(svdd);        % Plot the decision boundary
svplot.distance(svdd, results); % Plot the distance of test points from the decision boundary
