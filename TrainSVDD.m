clc; clear; close all;
addpath(genpath(pwd)); % Add paths

% Load Data
filename = 'GJOUT_interval.xlsx';
tbl = readtable(filename);
data = table2array([tbl(:,1), tbl(:,2)]);

% Data Normalization
data = 2 * (data - min(data)) ./ (max(data) - min(data)) - 1;

% Split Data
trainData = data(1:floor(0.8 * size(data, 1)), :);
testData = data(floor(0.2 * size(data, 1)) + 1:end, :);

% Define Labels
trainLabel = ones(size(trainData, 1), 1);
testLabel = ones(size(testData, 1), 1);

% Set SVDD Parameters
cost = 0.9;
kernel = BaseKernel('type', 'gaussian', 'gamma', 1.5);

% Optimization Settings
opt.method = 'bayes';
opt.variableName = {'cost', 'gamma'};
opt.variableType = {'real', 'real'};
opt.lowerBound = [10^-2, 2^-6];
opt.upperBound = [10^0, 2^6];
opt.maxIteration = 40;
opt.points = 3;
opt.display = 'on';

% SVDD Parameter Struct
svddParameter = struct('cost', cost, 'kernelFunc', kernel, ...
                       'optimization', opt, 'KFold', 5);

% Create and Train SVDD Model
svdd = BaseSVDD(svddParameter);
svdd.train(trainData, trainLabel);

% Test SVDD Model
results = svdd.test(testData, testLabel);

% Visualization of SVDD Results
svplot = SvddVisualization();
svplot.boundary(svdd);
svplot.distance(svdd, results);
