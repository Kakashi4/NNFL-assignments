%% Neuro Fuzzy Inference System
clc;
close all;
clear all;

% Load, shuffle and normalize data
[data4,a,b] = xlsread('data4.xlsx');
data4(:, end) = round(data4(:, end));
data4(:, 1:end-1) = normalize(data4(:, 1:end-1), 2);
data = data4(randperm(size(data4, 1)), :);

% Holdout cross-validation
X = data(1:105, :);
X_val = data(106:end, :);


% Train a system and predict on the validation set
fis = anfis(X);
anfisOutput = evalfis(fis, X_val(:, 1:end-1));

% Generate confusion matrix
out = round(anfisOutput);
confusion_matrix = zeros(3, 3);

targets = X_val(:, end);

for i = 1:length(anfisOutput)
    confusion_matrix(targets(i), out(i)) = confusion_matrix(targets(i), out(i)) + 1;
end

accuracy = trace(confusion_matrix)/45;

