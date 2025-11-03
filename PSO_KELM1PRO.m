
clc; clear; close all;

% 定义数据集
filename = 'treasury.dat';
data = readtable(filename);
disp(['数据集: ' filename]);
disp('数据基本信息:');
disp(head(data, 5));
disp(['数据大小: ', num2str(size(data))]);

% 处理缺失值
data = fillmissing(data, 'previous');

% 处理异常值（Winsorization）
numeric_data = data{:, vartype('numeric')};
Q1 = prctile(numeric_data, 25, 1);
Q3 = prctile(numeric_data, 75, 1);
IQR = Q3 - Q1;
lower_bound = Q1 - 1.5 * IQR;
upper_bound = Q3 + 1.5 * IQR;
numeric_data = max(min(numeric_data, upper_bound), lower_bound);
data{:, vartype('numeric')} = numeric_data;

% 获取特征和目标
nFeatures = size(data, 2) - 1;
targetVar = data.Properties.VariableNames{end};
fprintf('数据集信息: %d个样本, %d个特征, 目标变量: %s\n', ...
        size(data,1), nFeatures, targetVar);

% 分离特征和目标
X = table2array(data(:,1:end-1));
y = table2array(data(:,end));

% 划分数据集 (70% train, 30% test)
n = size(X, 1);
trainRatio = 0.7;
trainSize = floor(n * trainRatio);
X_train = X(1:trainSize, :);
y_train = y(1:trainSize);
X_test = X(trainSize+1:end, :);
y_test = y(trainSize+1:end);
fprintf('训练集: %d个样本, 测试集: %d个样本\n', ...
        size(X_train,1), size(X_test,1));

% 标准化
[X_train_norm, y_train_norm, X_test_norm, y_test_norm, ...
    min_y, max_y, min_y_test, max_y_test, min_x, max_x] = ...
    normalizeData(X_train, y_train, X_test, y_test);
%% KELM评估（使用所有特征，训练集优化）
fprintf('\n=== 基于PSO优化的KELM评估（使用所有特征，训练集优化） ===\n');
config = struct(...
    'kernelType', 'poly', ...
    'swarmSize', 20, ...
    'maxIterations', 30, ...
    'verbose', true);

tic;
[best_params, results] = evaluateWithKELM(...
    X_train_norm, y_train_norm, config, min_y, max_y);
pso_time = toc;
fprintf('PSO-KELM评估耗时: %.2f秒\n', pso_time);

% 使用所有训练数据训练最终KELM模型
X_train_opt = X_train_norm; % 使用所有特征
X_test_opt = X_test_norm; % 使用所有特征

% 训练最终KELM模型
C = max(best_params(1), 1e-6);
kernel_type = config.kernelType;
kernel_params = get_kernel_params(kernel_type, best_params(2), best_params(3)); % degree, theta
H = compute_kernel_matrix(X_train_opt, X_train_opt, kernel_type, kernel_params);
beta = H' * pinv(H * H' + eye(size(H,1))/C) * y_train_norm;

% 测试集预测
H_test = compute_kernel_matrix(X_test_opt, X_train_opt, kernel_type, kernel_params);
y_pred_norm = H_test * beta;
y_pred = y_pred_norm * (max_y - min_y) + min_y;

% 计算测试集指标
test_metrics = calculateExtendedMetrics(y_test, y_pred);

%% 显示结果
fprintf('\n=== 测试集指标 (%s) ===\n', filename);
fprintf('RMSE: %.4f\n', test_metrics.rmse);
fprintf('MAPE: %.4f%%\n', test_metrics.mape);
fprintf('MAE: %.4f\n', test_metrics.mae);
fprintf('SMAPE: %.4f\n', test_metrics.smape);
fprintf('R²: %.4f\n', test_metrics.r_squared);
fprintf('TIC: %.4f\n', test_metrics.tic);
fprintf('Willmott指数: %.4f\n', test_metrics.willmott_index);
fprintf('样本数量: %d\n', n);

%% 函数：KELM评估（使用所有特征，训练集优化）
function [best_params, results] = evaluateWithKELM(...
    X_train_norm, y_train_norm, config, min_y, max_y)
    
    kernel_type = config.kernelType;

    % PSO选项
    pso_options = optimoptions('particleswarm', ...
        'SwarmSize', config.swarmSize, ...
        'MaxIterations', config.maxIterations, ...
        'Display', 'off', ...
        'UseParallel', true); 

    if config.verbose
        fprintf('优化KELM参数，使用所有 %d 个特征...\n', size(X_train_norm, 2));
    end

    % PSO优化 [C, degree, kernel_theta]
    lb = [0.1, 1, 0.01]; % C, degree, theta
    ub = [100, 5, 10];
    fitness_func = @(params) evaluate_kelm(...
        params, X_train_norm, y_train_norm, X_train_norm, y_train_norm, ...
        kernel_type, min_y, max_y);

    try
        [best_params, best_fitness] = particleswarm(fitness_func, 3, lb, ub, pso_options);
    catch e
        fprintf('PSO优化失败: %s\n', e.message);
        best_fitness = Inf;
        best_params = [NaN, NaN, NaN];
    end

    % 使用最优参数评估训练集
    C = best_params(1);
    kernel_params = get_kernel_params(kernel_type, best_params(2), best_params(3));
    H = compute_kernel_matrix(X_train_norm, X_train_norm, kernel_type, kernel_params);
    beta = H' * pinv(H * H' + eye(size(H,1))/C) * y_train_norm;
    H_train = compute_kernel_matrix(X_train_norm, X_train_norm, kernel_type, kernel_params);
    y_pred_norm = H_train * beta;
    y_pred = y_pred_norm * (max_y - min_y) + min_y;

    % 计算训练集指标
    train_metrics = calculateExtendedMetrics(y_train_norm * (max_y - min_y) + min_y, y_pred);

    % 存储结果
    results = struct();
    results.Fitness = best_fitness;
    results.RMSE = train_metrics.rmse;
    results.MAPE = train_metrics.mape;
    results.MAE = train_metrics.mae;
    results.SMAPE = train_metrics.smape;
    results.BestParams = best_params;

    if config.verbose
        fprintf('优化完成 | Fitness: %.4f, RMSE: %.4f, MAPE: %.4f%%, MAE: %.4f, SMAPE: %.4f\n', ...
            best_fitness, results.RMSE, results.MAPE, results.MAE, results.SMAPE);
    end

    fprintf('\n=== PSO-KELM结果（训练集优化） ===\n');
    fprintf('总耗时: %.2f 秒\n', toc);
    fprintf('使用的特征数量: %d\n', size(X_train_norm, 2));
    fprintf('最优适应度值: %.4f\n', results.Fitness);
    fprintf('训练集性能指标:\n');
    fprintf('  RMSE: %.4f\n', results.RMSE);
    fprintf('  MAPE: %.4f%%\n', results.MAPE);
    fprintf('  MAE: %.4f\n', results.MAE);
    fprintf('  SMAPE: %.4f\n', results.SMAPE);
    fprintf('最优超参数: C=%.4f, degree=%.0f, kernel_theta=%.4f\n', ...
        best_params(1), best_params(2), best_params(3));
end

%% 函数：KELM评估
function fitness = evaluate_kelm(params, X_train, y_train, X_eval, y_eval, ...
    kernel_type, min_y, max_y)
    try
        C = params(1);
        kernel_params = get_kernel_params(kernel_type, params(2), params(3));
        H = compute_kernel_matrix(X_train, X_train, kernel_type, kernel_params);
        beta = H' * pinv(H * H' + eye(size(H,1))/C) * y_train;
        H_eval = compute_kernel_matrix(X_eval, X_train, kernel_type, kernel_params);
        y_pred_norm = H_eval * beta;
        y_pred = y_pred_norm * (max_y - min_y) + min_y;

        rmse = sqrt(mean((y_eval * (max_y - min_y) + min_y - y_pred).^2));
        fitness = rmse;

        if ~isfinite(fitness) || isnan(fitness)
            fitness = Inf;
        end
    catch e
        fprintf('Error in evaluate_kelm: %s\n', e.message);
        fitness = Inf;
    end
end

%% 函数：数据标准化
function [X_train_norm, y_train_norm, X_test_norm, y_test_norm, ...
    min_y, max_y, min_y_test, max_y_test, min_x, max_x] = ...
    normalizeData(X_train, y_train, X_test, y_test)
    if isempty(X_train) || isempty(y_train) || isempty(X_test) || isempty(y_test)
        error('Input data is empty');
    end

    [X_train_norm, ps_x] = mapminmax(X_train', 0, 1);
    X_train_norm = X_train_norm';
    X_test_norm = mapminmax('apply', X_test', ps_x)';
    min_x = ps_x.xmin;
    max_x = ps_x.xmax;
    max_x(max_x == min_x) = min_x(max_x == min_x) + 1;

    [y_train_norm, ps_y] = mapminmax(y_train', 0, 1);
    y_train_norm = y_train_norm';
    y_test_norm = mapminmax('apply', y_test', ps_y)';
    min_y = ps_y.xmin;
    max_y = ps_y.xmax;
    if max_y == min_y
        warning('y_train is constant, setting max_y - min_y = 1');
        max_y = min_y + 1;
    end

    min_y_test = min(y_test);
    max_y_test = max(y_test);
    if max_y_test == min_y_test
        warning('y_test is constant, setting max_y_test - min_y_test = 1');
        max_y_test = min_y_test + 1;
    end
end

%% 函数：性能指标
function metrics = calculateExtendedMetrics(y_true, y_pred)  
    valid_idx = ~isnan(y_true) & ~isnan(y_pred);
    y_true = y_true(valid_idx);
    y_pred = y_pred(valid_idx);
    
    if isempty(y_true)
        error('有效数据点不足');
    end
    
    n = length(y_true);
    res = y_true - y_pred;
    epsilon = 1e-6;
    
    metrics.rmse = sqrt(mean(res.^2));
    metrics.mape = 100 * mean(abs(res ./ (y_true + epsilon)));
    metrics.mae = mean(abs(res));
    metrics.smape = mean(2 * abs(y_pred - y_true) ./ (abs(y_pred) + abs(y_true) + epsilon)) * 100;
    
    ss_res = sum(res.^2);
    ss_tot = sum((y_true - mean(y_true)).^2);
    if ss_tot == 0
        if ss_res == 0
            metrics.r_squared = 1;
        else
            metrics.r_squared = NaN;
        end
    else
        metrics.r_squared = 1 - (ss_res / ss_tot);
    end
    
    numerator = sqrt(mean(res.^2));
    denominator = sqrt(mean(y_true.^2)) + sqrt(mean(y_pred.^2));
    if denominator == 0
        metrics.tic = 0;
    else
        metrics.tic = numerator / denominator;
    end
    
    numerator_d = sum(res.^2);
    denominator_d = sum((abs(y_pred - mean(y_true)) + abs(y_true - mean(y_true))).^2);
    if denominator_d == 0
        metrics.willmott_index = 1;
    else
        metrics.willmott_index = 1 - (numerator_d / denominator_d);
    end
end

%% 辅助函数

function K = compute_kernel_matrix(X1, X2, kernel_type, params)
    switch kernel_type
        case 'poly'
            K = (X1 * X2' + params.theta).^params.degree;
        case 'rbf'
            sq_dists = pdist2(X1, X2, 'squaredeuclidean');
            K = exp(-sq_dists / (2 * params.sigma^2));
        case 'linear'
            K = X1 * X2';
        otherwise
            error('未知的核函数类型');
    end
end

function params = get_kernel_params(kernel_type, degree, theta)
    switch kernel_type
        case 'poly'
            params = struct('degree', round(degree), 'theta', theta);
        case 'rbf'
            params = struct('sigma', degree);
        case 'linear'
            params = struct();
        otherwise
            error('未知的核函数类型');
    end
end