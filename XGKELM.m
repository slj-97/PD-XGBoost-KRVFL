function XGrunFeatureSelectionPipeline()
    clc; clear; close all;
   
    %% 1. 数据加载与预处理
    % 读取数据（请替换为您的实际数据文件）
    data = readtable('treasury.dat');
    disp('数据基本信息:');
    disp(head(data, 5));
    disp(['数据大小: ', num2str(size(data))]);

    % 处理缺失值
    data = fillmissing(data, 'previous'); % 使用前向填充

    % 处理异常值（使用Winsorization方法）
    numeric_data = data{:, vartype('numeric')};
    
    % 并行处理每列的异常值
    Q1 = prctile(numeric_data, 25, 1);
    Q3 = prctile(numeric_data, 75, 1);
    IQR = Q3 - Q1;
    lower_bound = Q1 - 1.5 * IQR;
    upper_bound = Q3 + 1.5 * IQR;
    numeric_data = max(min(numeric_data, upper_bound), lower_bound);
    data{:, vartype('numeric')} = numeric_data;

    % 获取特征和目标变量
    nFeatures = size(data, 2) - 1;
    targetVar = data.Properties.VariableNames{end};
    fprintf('数据集信息: %d个样本, %d个特征, 目标变量: %s\n', ...
            size(data,1), nFeatures, targetVar);
    
    [trainData, testData] = splitDataset(data, 0.7);
    fprintf('训练集: %d个样本, 测试集: %d个样本\n', ...
            size(trainData,1), size(testData,1));
    
    % 转换为数值矩阵并标准化
    [X_train, y_train, X_test, y_test] = prepareData(trainData, testData);

    %% 2. XGBoost-KELM评估
    fprintf('\n=== 基于XGBoost-KELM的模型评估 ===\n');

    % XGBoost-KELM配置
    config = struct(...
        'kernelType',      'poly', ...
        'numExperiments',  1, ...             % 实验次数，增加实验次数以提高稳定性
        'xi',             0.05, ...           % 适应度平衡参数
        'swarmSize',       20, ...            % PSO粒子数
        'maxIterations',   30, ...            % PSO最大迭代
        'verbose',         true);            % 显示进度
    
    % 使用所有特征（按原始顺序）
    selectedFeatures = 1:nFeatures;
    
    % 执行XGBoost-KELM评估
    tic;
    [optimalFeatures, results] = evaluateWithXGBoostKELM(...
        X_train, y_train, X_test, y_test, selectedFeatures, config, size(X_train, 2));
    elapsed_time = toc;
    fprintf('XGBoost-KELM评估耗时: %.2f秒\n', elapsed_time);
end

%% ========== XGBoost-KELM评估函数 ==========
function [optimalFeatures, results] = evaluateWithXGBoostKELM(...
    X_train, y_train, X_test, y_test, selectedFeatures, config, m_original)

    % 数据标准化
    [X_train_norm, y_train_norm, X_test_norm, y_test_norm, min_y, max_y] = ...
        normalizeData(X_train, y_train, X_test, y_test);

    % 定义权重方法（仅保留TBW）
    weight_methods = struct('TBW', @(E, params) threshold_weights(E, params.theta));
    method_names = {'TBW'};

    % 参数与初始设置
    totalFeatures = length(selectedFeatures);
    kernel_type = config.kernelType;
    xi = config.xi;
    minFeatures = totalFeatures; % 仅评估完整特征集
    num_steps = 1; % 仅一次评估

    % 初始化结果存储，包含所有七个指标
    all_metrics = struct('rmse', zeros(1, num_steps), ...
                         'mape', zeros(1, num_steps), ...
                         'mae', zeros(1, num_steps), ...
                         'smape', zeros(1, num_steps), ...
                         'r_squared', zeros(1, num_steps), ...
                         'tic', zeros(1, num_steps), ...
                         'willmott_index', zeros(1, num_steps));

    fprintf('特征数: %d\n', totalFeatures);
    tic;

    % 存储最佳超参数
    best_params_all = zeros(num_steps, 5);

    % 仅评估完整特征集
    currentFeat = selectedFeatures;
    X_train_current = X_train_norm(:, currentFeat);
    X_test_current = X_test_norm(:, currentFeat);

    if config.verbose
        fprintf('评估 %d 个特征...\n', totalFeatures);
    end

    % PSO 参数优化
    lb = [10, 0.05, 0.1, 1, 0.01];
    ub = [200, 0.5, 100, 5, 10];
    nvars = 5;
    pso_options = optimoptions('particleswarm', ...
        'SwarmSize', config.swarmSize, ...
        'MaxIterations', config.maxIterations, ...
        'Display', 'off', ...
        'UseParallel', true); % 避免并行警告

    fitness_func = @(params) evaluate_xgb_kelm_with_weight_selection(...
        params, X_train_current, y_train_norm, X_test_current, y_test_norm, ...
        kernel_type, method_names, weight_methods, xi, m_original);

    best_params = particleswarm(fitness_func, nvars, lb, ub, pso_options);

    % 存储超参数
    idx = 1;
    best_params_all(idx, :) = best_params;

    % 解码最优参数
    K = round(best_params(1));
    eta = best_params(2);
    C = best_params(3);
    kernel_params = get_kernel_params(kernel_type, best_params(4), best_params(5));
    best_method = 'TBW';

    % 训练 XGBoost
    mdl_xgb = fitrensemble(X_train_current, y_train_norm, ...
        'Method', 'LSBoost', 'NumLearningCycles', K, 'LearnRate', eta);
    y_pred_xgb = predict(mdl_xgb, X_train_current);

    % 计算权重
    E = (y_train_norm - y_pred_xgb).^2;
    weight_params = get_weight_params(best_method, E);
    alpha = compute_weights(E, best_method, weight_params, weight_methods);
    alpha = normalize_weights(alpha);

    % 计算 KELM 输出
    H = compute_kernel_matrix(X_train_current, X_train_current, kernel_type, kernel_params);
    D = diag(1 ./ (C * alpha));
    beta = H' * pinv(H * H' + D) * y_train_norm;

    H_test = compute_kernel_matrix(X_test_current, X_train_current, kernel_type, kernel_params);
    y_pred_norm = H_test * beta;
    y_pred = y_pred_norm * (max_y - min_y) + min_y;

    % 评估指标
    metrics = calculateExtendedMetrics(y_test, y_pred);

    % 存储结果
    all_metrics.rmse(idx) = metrics.rmse;
    all_metrics.mape(idx) = metrics.mape;
    all_metrics.mae(idx) = metrics.mae;
    all_metrics.smape(idx) = metrics.smape;
    all_metrics.r_squared(idx) = metrics.r_squared;
    all_metrics.tic(idx) = metrics.tic;
    all_metrics.willmott_index(idx) = metrics.willmott_index;

    if config.verbose
        fprintf('完成 | RMSE: %.4f\n', metrics.rmse);
    end

    elapsed_time = toc;

    % 计算适应度
    numFeatVec = totalFeatures';
    fitness = all_metrics.rmse' + xi * (numFeatVec / m_original);

    % 结果表格
    results = table();
    results.NumFeatures = numFeatVec;
    results.RMSE = all_metrics.rmse';
    results.MAPE = all_metrics.mape';
    results.MAE = all_metrics.mae';
    results.SMAPE = all_metrics.smape';
    results.R_squared = all_metrics.r_squared';
    results.TIC = all_metrics.tic';
    results.Willmott_index = all_metrics.willmott_index';
    results.Fitness = fitness;

    % 最优特征子集（完整特征集）
    optimalIdx = 1;
    optimalFeatures = selectedFeatures;

    % 输出所有指标
    fprintf('\n=== 最优特征子集结果 ===\n');
    fprintf('总耗时: %.2f 秒\n', elapsed_time);
    fprintf('特征数量: %d\n', results.NumFeatures(optimalIdx));
    fprintf('适应度值: %.4f\n', results.Fitness(optimalIdx));
    fprintf('测试集性能指标:\n');
    fprintf('  RMSE: %.4f\n', results.RMSE(optimalIdx));
    fprintf('  MAPE: %.4f%%\n', results.MAPE(optimalIdx));
    fprintf('  MAE: %.4f\n', results.MAE(optimalIdx));
    fprintf('  SMAPE: %.4f%%\n', results.SMAPE(optimalIdx));
    fprintf('  R²: %.4f\n', results.R_squared(optimalIdx));
    fprintf('  TIC: %.4f\n', results.TIC(optimalIdx));
    fprintf('  Willmott指数: %.4f\n', results.Willmott_index(optimalIdx));
    fprintf('特征索引: %s\n', mat2str(optimalFeatures));
    fprintf('最优超参数: K=%.0f, eta=%.4f, C=%.4f, degree=%.0f, kernel_theta=%.4f\n', ...
        best_params_all(optimalIdx, 1), best_params_all(optimalIdx, 2), ...
        best_params_all(optimalIdx, 3), best_params_all(optimalIdx, 4), ...
        best_params_all(optimalIdx, 5));
end

%% ========== XGBoost-KELM评估函数 ==========
function fitness = evaluate_xgb_kelm_with_weight_selection(params, X_train, y_train, X_test, y_test, kernel_type, method_names, weight_methods, xi, m_original)
    try
        K = round(params(1));
        eta = params(2);
        C = params(3);
        kernel_params = get_kernel_params(kernel_type, params(4), params(5));
        method = 'TBW';
        
        mdl_xgb = fitrensemble(X_train, y_train, 'Method', 'LSBoost', ...
            'NumLearningCycles', K, 'LearnRate', eta);
        y_pred_xgb = predict(mdl_xgb, X_train);
        E = (y_train - y_pred_xgb).^2;
        weight_params = get_weight_params(method, E);
        alpha = compute_weights(E, method, weight_params, weight_methods);
        alpha = normalize_weights(alpha);
        H = compute_kernel_matrix(X_train, X_train, kernel_type, kernel_params);
        D = diag(1 ./ (C * alpha));
        beta = H' * pinv(H * H' + D) * y_train;
        % 使用训练集计算RMSE
        H_train = compute_kernel_matrix(X_train, X_train, kernel_type, kernel_params);
        y_pred_norm = H_train * beta;
        y_pred = y_pred_norm * (max_y - min_y) + min_y;
        rmse = sqrt(mean((y_train - y_pred).^2));
        num_features = size(X_train, 2);
        fitness = rmse + xi * (num_features / m_original);
        
        if ~isfinite(fitness) || isnan(fitness)
            fitness = Inf;
        end
    catch
        fitness = Inf;
    end
end

%% ========== 辅助函数 ==========
function [X_train_norm, y_train_norm, X_test_norm, y_test_norm, min_y, max_y] = ...
    normalizeData(X_train, y_train, X_test, y_test)
    min_X = min(X_train, [], 1);
    max_X = max(X_train, [], 1);
    range_X = max_X - min_X;
    range_X = max(range_X, 1e-6);
    X_train_norm = (X_train - min_X) ./ range_X;
    X_test_norm = (X_test - min_X) ./ range_X;
    min_y = min(y_train);
    max_y = max(y_train);
    range_y = max_y - min_y;
    if range_y == 0
        range_y = eps;
    end
    y_train_norm = (y_train - min_y) / range_y;
    y_test_norm = (y_test - min_y) / range_y;
end

function metrics = calculateExtendedMetrics(y_true, y_pred)
    if length(y_true) ~= length(y_pred)
        error('真实值和预测值必须具有相同的长度');
    end
    
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

function val = getParam(params, field, default)
    if isfield(params, field)
        val = params.(field);
    else
        val = default;
    end
end

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

function [trainData, testData] = splitDataset(data, trainRatio)
    % 划分训练集和测试集
    n = size(data, 1);
    trainSize = floor(n * trainRatio);
    rng(42); % 固定随机种子保证可重复性
    idx = randperm(n);
    
    trainData = data(idx(1:trainSize), :);
    testData = data(idx(trainSize+1:end), :);
end

function [X_train, y_train, X_test, y_test] = prepareData(trainData, testData)
    % 转换为数值矩阵并分离特征/目标
    trainMatrix = table2array(trainData);
    testMatrix = table2array(testData);
    
    X_train = trainMatrix(:, 1:end-1);
    y_train = trainMatrix(:, end);
    X_test = testMatrix(:, 1:end-1);
    y_test = testMatrix(:, end);
end

function params = get_kernel_params(kernel_type, param1, param2)
    switch kernel_type
        case 'poly'
            params = struct('degree', round(param1), 'theta', param2);
        case 'rbf'
            params = struct('sigma', param1);
        case 'linear'
            params = struct();
        otherwise
            error('未知的核函数类型');
    end
end

function alpha = compute_weights(E, method, params, weight_methods)
    if isfield(weight_methods, method)
        alpha = weight_methods.(method)(E, params);
    else
        error('未知的权重方法');
    end
end

function params = get_weight_params(method, E)
    switch method
        case 'TBW'
            params.theta = prctile(E, 50);
        otherwise
            params = struct();
    end
end

function w = threshold_weights(E, theta)
    w = ones(size(E));
    mask = E > theta;
    w(mask) = theta ./ E(mask);
end

function w_norm = normalize_weights(w)
    w_min = min(w);
    w_max = max(w);
    if w_min == w_max
        w_norm = ones(size(w));
    else
        w_norm = 0.1 + 0.9 * (w - w_min) / (w_max - w_min);
    end
end