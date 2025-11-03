%% 主脚本：运行模糊特征选择和PSO优化的KELM
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

%% 特征选择
fprintf('\n=== 开始特征选择流程 ===\n');
params = struct('lambda', 0.2, 'alpha', 0.1, 'epsilon', 1e-6);
tic;
[selectedFeatures, scores] = FuzzyFeatureSelection(X_train_norm, y_train_norm, params);
fuzzy_time = toc;
fprintf('模糊特征选择耗时: %.2f秒\n', fuzzy_time);

%% KELM评估
fprintf('\n=== 基于PSO优化的KELM特征评估（训练集优化，最少保留10个特征） ===\n');
config = struct(...
    'kernelType', 'poly', ...
    'xi', 0.01, ...
    'swarmSize', 20, ...
    'maxIterations', 30, ...
    'verbose', true);

[optimalFeatures, results, best_params] = evaluateWithKELM(...
    X_train_norm, y_train_norm, selectedFeatures, config, nFeatures, min_y, max_y);

% 使用最优特征子集训练最终KELM模型
X_train_opt = X_train_norm(:, optimalFeatures);
X_test_opt = X_test_norm(:, optimalFeatures);

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

%% 函数：KELM评估
function [optimalFeatures, results, best_params] = evaluateWithKELM(...
    X_train_norm, y_train_norm, selectedFeatures, config, m_original, min_y, max_y)
    
    totalFeatures = length(selectedFeatures);
    kernel_type = config.kernelType;
    xi = config.xi;
    minFeatures = min(12, totalFeatures); % 至少保留10个特征，但不超过总特征数

    % PSO选项
    pso_options = optimoptions('particleswarm', ...
        'SwarmSize', config.swarmSize, ...
        'MaxIterations', config.maxIterations, ...
        'Display', 'off', ...
        'UseParallel', true); 
    
    % 存储结果
    results = table();
    results.NumFeatures = (totalFeatures:-1:minFeatures)';
    results.Fitness = zeros(totalFeatures-minFeatures+1, 1);
    results.RMSE = zeros(totalFeatures-minFeatures+1, 1);
    results.MAPE = zeros(totalFeatures-minFeatures+1, 1);
    results.MAE = zeros(totalFeatures-minFeatures+1, 1);
    results.SMAPE = zeros(totalFeatures-minFeatures+1, 1);
    results.R_squared = zeros(totalFeatures-minFeatures+1, 1);
    results.TIC = zeros(totalFeatures-minFeatures+1, 1);
    results.Willmott_index = zeros(totalFeatures-minFeatures+1, 1);
    results.BestParams = cell(totalFeatures-minFeatures+1, 1);

    tic;
    for numFeat = totalFeatures:-1:minFeatures
        currentFeat = selectedFeatures(1:numFeat);
        X_train_current = X_train_norm(:, currentFeat);

        if config.verbose
            fprintf('评估 %2d/%2d 特征...\n', numFeat, totalFeatures);
        end

        % PSO优化 [C, degree, kernel_theta]
        lb = [0.1, 1, 0.01]; % C, degree, theta
        ub = [100, 5, 10];
        fitness_func = @(params) evaluate_kelm(...
            params, X_train_current, y_train_norm, X_train_current, y_train_norm, ...
            kernel_type, xi, m_original, min_y, max_y);

        try
            [best_params, best_fitness] = particleswarm(fitness_func, 3, lb, ub, pso_options);
        catch e
            fprintf('PSO优化失败: %s\n', e.message);
            best_fitness = Inf;
            best_params = [NaN, NaN, NaN];
        end

        % 训练集评估
        C = best_params(1);
        kernel_params = get_kernel_params(kernel_type, best_params(2), best_params(3));
        H = compute_kernel_matrix(X_train_current, X_train_current, kernel_type, kernel_params);
        beta = H' * pinv(H * H' + eye(size(H,1))/C) * y_train_norm;
        H_train = compute_kernel_matrix(X_train_current, X_train_current, kernel_type, kernel_params);
        y_pred_norm = H_train * beta;
        y_pred = y_pred_norm * (max_y - min_y) + min_y;

        % 计算指标
        train_metrics = calculateExtendedMetrics(y_train_norm * (max_y - min_y) + min_y, y_pred);

        idx = totalFeatures - numFeat + 1;
        results.Fitness(idx) = best_fitness;
        results.RMSE(idx) = train_metrics.rmse;
        results.MAPE(idx) = train_metrics.mape;
        results.MAE(idx) = train_metrics.mae;
        results.SMAPE(idx) = train_metrics.smape;
        results.R_squared(idx) = train_metrics.r_squared;
        results.TIC(idx) = train_metrics.tic;
        results.Willmott_index(idx) = train_metrics.willmott_index;
        results.BestParams{idx} = best_params;

        if config.verbose
            fprintf('完成 | Fitness: %.4f, RMSE: %.4f, MAPE: %.4f%%, MAE: %.4f, SMAPE: %.4f\n', ...
                best_fitness, train_metrics.rmse, train_metrics.mape, train_metrics.mae, train_metrics.smape);
        end
    end
    elapsed_time = toc;

    % 选择最优特征
    [~, optimalIdx] = min(results.Fitness);
    optimalFeatures = selectedFeatures(1:results.NumFeatures(optimalIdx));
    best_params = results.BestParams{optimalIdx};

    fprintf('\n=== 最优特征子集结果 ===\n');
    fprintf('总耗时: %.2f 秒\n', elapsed_time);
    fprintf('最佳特征数量: %d (原特征数: %d)\n', results.NumFeatures(optimalIdx), totalFeatures);
    fprintf('最优适应度值: %.4f\n', results.Fitness(optimalIdx));
    fprintf('训练集性能指标:\n');
    fprintf('  RMSE: %.4f\n', results.RMSE(optimalIdx));
    fprintf('  MAPE: %.4f%%\n', results.MAPE(optimalIdx));
    fprintf('  MAE: %.4f\n', results.MAE(optimalIdx));
    fprintf('  SMAPE: %.4f\n', results.SMAPE(optimalIdx));
    fprintf('  R²: %.4f\n', results.R_squared(optimalIdx));
    fprintf('  TIC: %.4f\n', results.TIC(optimalIdx));
    fprintf('  Willmott指数: %.4f\n', results.Willmott_index(optimalIdx));
    fprintf('最优特征索引: %s\n', mat2str(optimalFeatures));
    fprintf('最优超参数: C=%.4f, degree=%.0f, kernel_theta=%.4f\n', ...
        best_params(1), best_params(2), best_params(3));
end

%% 函数：KELM评估
function fitness = evaluate_kelm(params, X_train, y_train, X_eval, y_eval, ...
    kernel_type, xi, m_original, min_y, max_y)
    try
        C = params(1);
        kernel_params = get_kernel_params(kernel_type, params(2), params(3));
        H = compute_kernel_matrix(X_train, X_train, kernel_type, kernel_params);
        beta = H' * pinv(H * H' + eye(size(H,1))/C) * y_train;
        H_eval = compute_kernel_matrix(X_eval, X_train, kernel_type, kernel_params);
        y_pred_norm = H_eval * beta;
        y_pred = y_pred_norm * (max_y - min_y) + min_y;

        rmse = sqrt(mean((y_eval * (max_y - min_y) + min_y - y_pred).^2));
        num_features = size(X_train, 2);
        fitness = rmse + xi * (num_features / m_original);

        if ~isfinite(fitness) || isnan(fitness)
            fitness = Inf;
        end
    catch e
        fprintf('Error in evaluate_kelm: %s\n', e.message);
        fitness = Inf;
    end
end

%% 函数：模糊特征选择
function [selectedFeatures, scores] = FuzzyFeatureSelection(X, y, params)
    lambda = getParam(params, 'lambda', 0.1);
    alpha = getParam(params, 'alpha', 0.01);
    epsilon = getParam(params, 'epsilon', 1e-10);
    [n, m] = size(X);
    
    if n == 0 || m == 0
        error('Input data X is empty');
    end
    
    K = floor(sqrt(n));
    fprintf('使用固定邻域大小 K=%d\n', K);
    
    Pi = zeros(n, m+1);
    D = pdist2(X, X, 'euclidean');
    for j = 1:m+1
        if j <= m
            data = X(:,j);
        else
            data = y;
        end
        mu = mean(data);
        sigma = std(data) + epsilon;
        for i = 1:n
            [~, idx] = sort(D(i,:));
            neighbors = data(idx(2:K+1));
            Pi(i,j) = mean(exp(-(neighbors - mu).^2 / (2*sigma^2)));
        end
    end
    d = Pi(:,end);

    E = cell(1, m+1);
    parfor j = 1:m+1
        data_j = Pi(:,j);
        E_temp = exp(-squareform(pdist(data_j)));
        E{j} = E_temp;
    end
    E_target = E{end};
    
    Ims = zeros(1, m);
    H = zeros(1, m);
    H_d = 0;
    for i = 1:n
        card_d = sum(E_target(i,:));
        H_d = H_d + log(n / card_d + epsilon);
    end
    H_d = H_d / n;
    
    for j = 1:m
        E_j = E{j};
        H_j = 0;
        for i = 1:n
            card = sum(E_j(i,:));
            H_j = H_j + log(n / card + epsilon);
        end
        H_j = H_j / n;
        H(j) = H_j;
        
        H_joint = 0;
        for i = 1:n
            card_joint = sum(min(E_j(i,:), E_target(i,:)));
            H_joint = H_joint + log(n / card_joint + epsilon);
        end
        H_joint = H_joint / n;
        
        I_jd = H_j + H_d - H_joint;
        Ims(j) = I_jd - lambda * H_j;
    end
    
    S1 = [];
    S2 = 1:m;
    finalScores = zeros(1,m);
    P = Pi(:,1:m) ./ (sum(Pi(:,1:m),2) + epsilon);
    
    [~, first] = max(Ims);
    S1 = [S1 first];
    S2(S2==first) = [];
    finalScores(first) = Ims(first);
    
    while ~isempty(S2)
        redundancy = zeros(1,length(S2));
        for k = 1:length(S2)
            j = S2(k);
            p_j = P(:,j);
            red_k = 0;
            for s = S1
                p_s = P(:,s);
                m_p = (p_j + p_s)/2;
                JS = 0.5*(sum(p_j.*log((p_j+epsilon)./(m_p+epsilon))) + ...
                          sum(p_s.*log((p_s+epsilon)./(m_p+epsilon))));
                red_k = red_k + exp(-JS);
            end
            redundancy(k) = red_k/length(S1);
        end
        
        scores = Ims(S2) - alpha * redundancy;
        [maxScore, idx] = max(scores);
        selected = S2(idx);
        S1 = [S1 selected];
        S2(S2==selected) = [];
        finalScores(selected) = maxScore;
    end
    
    [scores, order] = sort(finalScores, 'descend');
    selectedFeatures = order;
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

%% 辅助函数
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