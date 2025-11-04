function XGrunFeatureSelectionPipeline()
    % 主函数：完整特征选择流程（集成XGBoost-KRVFL评估）
    % 优化版本：增强并行计算性能
    clc; clear; close all;
   
    
    %% 1. 数据加载与预处理
    % 读取数据（请替换为您的实际数据文件）
    data = readtable('Treasury.dat');
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


    %% 3. 特征选择 (仅在训练集进行)
    fprintf('\n=== 开始特征选择流程 ===\n');
    
    % 参数设置
    params = struct(...
        'lambda',   0.2,    ... % 模糊熵权重
        'alpha',    0.1,   ... % 冗余惩罚系数
        'epsilon',  1e-6   ... % 数值稳定项
    );
    
    % 执行特征选择
    tic;
    [selectedFeatures, scores] = FuzzyFeatureSelection(X_train, y_train, params);
    fuzzy_time = toc;
    fprintf('模糊特征选择耗时: %.2f秒\n', fuzzy_time);
    
    % 显示结果
    % displayResults(selectedFeatures, scores, trainData.Properties.VariableNames(1:end-1));
    
    %% 4. XGBoost-KRVFL特征评估（核心修改部分）
    fprintf('\n=== 基于XGBoost-KRVFL的特征评估 ===\n');

    % XGBoost-KRVFL配置
    config = struct(...
        'kernelType',      'poly', ...
        'numExperiments',  1, ...             % 实验次数，增加实验次数以提高稳定性
        'xi',             0.05, ...           % 适应度平衡参数
        'swarmSize',       20, ...            % PSO粒子数
        'maxIterations',   30, ...            % PSO最大迭代
        'verbose',       true    );         % 显示进度
                     % 使用waitbar显示进度;
    
    % 执行XGBoost-KRVFL特征评估
    [optimalFeatures, results] = evaluateWithXGBoostKRVFL(...
        X_train, y_train, X_test, y_test, selectedFeatures, config,size(X_train, 2));
    
    % 显示最终结果
    % displayXGBoostKRVFLResults(optimalFeatures, results, config);
end
%% ========== XGBoost_KRVFL特征评估函数 ==========
function [optimalFeatures, results] = evaluateWithXGBoostKRVFL(...
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
    minFeatures = min(15, totalFeatures); 
    num_steps = totalFeatures - minFeatures + 1;

    % 初始化结果存储，包含所有七个指标
    all_metrics = struct('rmse', zeros(1, num_steps), ...
                         'mape', zeros(1, num_steps), ...
                         'mae', zeros(1, num_steps), ...
                         'smape', zeros(1, num_steps), ...
                         'r_squared', zeros(1, num_steps), ...
                         'tic', zeros(1, num_steps), ...
                         'willmott_index', zeros(1, num_steps));

    fprintf('初始特征数: %d\n', totalFeatures);
    tic;

    % 存储最佳超参数（用于最终输出）
    best_params_all = zeros(num_steps, 5); % 保存每轮特征数的超参数

    % 特征数量递减搜索
    for numFeat = totalFeatures:-1:minFeatures
        currentFeat = selectedFeatures(1:numFeat);
        X_train_current = X_train_norm(:, currentFeat);
        X_test_current = X_test_norm(:, currentFeat);

        if config.verbose
            fprintf('评估 %2d/%2d 特征...\n', numFeat, totalFeatures);
        end

        % PSO 参数优化
        lb = [10, 0.05, 0.1, 1, 0.01];
        ub = [200, 0.5, 100, 5, 10];
        nvars = 5;
        pso_options = optimoptions('particleswarm', ...
            'SwarmSize', config.swarmSize, ...
            'MaxIterations', config.maxIterations, ...
            'Display', 'off', ...
            'UseParallel', false); % 避免并行警告

        fitness_func = @(params) evaluate_xgb_KRVFL_with_weight_selection(...
            params, X_train_current, y_train_norm, X_test_current, y_test_norm, ...
            kernel_type, method_names, weight_methods, xi, m_original);

        best_params = particleswarm(fitness_func, nvars, lb, ub, pso_options);

        % 存储超参数
        idx = totalFeatures - numFeat + 1;
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

        % 计算 KRVFL 输出
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
    end

    elapsed_time = toc;

    % 计算适应度
    numFeatVec = (totalFeatures:-1:minFeatures)';
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

    % 最优特征子集
    [~, optimalIdx] = min(fitness);
    optimalFeatures = selectedFeatures(1:results.NumFeatures(optimalIdx));

    % 输出所有指标
    fprintf('\n=== 最优特征子集结果 ===\n');
    fprintf('总耗时: %.2f 秒\n', elapsed_time);
    fprintf('最佳特征数量: %d (原特征数: %d)\n', results.NumFeatures(optimalIdx), totalFeatures);
    fprintf('最优适应度值: %.4f\n', results.Fitness(optimalIdx));
    fprintf('测试集性能指标:\n');
    fprintf('  RMSE: %.4f\n', results.RMSE(optimalIdx));
    fprintf('  MAPE: %.4f%%\n', results.MAPE(optimalIdx));
    fprintf('  MAE: %.4f\n', results.MAE(optimalIdx));
    fprintf('  SMAPE: %.4f%%\n', results.SMAPE(optimalIdx));
    fprintf('  R²: %.4f\n', results.R_squared(optimalIdx));
    fprintf('  TIC: %.4f\n', results.TIC(optimalIdx));
    fprintf('  Willmott指数: %.4f\n', results.Willmott_index(optimalIdx));
    fprintf('最优特征索引: %s\n', mat2str(optimalFeatures));
    % fprintf('最优超参数: K=%.0f, eta=%.4f, C=%.4f, degree=%.0f, kernel_theta=%.4f\n', ...
    %     best_params_all(optimalIdx, 1), best_params_all(optimalIdx, 2), ...
    %     best_params_all(optimalIdx, 3), best_params_all(optimalIdx, 4), ...
    %     best_params_all(optimalIdx, 5));
end

   



%% XGBoost-KRVFL评估函数
function fitness = evaluate_xgb_KRVFL_with_weight_selection(params, X_train, y_train, X_test, y_test, kernel_type, method_names, weight_methods, xi, m_original)
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
        % 使用训练集计算RMSE（修复潜在问题1）
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
%% 模糊特征选择函数（优化并行计算）
function [selectedFeatures, scores] = FuzzyFeatureSelection(X, y, params)
    % 参数检查
    if nargin < 3, params = struct(); end
    lambda = getParam(params, 'lambda', 0.1);
    alpha = getParam(params, 'alpha', 0.01);
    epsilon = getParam(params, 'epsilon', 1e-10);
    use_parallel = getParam(params, 'parallel', false);
    [n, m] = size(X);
    
    %% 1. 计算固定邻域（sqrt(n)）
    K = floor(sqrt(n));
    fprintf('使用固定邻域大小 K=%d\n', K);
    
    %% 2. 构建模糊可能性分布
    fprintf('计算模糊可能性分布...\n');
    Pi = zeros(n, m+1); % 最后一列是目标变量
    
    % 计算所有特征和目标变量的PD
    for j = 1:m+1
        if j <= m
            data = X(:,j);
        else
            data = y;
        end
        mu = mean(data);
        sigma = std(data) + epsilon;
        
        % 预先计算距离矩阵（优化性能）
        if j == 1
            D = pdist2(X, X, 'euclidean');
        end
        
        for i = 1:n
            [~, idx] = sort(D(i,:));
            neighbors = data(idx(2:K+1)); % 去掉自身
            Pi(i,j) = mean(exp(-(neighbors - mu).^2 / (2*sigma^2))); 
        end
    end
    d = Pi(:,end); % 目标变量的PD

    %% 3. 计算特征重要性评分
    fprintf('计算特征评分...\n');
   
parfor j = 1:m+1
    data_j = Pi(:,j);
    E_temp = exp(-squareform(pdist(data_j)));
    E{j} = E_temp;
end
    E_target = E{end}; % 目标变量的模糊关系
 %% 3. 计算特征重要性评分
    fprintf('计算特征重要性...\n');
    Ims = zeros(1, m);
    H = zeros(1, m);   % 存储各特征熵值
    
    for j = 1:m
        E_j = E{j};
        
        % 计算模糊熵H(π_j) (式\ref{eq:fuzzy_entropy_def})
        H_j = 0;
        for i = 1:n
            card = sum(E_j(i,:)); % |[x_i]| (式\ref{eq:cardinal_value})
            H_j = H_j + log(n / card + epsilon);
        end
        H_j = H_j / n;
        H(j) = H_j;
        
        % 计算联合熵H(π_j,d)
        H_joint = 0;
        for i = 1:n
            card_joint = sum(min(E_j(i,:), E_target(i,:))); % |[x_i]_j ∩ [x_i]_d|
            H_joint = H_joint + log(n / card_joint);
        end
        H_joint = H_joint / n;
        
        % 计算目标熵H(d)
        H_d = 0;
        for i = 1:n
            card_d = sum(E_target(i,:));
            H_d = H_d + log(n / card_d);
        end
        H_d = H_d / n;
        
        % 计算互信息I(π_j;d) (式\ref{eq:fuzzy_mutual_information_entropy})
        I_jd = H_j + H_d - H_joint;
        
        % 综合评分 (式\ref{ims})
        Ims(j) = I_jd - lambda * H_j;
    end
    
    %% 4. 迭代选择特征（带冗余惩罚）
    fprintf('迭代选择特征...\n');
    S1 = [];       % 已选特征
    S2 = 1:m;      % 剩余特征
    finalScores = zeros(1,m);
    
    % 归一化PD (用于JS散度计算)
    P = Pi(:,1:m) ./ (sum(Pi(:,1:m)) + epsilon);
    
    % 选择第一个特征
    [~, first] = max(Ims);
    S1 = [S1 first]; 
    S2(S2==first) = [];
    finalScores(first) = Ims(first);
    
    % 迭代选择后续特征
    while ~isempty(S2)
        redundancy = zeros(1,length(S2));
        
        % 并行计算冗余度（如果特征和已选特征足够多）
        if use_parallel && length(S2) > 10 && length(S1) > 1
            parfor k = 1:length(S2)
                j = S2(k);
                p_j = P(:,j);
                
                % 计算与已选特征的JS散度
                red_k = 0;
                for s = S1
                    p_s = P(:,s);
                    m_p = (p_j + p_s)/2;
                    JS = 0.5*(sum(p_j.*log((p_j+epsilon)./(m_p+epsilon))) + ...
                              sum(p_s.*log((p_s+epsilon)./(m_p+epsilon))));
                    red_k = red_k + exp(-JS);
                end
                redundancy(k) = red_k/length(S1); % 平均冗余
            end
        else
            % 串行计算冗余度
            for k = 1:length(S2)
                j = S2(k);
                p_j = P(:,j);
                
                % 计算与已选特征的JS散度
                for s = S1
                    p_s = P(:,s);
                    m_p = (p_j + p_s)/2;
                    JS = 0.5*(sum(p_j.*log((p_j+epsilon)./(m_p+epsilon))) + ...
                              sum(p_s.*log((p_s+epsilon)./(m_p+epsilon))));
                    redundancy(k) = redundancy(k) + exp(-JS);
                end
                redundancy(k) = redundancy(k)/length(S1); % 平均冗余
            end
        end
        
        % 综合评分 = 重要性 - 冗余惩罚
        scores = Ims(S2) - alpha * redundancy;
        
        % 选择最佳特征
        [maxScore, idx] = max(scores);
        selected = S2(idx);
        
        S1 = [S1 selected];
        S2(S2==selected) = [];
        finalScores(selected) = maxScore;
    end
    
    %% 返回排序结果
    [scores, order] = sort(finalScores, 'descend');
    selectedFeatures = order;
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


%% ========== 其他辅助函数 ==========
function val = getParam(params, field, default)
    if isfield(params, field)
        val = params.(field);
    else
        val = default;
    end
end

function displayResults(selected, scores, featNames)
    fprintf('\n=== 特征重要性排名 ===\n');
    fprintf('Rank\tFeature\t\tScore\tRelImportance\n');
    fprintf('--------------------------------\n');
    
    maxScore = max(scores);
    for i = 1:length(selected)
        fid = selected(i);
        name = featNames{fid};
        rel = 100*scores(i)/maxScore;
        fprintf('%d\t%s\t%.4f\t%.1f%%\n', i, name, scores(i), rel);
    end
    
    figure;
    barh(scores(end:-1:1));
    set(gca, 'YTick', 1:length(selected), 'YTickLabel', featNames(selected(end:-1:1)));
    xlabel('Importance Score');
    title('Feature Importance Ranking');
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

% function displayXGBoostKRVFLResults(optimalFeatures, results, config)
%     % 显示XGBoost-KRVFL特征选择结果并创建可视化图表
%     % 参数:
%     %   optimalFeatures - 最优特征的索引
%     %   results - 包含各特征数量的性能指标结果表
%     %   config - 运行配置参数
% 
%     % 1. 显示结果摘要
%     fprintf('\n=== XGBoost-KRVFL特征选择可视化结果 ===\n');
%     fprintf('优化配置: 粒子数=%d, 最大迭代=%d\n', ...
%         config.swarmSize, config.maxIterations);
% 
%     % 2. 获取最优特征数量和对应的索引位置
%     [~, optimalIdx] = min(results.Fitness);
%     optimalNumFeatures = results.NumFeatures(optimalIdx);
% 
%     % 3. 打印最优特征详情
%     fprintf('最优特征子集: %d 个特征\n', optimalNumFeatures);
% fprintf('最优特征索引: %s\n', mat2str(optimalFeatures));
% fprintf('最优性能指标:\n');
% fprintf('  RMSE:  %.4f\n', results.RMSE(optimalIdx));
% fprintf('  MAPE:  %.4f%%\n', results.MAPE(optimalIdx));
% fprintf('  MAE:   %.4f\n', results.MAE(optimalIdx));
% fprintf('  SMAPE: %.4f\n', results.SMAPE(optimalIdx));

    
    % % 4. 创建特征数量vs指标图表
    % figure('Name', 'XGBoost-KRVFL Performance Metrics by Feature Count');
    % 
    % % 4.1 RMSE图
    % subplot(2,2,1);
    % errorbar(results.NumFeatures, results.RMSE, results.RMSE_Std, 'b-o', 'LineWidth', 1.5);
    % hold on;
    % plot(optimalNumFeatures, results.RMSE(optimalIdx), 'r*', 'MarkerSize', 10);
    % hold off;
    % grid on;
    % title('RMSE vs Feature Count');
    % xlabel('Number of Features');
    % ylabel('RMSE');
    % 
    % % 4.2 MAPE图
    % subplot(2,2,2);
    % errorbar(results.NumFeatures, results.MAPE, results.MAPE_Std, 'g-o', 'LineWidth', 1.5);
    % hold on;
    % plot(optimalNumFeatures, results.MAPE(optimalIdx), 'r*', 'MarkerSize', 10);
    % hold off;
    % grid on;
    % title('MAPE vs Feature Count');
    % xlabel('Number of Features');
    % ylabel('MAPE (%)');
    % 
    % % 4.3 MAE图
    % subplot(2,2,3);
    % errorbar(results.NumFeatures, results.MAE, results.MAE_Std, 'm-o', 'LineWidth', 1.5);
    % hold on;
    % plot(optimalNumFeatures, results.MAE(optimalIdx), 'r*', 'MarkerSize', 10);
    % hold off;
    % grid on;
    % title('MAE vs Feature Count');
    % xlabel('Number of Features');
    % ylabel('MAE');
    % 
    % % 4.4 适应度图
    % subplot(2,2,4);
    % plot(results.NumFeatures, results.Fitness, 'k-o', 'LineWidth', 1.5);
    % hold on;
    % plot(optimalNumFeatures, results.Fitness(optimalIdx), 'r*', 'MarkerSize', 10);
    % hold off;
    % grid on;
    % title('Fitness vs Feature Count');
    % xlabel('Number of Features');
    % ylabel('Fitness Value');
    % 
    % % 调整图形大小和布局
    % set(gcf, 'Position', [100, 100, 900, 700]);
    % sgtitle('XGBoost-KRVFL Feature Selection Results');
    % 
    % % 5. 创建单一指标比较图
    % figure('Name', 'Optimal Feature Subset Metrics');
    % 
    % % 设置柱状图数据
    % metric_labels = {'RMSE', 'MAPE (%)', 'MAE', 'SMAE'};
    % metric_values = [
    %     results.RMSE(optimalIdx), 
    %     results.MAPE(optimalIdx), 
    %     results.MAE(optimalIdx), 
    %     results.SMAE(optimalIdx)
    % ]; 
    % metric_errors = [
    %     results.RMSE_Std(optimalIdx), 
    %     results.MAPE_Std(optimalIdx), 
    %     results.MAE_Std(optimalIdx), 
    %     results.SMAE_Std(optimalIdx)
    % ];
    % 
    % % 绘制带误差线的柱状图
    % bar(1:4, metric_values);
    % hold on;
    % errorbar(1:4, metric_values, zeros(size(metric_errors)), metric_errors, '.k');
    % hold off;
    % 
    % % 设置图形属性
    % set(gca, 'XTick', 1:4, 'XTickLabel', metric_labels);
    % title(['最优特征子集 (', num2str(optimalNumFeatures), ' 个特征) 性能指标']);
    % ylabel('指标值');
    % grid on;
    % 
    % % 注释最优值
    % for i = 1:4
    %     text(i, metric_values(i) + metric_errors(i) + 0.01, ...
    %         sprintf('%.4f ± %.4f', metric_values(i), metric_errors(i)), ...
    %         'HorizontalAlignment', 'center');
    % end
    % 
    % % 6. 打印特征数量与适应度关系表格
    % fprintf('\n特征数量与评估指标关系表:\n');
    % disp(results);
% end

% % 训练KRVFL模型
% H = compute_kernel_matrix(X_train_current, X_train_current, kernel_type, kernel_params);
% A = H * H' + eye(size(H,1)) / C;
% [temp, ~] = pcg(A, y_train_norm, 1e-6, 100);
% beta = H' * temp;(备用)
