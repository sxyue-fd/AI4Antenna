% generate_antenna_dataset.m
% 描述：使用遗传算法(GA)半优化策略生成小批量天线数据集

% 主要特性：
% 使用空气介质像素贴片天线为计算平台
% 数据集：X = [B,C,H,W]维的张量，通道C表示馈电位置和像素贴片
% 数据集标签: y = [B, S]，S11向量，以mag而非db表示
% 半优化（semi-optim）策略：使用遗传算法GA，增加数据集中良好匹配的数据点
% 数据集配比：50% 纯随机 + 50% 半优化 （待优化）
% 在半优化中使用低精度仿真加速


clear; clc; close all;

%% 0) User Configuration
% =========================================================================
% --- 数据集规模配置 ---
num_optimized_designs_target = 50;  % 目标通过GA生成的"半优化"样本数
num_random_designs_target    = 100;  % 目标纯随机生成的样本数

% --- GA 半优化配置 ---
ga_options.PopulationSize = 40;     % 种群大小
ga_options.MaxGenerations = 10;     % 最大迭代代数
ga_options.FitnessLimit   = -12;    % 适应度函数提前终止阈值 (例如S11 < -12dB)
ga_options.StallGenLimit  = 5;      % 如果5代最优解都没变化，则停止
ga_options.EliteCount     = 4;      % 精英数量
ga_options.CrossoverFraction = 0.7; % 交叉比例
ga_options.PlotFcn        = @gaplotbestf; % 绘制适应度曲线
ga_fitness_threshold      = -3;     % [dB] 用于从GA种群中筛选"好"天线的S11阈值

% --- 天线负载与仿真配置 ---
pixelResolution_N  = 16;      % 贴片分辨率 (N x N)
randomFillFactorRange    = [0.5, 0.9];     % 金属像素填充率范围 [min, max]
overlap_mm         = 0.8;     % 像素间重叠距离 (mm)

% --- 高保真仿真参数 (用于最终数据集) ---
hf_params.centerFreq_GHz     = 2.45;
hf_params.freqSpan_GHz       = 0.4;
hf_params.numFreqPoints      = 21;
hf_params.meshLambdaFraction = 20; % 更精细的网格

% --- 低保真仿真参数 (用于GA适应度函数) ---
lf_params.centerFreq_GHz     = 2.45;
lf_params.freqSpan_GHz       = 0.4;
lf_params.numFreqPoints      = 5;  % 更少的频点以加速
lf_params.meshLambdaFraction = 12; % 更粗糙的网格以加速

% --- 并行计算配置 ---
% 0表示使用所有可用worker, 您也可以指定一个固定值, e.g., 16
num_workers_to_use = 0; 
dataset_filename = sprintf('antenna_dataset_%s.h5', datestr(now,'yyyymmdd_HHMMSS')); % HDF5格式
% =========================================================================

%% 1) 初始化环境与设计参数
assert(license('test','Distrib_Computing_Toolbox')==1, '未检测到 Parallel Computing Toolbox 许可证。');
assert(license('test','Antenna_Toolbox')==1, '未检测到 Antenna Toolbox 许可证。');
assert(license('test','Optimization_Toolbox')==1, '未检测到 Optimization Toolbox 许可证。');

fprintf('正在计算高/低保真共享设计参数...\n');
% 计算高保真和低保真参数
designParams_HF = design_antenna_parameters(hf_params, pixelResolution_N, overlap_mm);
designParams_LF = design_antenna_parameters(lf_params, pixelResolution_N, overlap_mm);
fprintf('参数计算完成。\n\n');

% 一次性启动全局并行池
fprintf('正在启动全局并行池...\n');
if num_workers_to_use > 0
    pool = parpool(num_workers_to_use);
else
    pool = parpool();
end
 % 确保程序结束时关闭并行池
fprintf('并行池已启动，包含 %d 个 workers。\n\n', pool.NumWorkers);

%% 2) GA半优化阶段：收集优良设计
fprintf('=== （阶段1) 开始遗传算法半优化阶段（低精度计算) ===\n');
fprintf('目标：收集 S11 < %.1f dB 的设计\n\n', ga_fitness_threshold);

% 初始化用于收集优良设计的持久化变量
ga_output_collector('reset'); 

% 配置GA选项
num_vars = pixelResolution_N^2;
ga_opts = optimoptions('ga', ...
    'PopulationType', 'bitstring', ...
    'PopulationSize', ga_options.PopulationSize, ...
    'MaxGenerations', ga_options.MaxGenerations, ...
    'FitnessLimit', ga_options.FitnessLimit, ...
    'StallGenLimit', ga_options.StallGenLimit, ...
    'EliteCount', ga_options.EliteCount, ...
    'CrossoverFraction', ga_options.CrossoverFraction, ...
    'Display', 'iter', ...
    'PlotFcn', ga_options.PlotFcn, ...
    'UseParallel', true, ... % 在GA内部使用并行计算
    'OutputFcn', @(opts, state, flag) ga_output_collector(opts, state, flag, ga_fitness_threshold));

% 定义适应度函数句柄
fitness_fcn = @(x) fitness_function_antenna(x, designParams_LF);

% 启动计时器
tic_ga = tic;
% 运行GA
[~, ~, ~, ~] = ga(fitness_fcn, num_vars, [], [], [], [], [], [], [], ga_opts);
time_ga_lf = toc(tic_ga); % 计时结束

% 从OutputFcn中获取收集到的设计
optimized_designs_raw = ga_output_collector('get');

if isempty(optimized_designs_raw)
    warning('GA 未能找到任何满足阈值的设计！');
    optimized_designs = [];
else
    % 去重
    optimized_designs = unique(cell2mat(optimized_designs_raw), 'rows');
end

fprintf('\nGA半优化阶段完成。\n');
fprintf('  -> 原始收集设计数: %d\n', numel(optimized_designs_raw));
fprintf('  -> 去重后独特设计数: %d\n\n', size(optimized_designs, 1));


%% 3) 随机生成阶段
fprintf('=== (阶段2) 开始纯随机生成阶段 ===\n');
num_designs_to_generate = max(0, num_random_designs_target);
random_designs = zeros(num_designs_to_generate, num_vars);
rng('shuffle');
feed_linear_idx = sub2ind([pixelResolution_N, pixelResolution_N], ...
        designParams_HF.feedPixelIdx(1), designParams_HF.feedPixelIdx(2));

for i = 1:num_designs_to_generate
    % 随机生成可变填充率
    targetFill = randomFillFactorRange(1) + rand() * (randomFillFactorRange(2) - randomFillFactorRange(1));
    vec = rand(1, num_vars) < targetFill;
    % 强制馈电点为1
    vec(feed_linear_idx) = 1;
    random_designs(i, :) = vec;
end
fprintf('生成了 %d 个纯随机设计。\n\n', size(random_designs, 1));


%% 4) 最终高保真仿真与数据整合
fprintf('=== (阶段3) 开始最终高保真仿真阶段 ===\n');

% 合并所有需要仿真的设计
all_designs_to_simulate = [optimized_designs; random_designs];
num_total_tasks = size(all_designs_to_simulate, 1);

% --- 4.1 仿真GA设计 ---
num_ga_tasks = size(optimized_designs, 1);
results_ga = cell(1, num_ga_tasks);
if num_ga_tasks > 0
    fprintf('正在对 %d 个GA设计进行高保真仿真...\n', num_ga_tasks);
    tic_ga_hf = tic; % 计时开始
    parfor k = 1:num_ga_tasks
        results_ga{k} = simulate_single_antenna_hf(optimized_designs(k,:), designParams_HF);
    end
    time_ga_hf = toc(tic_ga_hf); % 计时结束
else
    time_ga_hf = 0;
end

% --- 4.2 仿真随机设计 ---
num_rand_tasks = size(random_designs, 1);
results_rand = cell(1, num_rand_tasks);
if num_rand_tasks > 0
    fprintf('正在对 %d 个随机设计进行高保真仿真...\n', num_rand_tasks);
    tic_rand_hf = tic; % 计时开始
    parfor k = 1:num_rand_tasks
        results_rand{k} = simulate_single_antenna_hf(random_designs(k,:), designParams_HF);
    end
    time_rand_hf = toc(tic_rand_hf); % 计时结束
else
    time_rand_hf = 0;
end
fprintf('高保真仿真完成。\n\n');

%% 5) 数据整理与保存
fprintf('=== (阶段4) 数据整理与保存 ===\n');

all_results = [results_ga, results_rand];
valid_indices = ~cellfun(@(res) isempty(res) || ~res.isValid, all_results);
num_valid_samples = sum(valid_indices);

if num_valid_samples == 0
    error('未能成功生成任何有效数据点，程序终止。');
end

% 初始化 (MATLAB格式)
X_matlab = zeros(pixelResolution_N, pixelResolution_N, 2, num_valid_samples);
Y_matlab = zeros(hf_params.numFreqPoints, num_valid_samples);
freq_vector = designParams_HF.freq_sweep;

count = 1;
for i = 1:length(all_results)
    if valid_indices(i)
        X_matlab(:,:,:,count) = all_results{i}.X;
        Y_matlab(:,count) = all_results{i}.Y;
        count = count + 1;
    end
end

% 转换为PyTorch格式
X_pytorch = permute(X_matlab, [4, 3, 1, 2]); % [B, C, H, W]
Y_pytorch = Y_matlab'; % [B, Features]

fprintf('成功生成 %d 个有效数据点。\n', num_valid_samples);
fprintf('  - PyTorch 输入 X 维度: %s\n', mat2str(size(X_pytorch)));
fprintf('  - PyTorch 输出 Y 维度: %s\n', mat2str(size(Y_pytorch)));

% 保存到 HDF5 文件
if exist(dataset_filename, 'file')
    delete(dataset_filename);
end
h5create(dataset_filename, '/X', size(X_pytorch), 'Datatype', 'single');
h5write(dataset_filename, '/X', single(X_pytorch)); % 常用单精度

h5create(dataset_filename, '/Y', size(Y_pytorch), 'Datatype', 'single');
h5write(dataset_filename, '/Y', single(Y_pytorch));

h5create(dataset_filename, '/freq_hz', size(freq_vector), 'Datatype', 'double');
h5write(dataset_filename, '/freq_hz', freq_vector);

fprintf('\n数据集已保存到: %s\n\n', dataset_filename);
cleanupObj = onCleanup(@() finalize_env());

%% 6) 性能总结
% =========================================================================
fprintf('=== 性能总结 ===\n');
fprintf('GA 低保真优化阶段耗时:         %.2f 秒\n', time_ga_lf);
fprintf('GA 设计高保真仿真阶段耗时:     %.2f 秒 (共 %d 个)\n', time_ga_hf, num_ga_tasks);
fprintf('随机设计高保真仿真阶段耗时:   %.2f 秒 (共 %d 个)\n', time_rand_hf, num_rand_tasks);
fprintf('---------------------------------------------------\n');
fprintf('总计高保真仿真耗时:             %.2f 秒\n', time_ga_hf + time_rand_hf);
fprintf('总程序运行耗时:                 %.2f 秒\n', toc(pool.StartTime));
% =========================================================================
%% ====== 辅助函数 ======

function params = design_antenna_parameters(sim_params, N, overlap_mm)
    f_center = sim_params.centerFreq_GHz * 1e9;
    c = physconst('LightSpeed');
    lambda0 = c / f_center;
    
    params.L = lambda0 / 2;
    params.W = params.L;
    params.h = lambda0 / 50;
    params.substrateMaterial = dielectric('Air');
    extension = 12 * params.h;
    board_L = params.L + extension;
    board_W = params.W + extension;
    params.ground = antenna.Rectangle('Length', board_L, 'Width', board_W, 'Center', [0 0]);

    params.pixelResolution_N = N;
    params.overlap = overlap_mm / 1000;
    pixel_L = params.L / N;
    pixel_W = params.W / N;
    
    params.pixelShapes = cell(N, N);
    startX = -params.L/2 + pixel_L/2;
    startY = -params.W/2 + pixel_W/2;
    for r = 1:N
        for c = 1:N
            centerX = startX + (c-1)*pixel_L;
            centerY = startY + (r-1)*pixel_W;
            params.pixelShapes{r,c} = antenna.Rectangle('Length', pixel_L + params.overlap, ...
                                                        'Width',  pixel_W + params.overlap, ...
                                                        'Center', [centerX, centerY]);
        end
    end
    
    initialFeedLocation = [params.L/4, 0];
    feed_c_idx = floor((initialFeedLocation(1) - (-params.L/2)) / pixel_L) + 1;
    feed_r_idx = floor((initialFeedLocation(2) - (-params.W/2)) / pixel_W) + 1;
    params.feedPixelIdx = [max(1, min(N, feed_r_idx)), max(1, min(N, feed_c_idx))];
    
    targetPixelShape = params.pixelShapes{params.feedPixelIdx(1), params.feedPixelIdx(2)};
    params.finalFeedLocation = targetPixelShape.Center;
    params.feedDiameter = min(pixel_L, pixel_W) / 10;
    
    f_start = (sim_params.centerFreq_GHz - sim_params.freqSpan_GHz/2) * 1e9;
    f_stop = (sim_params.centerFreq_GHz + sim_params.freqSpan_GHz/2) * 1e9;
    params.freq_sweep = linspace(f_start, f_stop, sim_params.numFreqPoints);
    params.maxEdge = lambda0 / sim_params.meshLambdaFraction;
end


function fitness = fitness_function_antenna(designVector, designParams_LF)
    % GA适应度函数，使用低保真参数进行快速评估
    N = designParams_LF.pixelResolution_N;
    designMatrix = reshape(designVector, N, N);

    % 强制馈电点为1
    designMatrix(designParams_LF.feedPixelIdx(1), designParams_LF.feedPixelIdx(2)) = 1;
    
    pixel_indices = find(designMatrix);
    if isempty(pixel_indices)
        fitness = 10; % 惩罚空设计
        return;
    end
    
    patchShape = designParams_LF.pixelShapes{pixel_indices(1)};
    for i = 2:length(pixel_indices)
        patchShape = patchShape + designParams_LF.pixelShapes{pixel_indices(i)};
    end
    
    ant = pcbStack(...
        'BoardShape', designParams_LF.ground, ...
        'BoardThickness', designParams_LF.h, ...
        'Layers', {patchShape, designParams_LF.substrateMaterial, designParams_LF.ground}, ...
        'FeedDiameter', designParams_LF.feedDiameter, ...
        'FeedLocations', [designParams_LF.finalFeedLocation, 1, 3]);
    
    try
        s = sparameters(ant, designParams_LF.freq_sweep);
        s11_db = rfparam(s, 1, 1, 'db'); 
        fitness = min(s11_db); % 目标是最小化S11(dB)，即让匹配更好
        if isnan(fitness) || isinf(fitness)
            fitness = 10; % 惩罚仿真失败
        end
    catch
        fitness = 10; % 惩罚建模或仿真错误
    end
end


function [state, options, optchanged] = ga_output_collector(options, state, flag, threshold_db)
    % GA的OutputFcn，用于在每一代收集满足条件的个体
    persistent good_designs;
    optchanged = false;

    if ischar(options) && strcmp(options, 'reset')
        good_designs = {};
        state = []; options = [];
        return;
    elseif ischar(options) && strcmp(options, 'get')
        state = good_designs; % 特殊调用方式，用于获取最终结果
        options = [];
        return;
    end
    
    if strcmp(flag, 'iter')
        scores = state.Score;
        population = state.Population;
        
        % 找到当前代中所有满足阈值的个体
        idx_good = find(scores < threshold_db);
        
        for i = 1:length(idx_good)
            good_designs{end+1} = population(idx_good(i), :);
        end
    end
end


function result = simulate_single_antenna_hf(designVector, designParams_HF)
    % 单个天线的高保真仿真函数，用于最终的parfor循环
    N = designParams_HF.pixelResolution_N;
    designMatrix = reshape(designVector, N, N);
    
    % 这个函数内不再强制馈电点，因为传入的设计已经处理过
    pixel_indices = find(designMatrix);
    if isempty(pixel_indices)
        result.isValid = false; return;
    end
    
    patchShape = designParams_HF.pixelShapes{pixel_indices(1)};
    for i = 2:length(pixel_indices)
        patchShape = patchShape + designParams_HF.pixelShapes{pixel_indices(i)};
    end
    
    ant = pcbStack(...
        'BoardShape', designParams_HF.ground, ...
        'BoardThickness', designParams_HF.h, ...
        'Layers', {patchShape, designParams_HF.substrateMaterial, designParams_HF.ground}, ...
        'FeedDiameter', designParams_HF.feedDiameter, ...
        'FeedLocations', [designParams_HF.finalFeedLocation, 1, 3]);
    
    try
        s = sparameters(ant, designParams_HF.freq_sweep);
        s11_mag = abs(squeeze(s.Parameters(1,1,:)));
        
        % 准备输出数据
        result.Y = s11_mag;
        
        X_data = zeros(N, N, 2);
        X_data(:,:,1) = designMatrix;
        feed_matrix = zeros(N, N);
        feed_matrix(designParams_HF.feedPixelIdx(1), designParams_HF.feedPixelIdx(2)) = 1;
        X_data(:,:,2) = feed_matrix;
        result.X = X_data;
        
        result.isValid = true;
    catch ME
        fprintf('高保真仿真失败: %s\n', ME.message);
        result.isValid = false;
    end
end

function finalize_env()
    % 清理函数
    try delete(gcp('nocreate')); end
    fprintf('并行池已关闭，环境已清理。\n');
end