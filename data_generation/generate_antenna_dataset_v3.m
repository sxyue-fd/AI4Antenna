function generate_antenna_dataset_v3()
% generate_antenna_dataset_v3.m
% =========================================================================
% 描述：
%   本脚本用于生成面向深度学习的小批量天线数据集。采用"GA半优化 + 随机"
%   两阶段策略，结合低/高保真电磁仿真，输出可直接用于 PyTorch 的
%   (X, Y) 张量和完整仿真配置的 HDF5 文件。
%
% 特点：
%   1) 使用像素化贴片天线 (N×N)，空气介质。
%   2) 个体编码 = [N^2 像素比特 | 馈电行bits | 馈电列bits]。
%      → 支持 N 非 2 的幂，通过二进制+clip 解码。
%   3) 半优化策略：遗传算法(GA)在低保真配置下寻找候选，Top-K 再送入
%      高保真仿真，保证一定比例的优质天线。
%   4) 随机策略：随机像素+随机馈电位置，补充多样性。
%   5) 高保真仿真：全波电磁仿真 (sparameters)，输出 S11 幅度谱。
%   6) 输出格式：
%        - X: [B, C=2, H=N, W=N] → 通道1=像素矩阵, 通道2=馈电位置掩码
%        - Y: [B, numFreqPoints] → S11 幅度 (mag)
%        - freq_hz: 频率向量
%        - meta_json: 完整配置 (参数、目标数、GA选项等)
%   7) 使用并行池加速 (parfor)，并在每个 worker 内强制 maxNumCompThreads(1)
%      避免过度线程化。
%
% 用户可配置内容：
%   - num_optimized_designs_target, num_random_designs_target
%   - GA 配置 (PopulationSize, MaxGenerations, FitnessLimit 等)
%   - 天线几何参数 (patch 尺寸、介质厚度、地板尺寸、初始馈电点直径)
%   - 高/低保真仿真频段、频点数量、网格划分参数
%   - 并行池 worker 数 (num_workers_to_use)
%   - 输出目录、进度打印频率
%
% 输出：
%   - dataset_out/antenna_dataset_<时间戳>.h5
%     包含 /X, /Y, /freq_hz, /meta_json
%
% 注意事项：
%   - 需安装 Antenna Toolbox, Optimization Toolbox, Parallel Computing Toolbox。
%   - 建议在 parpool 创建后执行 pctRunOnAll maxNumCompThreads(1)，
%     避免每个 worker 内再开多线程，提升 CPU 利用率。
%   - 高保真仿真耗时主要受网格划分与频扫影响，任务数大时请合理规划服务器资源。
%   
% 作者：sxyue
% 版本：v3 (2025.10.09)

% v3更新：
% 修改适应度和分数函数，引入带宽奖励项
% =========================================================================

clear; clc; close all;

%% 0) User Configuration
% =========================================================================
% --- 数据集规模配置 ---
num_optimized_designs_target = 128;  % 目标通过GA生成的"半优化"样本数
num_random_designs_target    = 1;  % 目标纯随机生成的样本数

% --- GA 半优化配置 ---
%  总候选=PopulationSize*MaxGenerations，从中选取top-K
ga_options.PopulationSize = 32;     % 种群大小
ga_options.Generations    = 16;     % 迭代代数
ga_options.FitnessLimit   = -Inf;   % 适应度函数早停阈值 (例如S11 < -40dB)
ga_options.StallGenLimit  = 100;    % 如果n代最优解都没变化，则早停
ga_options.EliteCount     = 1;      % 精英数量
ga_options.CrossoverFraction = 0.8; % 交叉比例
ga_options.mutationRate   = 0.02;   % 变异率
ga_options.PlotFcn        = @gaplotbestf; % 绘制适应度曲线

% --- 天线仿真配置 ---
pixelResolution_N  = 16;      % 贴片分辨率 (N x N)
randomFillFactorRange    = [0.5, 0.9];     % 金属像素填充率范围 [min, max]
overlap_mm         = 0.2;     % 像素间重叠距离 (mm)

geom.patch_L_mm     = 14;     % 贴片长度 L
geom.patch_W_mm     = 14;     % 贴片宽度 W
geom.sub_thick_mm   = 2.5;      % 介质厚度 h
geom.substrate_name= 'Air';   % 介质名，从MATLAB库中选取

geom.board_L_mm     = 30;
geom.board_W_mm     = 30;

geom.feed_init_xy  = [geom.patch_L_mm/4, 0]; % 初始馈电点（用于随机/默认）
geom.feed_diam_mm   = min(geom.patch_L_mm/pixelResolution_N, ...
                         geom.patch_W_mm/pixelResolution_N) / 10;


% --- 高保真仿真参数 (用于最终数据集) ---
hf_params.fmin_GHz     = 8;
hf_params.fmax_GHz     = 12;
hf_params.numFreqPoints      = 41;
hf_params.meshLambdaFraction = 10; % 更精细的网格

% --- 低保真仿真参数 (用于GA适应度函数) ---
lf_params.fmin_GHz     = 8;
lf_params.fmax_GHz     = 12;
lf_params.numFreqPoints      = 11;  % 更少的频点以加速
lf_params.meshLambdaFraction = 5; % 更粗糙的网格以加速

% --- 并行计算配置 ---
% 0表示使用所有可用worker, 也可以直接指定为cpu物理核数（不要用线程数）, e.g., 32
num_workers_to_use = 32; 

% --- 输出配置 ---
output_dir = fullfile(pwd, 'dataset_out');  % 统一输出文件夹
if ~exist(output_dir, 'dir'), mkdir(output_dir); end
ts = string(datetime("now","Format","yyyyMMdd_HHmmss"));  % 替代 datestr/now
dataset_filename = fullfile(output_dir, "antenna_dataset_" + ts + ".h5");
print_every = 20; % 每多少个点打印一次输出
% =========================================================================

%% 1) 初始化环境与设计参数
assert(license('test','Distrib_Computing_Toolbox')==1, '未检测到 Parallel Computing Toolbox 许可证。');
assert(license('test','Antenna_Toolbox')==1, '未检测到 Antenna Toolbox 许可证。');
assert(license('test','Optimization_Toolbox')==1, '未检测到 Optimization Toolbox 许可证。');

fprintf('正在计算高/低保真共享设计参数...\n');
% 计算高保真和低保真参数
designParams_HF = design_antenna_parameters(hf_params, geom, pixelResolution_N, overlap_mm);
designParams_LF = design_antenna_parameters(lf_params, geom, pixelResolution_N, overlap_mm);
fprintf('参数计算完成。\n\n');

% 一次性启动全局并行池
fprintf('正在启动全局并行池...\n');
if num_workers_to_use > 0
    pool = parpool(num_workers_to_use);
else
    pool = parpool();
end

try
    pctRunOnAll maxNumCompThreads(1)
catch ME
    warning('设定 worker 内线程数失败（可忽略）');
end

% 确保程序结束时关闭并行池
cleanupObj = onCleanup(@() finalize_env());
fprintf('并行池已启动，包含 %d 个 workers。\n\n', pool.NumWorkers);

% 启动全局计时器
tic_global = tic;
%% 2) GA半优化阶段：收集优良设计
fprintf('=== （阶段1) 开始遗传算法半优化阶段（低精度计算) ===\n');

% 配置GA选项
nbits = ceil(log2(pixelResolution_N));
num_vars = pixelResolution_N^2 + 2*nbits;
ga_opts = optimoptions('ga', ...
    'PopulationType', 'bitstring', ...
    'PopulationSize', ga_options.PopulationSize, ...
    'MaxGenerations', ga_options.Generations, ...
    'FitnessLimit', ga_options.FitnessLimit, ...
    'StallGenLimit', ga_options.StallGenLimit, ...
    'EliteCount', ga_options.EliteCount, ...
    'CrossoverFraction', ga_options.CrossoverFraction, ...
    'MutationFcn', {@mutationuniform, ga_options.mutationRate}, ... 
    'Display', 'iter', ...
    'PlotFcn', [], ...
    'UseParallel', true, ... % 在GA内部使用并行计算
    'OutputFcn', @ga_output_allpop_with_scores);

% 定义适应度函数句柄
fitness_fcn = @(x) fitness_function_antenna(x, designParams_LF);

% 启动计时器
tic_ga = tic;
% 运行GA
ga_output_allpop_with_scores('reset');
[~, ~, ~, ga_output] = ga(fitness_fcn, num_vars, [], [], [], [], [], [], [], ga_opts);
hist = ga_output_allpop_with_scores('get');

% 合并所有代
P = vertcat(hist.vecs{:});        % [M x D]，D=num_vars
S = vertcat(hist.scores{:});      % [M x 1]

% 去重：对 bitstring 做哈希，重复项保留"最优分数"（更小）
keys = cell(size(P,1),1);
for i = 1:size(P,1)
    keys{i} = DataHash(P(i,:));   % 你已有的 DataHash；没有就用 md5
end
[ukeys, ~, ic] = unique(keys, 'stable');

% 聚合每个唯一键的"最佳分数"及其对应向量
best_score = inf(numel(ukeys),1);
best_vec   = false(numel(ukeys), size(P,2));
for i = 1:numel(ic)
    k = ic(i);
    if S(i) < best_score(k)
        best_score(k) = S(i);
        best_vec(k,:) = P(i,:);
    end
end

% 现在 best_vec/best_score 是"唯一候选池 + 已有分数"
[~, ord] = sort(best_score, 'ascend');  % dB 越小越好
candidates = best_vec(ord,:);
scores     = best_score(ord);

% 取 Top-K
K = num_optimized_designs_target;
takeK = min(K, size(candidates,1));
optimized_designs = candidates(1:takeK, :);
optimized_scores  = scores(1:takeK);     %#ok<NASGU> % 如需记录

% 若仍不足（极端情况）补齐随机
if size(optimized_designs,1) < K
    warning('候选不足，随机补齐至目标数；建议增大 PopulationSize/Generations 或提高变异率。');
    need = K - size(optimized_designs,1);
    pad  = randi([0 1], need, size(candidates,2));
    optimized_designs = [optimized_designs; pad];
end

time_ga_lf = toc(tic_ga); % 计时结束
fprintf('\nGA半优化阶段完成（Top-K 选择）。\n');
fprintf('  -> 候选数: %d，选取 %d 个。\n\n', size(candidates,1), size(optimized_designs,1));


%% 3) 随机生成阶段
fprintf('=== (阶段2) 开始纯随机生成阶段 ===\n');
num_designs_to_generate = max(0, num_random_designs_target);

% ---- 位数：像素 N^2 + 馈电行/列二进制位 ----
N = pixelResolution_N;
nbits = ceil(log2(N));


rng('shuffle');

% 1) 每个样本的随机填充率（行向量/列向量均可）
fill_vec = randomFillFactorRange(1) + (randomFillFactorRange(2) - randomFillFactorRange(1)) * rand(num_designs_to_generate, 1);

% 2) 向量化生成像素比特：U < fill_i
U = rand(num_designs_to_generate, N^2);     % 每行一个样本
pix_bits = U < fill_vec;                    % 隐式扩展，得到 logical(B, N^2)

% 3) 随机馈电位置（1..N），并把该像素强制置 1（列主序线性索引）
feed_r = randi([1, N], num_designs_to_generate, 1);
feed_c = randi([1, N], num_designs_to_generate, 1);
lin_idx_pix = feed_r + (feed_c - 1) * N;    % N×N 的列主序线性索引
pix_bits(sub2ind([num_designs_to_generate, N^2], (1:num_designs_to_generate)', lin_idx_pix)) = true;

% 4) 馈电行/列的二进制编码（MSB-first），val0 ∈ [0..N-1]
r0 = uint16(feed_r - 1);  c0 = uint16(feed_c - 1);
r_bits = false(num_designs_to_generate, nbits);
c_bits = false(num_designs_to_generate, nbits);
for k = 1:nbits
    % bitget 第 k 位（LSB-first），我们写到 MSB-first 列（nbits-k+1）
    r_bits(:, nbits-k+1) = bitget(r0, k);
    c_bits(:, nbits-k+1) = bitget(c0, k);
end

% 5) 拼接得到最终设计向量（与 GA 完全一致的格式）
random_designs = [pix_bits, r_bits, c_bits];  % logical(B, N^2+2*nbits)
fprintf('生成了 %d 个纯随机设计。\n\n', size(random_designs, 1));

%% 4) 最终高保真仿真与数据整合
fprintf('=== (阶段3) 开始最终高保真仿真阶段 ===\n');

% --- 4.1 仿真GA设计 ---
num_ga_tasks = size(optimized_designs, 1);
results_ga = cell(1, num_ga_tasks);

if num_ga_tasks > 0
    fprintf('正在对 %d 个GA设计进行高保真仿真...\n', num_ga_tasks);
    [dq_ga, ~] = setup_progress_tracker(num_ga_tasks, print_every, 'GA-HF');
    tic_ga_hf = tic; % 计时开始
    parfor k = 1:num_ga_tasks
        results_ga{k} = simulate_single_antenna_hf(optimized_designs(k,:), designParams_HF);
        send(dq_ga, 1);
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
    [dq_rd, ~] = setup_progress_tracker(num_rand_tasks, print_every, 'RAND-HF');
    tic_rand_hf = tic; % 计时开始
    parfor k = 1:num_rand_tasks
        results_rand{k} = simulate_single_antenna_hf(random_designs(k,:), designParams_HF);
        send(dq_rd, 1);
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

% 压缩 & 分块（更快更小）
B = size(X_pytorch,1); C = size(X_pytorch,2); H = size(X_pytorch,3); W = size(X_pytorch,4);
chunkX = [min(64,B) C H W];
h5create(dataset_filename, '/X', [B C H W], 'Datatype','single', ...
         'ChunkSize',chunkX,'Deflate',5);
h5write(dataset_filename, '/X', single(X_pytorch)); % 常用单精度

chunkY = [min(256,B) size(Y_pytorch,2)];
h5create(dataset_filename, '/Y', size(Y_pytorch), 'Datatype', 'single', ...
         'ChunkSize',chunkY,'Deflate',5);
h5write(dataset_filename, '/Y', single(Y_pytorch));

h5create(dataset_filename, '/freq_hz', size(freq_vector), 'Datatype', 'double');
h5write(dataset_filename, '/freq_hz', freq_vector);

% 把完整配置写入 HDF5 属性（JSON）
meta = struct('hf_params',hf_params,'lf_params',lf_params, ...
              'geom',geom,'N',pixelResolution_N,'overlap_mm',overlap_mm, ...
              'targets',struct('ga',num_optimized_designs_target,'rand',num_random_designs_target), ...
              'ga_options',ga_options,'timestamp',char(datetime("now")));

meta_json = jsonencode(make_json_serializable(meta));
h5writeatt(dataset_filename, '/', 'meta_json', meta_json);

fprintf('\n数据集已保存到: %s\n\n', dataset_filename);

%% 6) 性能总结
% =========================================================================
fprintf('=== 性能总结 ===\n');
fprintf('GA 低保真优化阶段耗时:         %.2f 秒(共 %d 次仿真)\n', time_ga_lf, ga_output.funccount);
fprintf('GA 设计高保真仿真阶段耗时:     %.2f 秒 (共 %d 次仿真)\n', time_ga_hf, num_ga_tasks);
fprintf('随机设计高保真仿真阶段耗时:   %.2f 秒 (共 %d 次仿真)\n', time_rand_hf, num_rand_tasks);
fprintf('---------------------------------------------------\n');
fprintf('总计高保真仿真耗时:             %.2f 秒\n', time_ga_hf + time_rand_hf);
fprintf('总程序运行耗时:                 %.2f 秒\n', toc(tic_global));
% =========================================================================
end
%% ====== 辅助函数 ======

function params = design_antenna_parameters(sim_params, geom, N, overlap_mm)
    f_center = ((sim_params.fmin_GHz + sim_params.fmax_GHz)/2) * 1e9;
    c = physconst('LightSpeed');
    lambda0 = c / f_center;
    
    % 基本几何
    params.L = geom.patch_L_mm / 1e3;
    params.W = geom.patch_W_mm / 1e3;
    params.h = geom.sub_thick_mm / 1e3;

    % 介质
    params.substrateMaterial = dielectric(geom.substrate_name);
    params.substrateMaterial.Thickness = params.h;

    % 地板
    params.ground = antenna.Rectangle('Length', geom.board_L_mm / 1e3, ...
                                      'Width',  geom.board_W_mm / 1e3, ...
                                      'Center', [0 0]);

    % 像素化参数
    params.pixelResolution_N = N;
    params.overlap = overlap_mm / 1000;
    pixel_L = params.L / N;  pixel_W = params.W / N;

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

    % 初始馈电像素索引（仅作为默认或随机阶段使用）
    feed_c_idx = floor((geom.feed_init_xy(1) - (-params.L/2)) / pixel_L) + 1;
    feed_r_idx = floor((geom.feed_init_xy(2) - (-params.W/2)) / pixel_W) + 1;
    params.feedPixelIdx = [max(1, min(N, feed_r_idx)), max(1, min(N, feed_c_idx))];

    % 注意：最终馈电点位置由传入的 feedIdx 决定（见 create_antenna_model）
    targetPixelShape = params.pixelShapes{params.feedPixelIdx(1), params.feedPixelIdx(2)};
    params.finalFeedLocation = targetPixelShape.Center;
    params.feedDiameter = geom.feed_diam_mm / 1e3;

    % 频率与网格
    f_start = sim_params.fmin_GHz * 1e9;
    f_stop  = sim_params.fmax_GHz * 1e9;
    params.freq_sweep = linspace(f_start, f_stop, sim_params.numFreqPoints);
    params.maxEdge = lambda0 / sim_params.meshLambdaFraction;
end


function fitness = fitness_function_antenna(designVector, designParams_LF)
    % GA适应度函数，使用低保真参数进行快速评估
    % --- 解码像素与馈电位置（兼容旧/新两种向量长度）---
    N = designParams_LF.pixelResolution_N;
    nbits = ceil(log2(N));
    dv = designVector(:)';  % 保证行向量
    expected_new = N^2 + 2*nbits;

    if numel(dv) == expected_new
        % 新格式：像素 + 馈电行/列二进制位
        [pix_bits, feed_rc] = split_design_bits(dv, N);
    elseif numel(dv) == N^2
        % 兼容旧格式：只有像素，馈电位置用默认（或 HF/LF 中的 feedPixelIdx）
        pix_bits = dv;
        feed_rc  = designParams_LF.feedPixelIdx;
    else
        error('设计向量长度不匹配：得到 %d，但期望 %d（新）或 %d（旧）。', ...
              numel(dv), expected_new, N^2);
    end

    % 像素矩阵
    designMatrix = reshape(pix_bits, N, N);

    % 强制馈电像素为金属
    designMatrix(feed_rc(1), feed_rc(2)) = 1;
    
    pixel_indices = find(designMatrix);
    if isempty(pixel_indices)
        fitness = 10; % 惩罚空设计
        return;
    end
    
    rectCells = designParams_LF.pixelShapes(pixel_indices);
    patchShape = union_rectangles_batch(rectCells, 256);

    ant = create_antenna_model(patchShape, feed_rc, designParams_LF);
    
    try
        s = sparameters(ant, designParams_LF.freq_sweep);
        s11_complex = squeeze(s.Parameters(1,1,:));
        s11_db = 20 * log10(abs(s11_complex + eps));

        % 使用s11最小值和带宽的组合适应度函数
        % 1) 原始适应度项：S11最小值 (代表匹配深度)
        term_s11 = min(s11_db);
        
        % 2) 带宽奖励项：低于-10dB的频点数量 (代表匹配宽度)
        %    我们希望最大化这个数量，因此在cost中它应该是负的奖励
        num_points_below_10db = sum(s11_db < -10);

        % 3) 定义带宽奖励的权重 (w_bw)
        %    这是一个超参数，用于平衡"深度"和"宽度"的重要性。
        %    term_s11 的典型值为-30，num_points_below_10db 的典型值为5（5%带宽，0.1GHz间隔）
        %    选择 w_bw = 6 使得带宽奖励的量级与S11项大致相当。
        w_bw = 6; 
        
        % 3) 组合适应度函数
        %    我们的目标是最小化cost，所以奖励项要用减法。
        fitness = term_s11 - w_bw * num_points_below_10db;

        if isnan(fitness) || isinf(fitness)
            fitness = 10; % 惩罚仿真失败
        end
    catch
        fitness = 10; % 惩罚建模或仿真错误
    end
end


function [state, options, optchanged] = ga_output_allpop_with_scores(options, state, flag)
% 收集每一代的全量种群及其适应度分数；支持 'reset' / 'get'
    persistent all_vecs all_scores
    optchanged = false;

    if ischar(options)
        switch options
            case 'reset'
                all_vecs = {}; all_scores = {};
                state = []; options = [];
                return;
            case 'get'
                % 返回 cell -> 由调用方合并
                state   = struct('vecs',{all_vecs}, 'scores',{all_scores});
                options = [];
                return;
        end
    end

    if strcmp(flag, 'iter')
        % 每一代：追加这代的 Population 和 Score
        all_vecs{end+1}   = state.Population; 
        all_scores{end+1} = state.Score;      
    end
end



function result = simulate_single_antenna_hf(designVector, designParams_HF)
    % 单个天线的高保真仿真函数，用于最终的parfor循环
    N = designParams_HF.pixelResolution_N;
    nbits = ceil(log2(N));
    dv = designVector(:)';  % 保证行向量
    expected_new = N^2 + 2*nbits;

    if numel(dv) == expected_new
        [pix_bits, feed_rc] = split_design_bits(dv, N);
    elseif numel(dv) == N^2
        pix_bits = dv;
        feed_rc  = designParams_HF.feedPixelIdx;
    else
        error('设计向量长度不匹配：得到 %d，但期望 %d（新）或 %d（旧）。', ...
              numel(dv), expected_new, N^2);
    end

    designMatrix = reshape(pix_bits, N, N);
    designMatrix(feed_rc(1), feed_rc(2)) = 1;

    pixel_indices = find(designMatrix);
    if isempty(pixel_indices)
        result.isValid = false; return;
    end
    
    rectCells = designParams_HF.pixelShapes(pixel_indices);
    patchShape = union_rectangles_batch(rectCells, 64);
    
    ant = create_antenna_model(patchShape, feed_rc, designParams_HF);
    
    try
        s = sparameters(ant, designParams_HF.freq_sweep);
        s11_mag = abs(squeeze(s.Parameters(1,1,:)));
        
        % 准备输出数据
        result.Y = s11_mag;
        
        X_data = zeros(N, N, 2);
        X_data(:,:,1) = designMatrix;
        feed_matrix = zeros(N, N);
        feed_matrix(feed_rc(1), feed_rc(2)) = 1;
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

function [dq, tick_fn] = setup_progress_tracker(total_count, print_every, label)
% 用 DataQueue + 闭包打印进度；parfor 内 send(dq, 1) 即可。
    if nargin < 3, label = 'Progress'; end
    dq = parallel.pool.DataQueue;
    S = struct('done',0,'total',total_count,'t0',tic,'print_every',max(1,print_every));
    afterEach(dq, @(~) tick());
    function tick()
        S.done = S.done + 1;
        if mod(S.done, S.print_every)==0 || S.done==S.total
            t = toc(S.t0);
            fprintf('\r[%s] %d/%d done (%.1fs elapsed)', label, S.done, S.total, t);
            if S.done==S.total, fprintf('\n'); end
        end
    end
    tick_fn = @tick; % 备用：非 parfor 场景手动 tick
end

function shape = union_rectangles_batch(rectCells, batch)
% rectCells: {1xK} 的 antenna.Rectangle cell（非空）
% batch: 每多少个执行一次简化；默认 64
    if nargin < 2, batch = 64; end
    ids = find(~cellfun('isempty', rectCells));
    if isempty(ids), shape = []; return; end
    shape = rectCells{ids(1)};
    cnt = 1;
    for ii = 2:numel(ids)
        shape = shape + rectCells{ids(ii)};
        cnt = cnt + 1;
        if mod(cnt, batch) == 0
            try
                shape = shape.simplify; % 新版 toolbox 支持；旧版可忽略
            catch
            end
        end
    end
end

function S2 = make_json_serializable(S)
% 递归地将结构中的 function_handle / objects 转为可 jsonencode 的类型
    if isa(S, 'function_handle')
        S2 = func2str(S);
    elseif isstruct(S)
        fn = fieldnames(S);
        for k = 1:numel(fn)
            S.(fn{k}) = make_json_serializable(S.(fn{k}));
        end
        S2 = S;
    elseif iscell(S)
        for k = 1:numel(S)
            S{k} = make_json_serializable(S{k});
        end
        S2 = S;
    elseif isa(S, 'optim.options.GA')
        % options 对象也不好直接编码，转成 struct 再清洗
        S2 = make_json_serializable(struct(S));
    else
        S2 = S; % 数值/字符/字符串/逻辑/表等可直接编码
    end
end

function [pix_bits, feed_rc] = split_design_bits(vec, N)
% vec: [1 x (N^2 + nbits_r + nbits_c)] 的 bit 向量
    nbits = ceil(log2(N));
    pix_bits = vec(1:N^2);
    feed_r_bits = vec(N^2 + (1:nbits));
    feed_c_bits = vec(N^2 + nbits + (1:nbits));
    feed_r = bits2int(feed_r_bits) + 1;  % [0..N-1] -> [1..N]
    feed_c = bits2int(feed_c_bits) + 1;
    feed_rc = [min(N,max(1,feed_r)), min(N,max(1,feed_c))];
end

function v = bits2int(b)
% b: [1 x nbits] logical/double in {0,1}; MSB-first
    n = numel(b); v = 0;
    for i = 1:n
        v = bitshift(v,1) + (b(i)~=0);
    end
end

function bits = int2bits(val, nbits)
% val: >=0 的整数；输出 MSB-first 二进制
    bits = false(1,nbits);
    for i = nbits:-1:1
        bits(i) = bitand(val,1);
        val = bitshift(val,-1);
    end
end

function ant = create_antenna_model(patchShape, feed_rc, params)
% feed_rc: [row, col] 像素索引，决定最终 FeedLocations
    N = params.pixelResolution_N;
    pixel_L = params.L / N;  pixel_W = params.W / N;
    startX = -params.L/2 + pixel_L/2;
    startY = -params.W/2 + pixel_W/2;

    cx = startX + (feed_rc(2)-1)*pixel_L;
    cy = startY + (feed_rc(1)-1)*pixel_W;

    ant = pcbStack( ...
        'BoardShape', params.ground, ...
        'BoardThickness', params.h, ...
        'Layers', {patchShape, params.substrateMaterial, params.ground}, ...
        'FeedDiameter', params.feedDiameter, ...
        'FeedLocations', [cx, cy, 1, 3] );

    try
        m = mesh(ant, 'MaxEdgeLength', params.maxEdge);
    catch
        warning('网格控制失败，将采用默认网格');
    end
end

function h = DataHash(A)
% 简易哈希：对 uint8 序列做 MD5
    if ~isa(A,'uint8')
        A = uint8(A);
    end
    md = java.security.MessageDigest.getInstance('MD5');
    md.update(A(:));
    h = char(org.apache.commons.codec.binary.Hex.encodeHex(md.digest())).';
end