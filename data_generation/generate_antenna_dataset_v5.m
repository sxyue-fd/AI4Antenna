function generate_antenna_dataset_v5()
% generate_antenna_dataset_v5.m
% =========================================================================
% 描述：
%   本脚本用于生成面向深度学习的高质量宽带像素化贴片天线数据集。
%   相比早期版本，本版本引入"biased evolution（偏置进化）"
%   采样策略，在低保真电磁仿真下定向搜索宽带匹配天线结构，
%   并将筛选得到的 Top-K 设计送入高保真全波仿真，最终输出
%   可直接用于 PyTorch 训练的 (X, Y) 数据集及完整实验元数据。
% 方法概述：
%   - 设计空间：N×N 像素化贴片天线，空气介质，单端馈电。
%   - 目标：在指定频段内获得宽带、深度良好且单主通带的
%           S11 匹配响应。
%   - 策略：
%       (1) 低保真阶段：基于 biased evolution 的宽带定向采样
%       (2) 高保真阶段：对优选样本进行全波精细仿真
%       (3) （可选）随机像素设计用于补充数据多样性
%特点：
% Biased Evolution：
%      - 初始随机像素池（填充率分布可控）。
%      - 基于指数偏置的选择策略（beta），兼顾探索与开发。
%      - 变异 + 交叉生成子代，并通过 archive 去重保留多样性。
%      - 在低保真仿真下进行多轮进化，显著提升宽带样本命中率。
% 宽带 cost 设计（低保真阶段）：
%      cost 由以下部分加权组成：
%      - 连续 -10 dB 主通带带宽奖励（饱和型）
%      - S11 深度奖励（限幅，避免尖峰支配）
%      - 多通带数量惩罚
%      - 主通带之外额外匹配带宽惩罚
%      - 主通带中心频率偏离目标 f0 的惩罚
%
%数据输出格式（面向 PyTorch）：
%        - X: [B, C=2, H=N, W=N]
%             通道1：像素金属分布
%             通道2：馈电位置掩码
%        - Y: [B, numFreqPoints]
%             S11 幅度谱 |S11|
%        - freq_hz: 频率向量
%        - meta_json: 完整实验配置、参数、统计与耗时信息
%
% 用户可配置内容：
%   - 像素分辨率 N、像素填充率范围
%   - biased evolution 参数（init_pool_size, num_iters,
%     children_per_iter, mutation/crossover 概率等）
%   - 宽带 cost 权重（带宽、深度、多通带、中心频率约束）
%   - 高/低保真仿真频段、频点数、网格精度
%   - 并行 worker 数、输出路径与打印频率
%
% 输出：
%   - dataset_out/antenna_dataset_<timestamp>.h5
%     包含：
%       /X, /Y, /freq_hz, HDF5 根属性 meta_json
%
% 注意事项：
%   - 依赖工具箱：
%       * Antenna Toolbox
%       * Optimization Toolbox
%       * Parallel Computing Toolbox
%   - 高保真仿真耗时主要受网格精度与频扫点数影响，
%     建议在服务器或多核工作站上运行。
%   - meta_json 中记录了完整 cfg 与运行统计
%
% 作者：xyzhu
% 版本：v5 (2026.1.14)
%
% v5 更新要点：
%   - 引入biased evolution 宽带采样策略
%   - 低保真阶段显式优化"连续宽带 + 单主通带"目标
%   - 新增主通带中心频率约束（f0 penalty）
% =========================================================================
clear; clc; close all;

%% 0) User Configuration
% =========================================================================
% --- 数据集规模配置 ---
num_top_per_feed           = 200;    % 每个馈电位置保留的biased evolution优秀样本数
num_random_designs_target    = 50;  % 目标纯随机生成的样本数

% --- 天线仿真配置 ---
pixelResolution_N  = 16;      % 贴片分辨率 (N x N)
num_optimized_designs_target = pixelResolution_N^2 * num_top_per_feed;
randomFillFactorRange    = [0.6, 0.9];     % 金属像素填充率范围 [min, max]
overlap_mm         = 0.2;     % 像素间重叠距离 (mm)

geom.patch_L_mm     = 14     ;     % 贴片长度 L
geom.patch_W_mm     = 14;     % 贴片宽度 W
geom.sub_thick_mm   = 2.5;      % 介质厚度 h
geom.substrate_name= 'Air';   % 介质名，从MATLAB库中选取

geom.board_L_mm     = 30;
geom.board_W_mm     = 30;

geom.feed_init_xy  = [geom.patch_L_mm/4, 0]; % 初始馈电点（用于随机/默认）
geom.feed_diam_mm   = min(geom.patch_L_mm/pixelResolution_N, ...
                         geom.patch_W_mm/pixelResolution_N) / 10;

 cfg = struct();
        cfg.init_pool_size   = 600;   % 初始随机池子大小（越大越好，但更慢）
        cfg.num_iters        = 20;    % 进化轮数
        cfg.children_per_iter= 100;   % 每轮生成多少子代
        cfg.beta     = 0.15;          % 选择偏置强度（越大越贪）
        cfg.keep_size        = 5000;  % archive 保留池大小（去重后再保留）
        cfg.pm               = 0.05;  % mutation 翻转率（建议 0.01~0.05）
        cfg.pc               = 0.4;   % crossover 概率（0~0.5 常见）
        cfg.roi_fmin_GHz     = 8;     % 只在 ROI 里算 cost
        cfg.roi_fmax_GHz     = 12;
        cfg.bw               = 0.12;  % 饱和带宽
        cfg.w_bw             = 1.5;   % 带宽奖励权重（在 cost 内部会归一化）
        cfg.depth            = 0.5;   % 深度奖励权重
        cfg.depth_thr_db     = -20;   % 深度项限幅
        cfg.thr_db           = -10;   % 匹配阈值
        cfg.pp2              =0;      % P2偏置选择的概率
        cfg.w_multiband_count   = 8.0;   % 惩罚额外通带"个数"
        cfg.w_multiband_extraBW = 20.0;  % 惩罚主通带之外的匹配带宽
        cfg.f0_GHz = 10;     % 目标主通带中心频率
        cfg.df_GHz = 0.5;    % 容忍尺度：0.5GHz（越小约束越严格）
        cfg.w_f0   = 4.0;    % 频率惩罚权重：建议从 2~10 试

sim_calls=cfg.children_per_iter*cfg.num_iters+cfg.init_pool_size;
% --- 高保真仿真参数 (用于最终数据集) ---
hf_params.fmin_GHz     = 8;
hf_params.fmax_GHz     = 12;
hf_params.numFreqPoints      = 41;
hf_params.meshLambdaFraction = 10; % 更精细的网格

% --- 低保真仿真参数 (用于biased evolution) ---
lf_params.fmin_GHz     = 8;
lf_params.fmax_GHz     = 12;
lf_params.numFreqPoints      = 21;  % 更少的频点以加速
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
%% 2)biased evolution（偏置进化）每个 feed 点生成 K 个宽带样本
fprintf('=== (阶段1) 开始biased evolution ===\n');
% 启动全局计时器
tic_global = tic;
N = pixelResolution_N;
K = num_top_per_feed;
optimized_designs = false(N*N*K, N^2);%现在只跑一个馈电点，否则应为N*N
optimized_scores  = zeros(N*N*K, 1);%现在只跑一个馈电点，否则应为N*N
optimized_feed_rc = zeros(N*N*K, 2);%现在只跑一个馈电点，否则应为N*N

tic_lf = tic;

write_ptr = 1;

for r_feed = 1:N
    for c_feed = 1:N
        feed_rc_fixed = [r_feed, c_feed];
       [top_designs, top_costs] = sample_wideband_designs_for_feed( ...
    designParams_LF, feed_rc_fixed, K, cfg, randomFillFactorRange);

        range = write_ptr:(write_ptr+K-1);
        optimized_designs(range,:) = top_designs;
        optimized_scores(range)    = top_costs;
        optimized_feed_rc(range,:) = repmat(feed_rc_fixed, K, 1);
        write_ptr = write_ptr + K;
        fprintf('Feed (%2d,%2d): got %d wideband samples.\n', r_feed, c_feed, K);
    end
end
final_topK = size(optimized_designs,1);%最终总样品数
time_BE_lf = toc(tic_lf);
fprintf('Biased Evolution 低保真采样阶段耗时: %.2f 秒\n', time_BE_lf);
fprintf('低保真：每个馈电点评估样品数 = %d', sim_calls);
fprintf('低保真：每个馈电点最终保留样本数(top-K) = %d（K=%d）\n', size(top_designs,1), K);

%% 3) 随机生成阶段
fprintf('=== (阶段2) 开始纯随机生成阶段 ===\n');
num_designs_to_generate = max(0, num_random_designs_target);

% ---- 位数：像素 N^2 + 馈电行/列二进制位 ----
N = pixelResolution_N;

rng('shuffle');

% 1) 每个样本的随机填充率（行向量/列向量均可）
fill_vec = randomFillFactorRange(1) + (randomFillFactorRange(2) - randomFillFactorRange(1)) * rand(num_designs_to_generate, 1);

% 2) 向量化生成像素比特：U < fill_i
U = rand(num_designs_to_generate, N^2);
pix_bits = U < fill_vec;    

feed_r = randi([1, N], num_designs_to_generate, 1);
feed_c = randi([1, N], num_designs_to_generate, 1);
lin_idx_pix = feed_r + (feed_c - 1) * N;
pix_bits(sub2ind([num_designs_to_generate, N^2], (1:num_designs_to_generate)', lin_idx_pix)) = true;

% 染色体只保留像素 bits
random_designs = pix_bits;    % logical(B, N^2)
fprintf('生成了 %d 个纯随机设计。\n\n', size(random_designs, 1));

%% 4) 最终高保真仿真与数据整合
fprintf('=== (阶段3) 开始最终高保真仿真阶段 ===\n');

% --- 4.1 仿真biased evolution设计 ---
num_BE_tasks = size(optimized_designs, 1);
results_BE = cell(1, num_BE_tasks);

if num_BE_tasks > 0
    fprintf('正在对 %d 个biased evolution设计进行高保真仿真...\n', num_BE_tasks);
    [dq_BE, ~] = setup_progress_tracker(num_BE_tasks, print_every, 'BE-HF');
    tic_BE_hf = tic; % 计时开始
    parfor k = 1:num_BE_tasks
        feed_rc = optimized_feed_rc(k, :);   % [r, c]
        results_BE{k} = simulate_single_antenna_hf(optimized_designs(k,:), designParams_HF, feed_rc);
        send(dq_BE, 1);
    end
    time_BE_hf = toc(tic_BE_hf); % 计时结束
else
    time_BE_hf = 0;
end

% --- 4.2 仿真随机设计 ---
num_rand_tasks = size(random_designs, 1);
results_rand = cell(1, num_rand_tasks);
if num_rand_tasks > 0
    fprintf('正在对 %d 个随机设计进行高保真仿真...\n', num_rand_tasks);
    [dq_rd, ~] = setup_progress_tracker(num_rand_tasks, print_every, 'RAND-HF');
    tic_rand_hf = tic; % 计时开始
    parfor k = 1:num_rand_tasks
        feed_rc = [feed_r(k), feed_c(k)];
        results_rand{k} = simulate_single_antenna_hf(random_designs(k,:), designParams_HF, feed_rc);
        send(dq_rd, 1);
    end
    time_rand_hf = toc(tic_rand_hf); % 计时结束
else
    time_rand_hf = 0;
end
fprintf('高保真仿真完成。\n\n');

%% 5) 数据整理与保存
fprintf('=== (阶段4) 数据整理与保存 ===\n');

all_results = [results_BE, results_rand];
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
% ========== 把更完整的配置/运行信息写入 HDF5 属性（JSON） ==========
run_info = struct();
run_info.timestamp      = char(datetime("now"));
run_info.dataset_file   = dataset_filename;

% --- 规模与目标 ---
run_info.targets = struct( ...
    'num_top_per_feed', num_top_per_feed, ...
    'num_optimized_designs_target', num_optimized_designs_target, ...
    'num_random_designs_target', num_random_designs_target );

% --- 随机策略相关 ---
run_info.random = struct( ...
    'randomFillFactorRange', randomFillFactorRange );

% --- 并行配置 ---
run_info.parallel = struct( ...
    'num_workers_to_use', num_workers_to_use, ...
    'print_every', print_every );

% --- 关键参数快照 ---
run_info.params = struct( ...
    'hf_params', hf_params, ...
    'lf_params', lf_params, ...
    'geom', geom, ...
    'N', pixelResolution_N, ...
    'overlap_mm', overlap_mm );

% --- biased evolution 的 cfg
run_info.biased_evo_cfg = cfg;

% --- 统计与耗时（如果变量存在就写入）---
perf = struct();
if exist('time_BE_lf','var'), perf.time_BE_lf = time_BE_lf; end
if exist('time_BE_hf','var'),      perf.time_BE_hf      = time_BE_hf;      end
if exist('time_rand_hf','var'),    perf.time_rand_hf    = time_rand_hf;    end
if exist('tic_global','var'),      perf.time_total      = toc(tic_global); end
run_info.perf = perf;


% --- 本次仿真产物统计 ---
counts = struct();
if exist('num_BE_tasks','var'),      counts.num_BE_tasks      = num_BE_tasks;      end
if exist('num_rand_tasks','var'),    counts.num_rand_tasks    = num_rand_tasks;    end
if exist('num_valid_samples','var'), counts.num_valid_samples = num_valid_samples; end
run_info.counts = counts;

% --- 可选：记录 MATLAB / toolbox 版本（排查复现实验很有用）---
try
    run_info.matlab = struct( ...
        'version', version, ...
        'release', version('-release') );
catch
end

% 写入 HDF5 根属性
meta_json = jsonencode(make_json_serializable(run_info));
h5writeatt(dataset_filename, '/', 'meta_json', meta_json);
% ==========================================================

fprintf('\n数据集已保存到: %s\n\n', dataset_filename);

%% 6) 性能总结
% =========================================================================
fprintf('=== 性能总结 ===\n');
fprintf('低保真采样耗时:                 %.2f 秒(每个馈电点共 %d 次仿真)\n', time_BE_lf,sim_calls);
fprintf('低保真最终保留(top-K)样本数:     %d\n', final_topK);

fprintf('BE 设计高保真仿真耗时:           %.2f 秒 (共 %d 次仿真)\n', time_BE_hf, num_BE_tasks);
fprintf('随机设计高保真仿真耗时:          %.2f 秒 (共 %d 次仿真)\n', time_rand_hf, num_rand_tasks);
fprintf('总计高保真仿真耗时:              %.2f 秒\n', time_BE_hf + time_rand_hf);
fprintf('总程序运行耗时:                  %.2f 秒\n', toc(tic_global));
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


function result = simulate_single_antenna_hf(designVector, designParams_HF, feed_rc)
    N = designParams_HF.pixelResolution_N;
    dv = designVector(:)';  % 行向量

    if numel(dv) ~= N^2
        error('设计向量长度不匹配：得到 %d，但期望 %d。', numel(dv), N^2);
    end

    pix_bits = dv;

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
    elseif isa(S, 'optim.options.BE')
        % options 对象也不好直接编码，转成 struct 再清洗
        S2 = make_json_serializable(struct(S));
    else
        S2 = S; % 数值/字符/字符串/逻辑/表等可直接编码
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
function [top_designs, top_costs] = sample_wideband_designs_for_feed( ...
    designParams_LF, feed_rc_fixed, K, cfg, randomFillFactorRange)
    N = designParams_LF.pixelResolution_N;
    % D = N^2;

    % --- 1) 初始化随机池 (init pool) ---
    pool = random_pixel_pool_diverse(cfg.init_pool_size, N, randomFillFactorRange);
    % 保证馈电像素为金属（虽然后面建模也会强制，但提前固定更好）
    lin = feed_rc_fixed(1) + (feed_rc_fixed(2)-1)*N; % row-major注意 reshape 方式
    pool(:, lin) = true;

    % --- 2) 计算初始 cost (Step I) ---
   [costs] = eval_cost_batch(pool, designParams_LF, feed_rc_fixed, cfg);

    % archive：用 map 去重（hash）
    [pool, costs] = dedup_pool(pool, costs);

    % --- 3) 进化循环：biased select -> mutate/crossover -> eval -> merge/keep ---
    for it = 1:cfg.num_iters

        % (Step II) 指数偏置选择父代
        parent_idx = biased_select_indices(costs, cfg.children_per_iter, cfg.beta);

        % (Step III) 生成子代（mutation / crossover）
        children = gen_children(pool, costs, parent_idx, cfg, N, feed_rc_fixed);

        % 评估子代 cost
        [child_costs] = eval_cost_batch(children, designParams_LF, feed_rc_fixed, cfg);
        % 合并
        pool2  = [pool; children];
        costs2 = [costs; child_costs];

        % 去重
        [pool2, costs2] = dedup_pool(pool2, costs2);

        % 保留：偏置保留或直接取最好 keep_size
        [pool, costs] = keep_archive(pool2, costs2, cfg.keep_size, cfg.beta);

        % 可选：打印一下当前最优、以及宽带达标比例
        if mod(it, 5) == 0 || it == cfg.num_iters
            fprintf('  iter %2d: best cost=%.3f, pool=%d\n', it, min(costs), size(pool,1));
        end
    end
    % --- 输出 top-K（按 cost 最小）---
    [~, ord] = sort(costs, 'ascend');
    ord = ord(1:min(K,numel(ord)));
    top_designs = pool(ord,:);
    top_costs   = costs(ord);
end
function costs = eval_cost_batch(pool, designParams_LF, feed_rc_fixed, cfg)
    B = size(pool,1);
    costs = zeros(B,1);

    % ROI 频段索引
    f = designParams_LF.freq_sweep;
    roi = (f >= cfg.roi_fmin_GHz*1e9) & (f <= cfg.roi_fmax_GHz*1e9);

    parfor i = 1:B
        costs(i) = eval_cost_single(pool(i,:), designParams_LF, feed_rc_fixed, cfg, roi);
    end
end

function cost = eval_cost_single(vec, designParams_LF, feed_rc_fixed, cfg, roi)
    N = designParams_LF.pixelResolution_N;

    % bit -> 矩阵
    M = reshape(vec, N, N);
    M(feed_rc_fixed(1), feed_rc_fixed(2)) = 1;

    idx = find(M);
    if isempty(idx)
        cost = 1e3; return;
    end

    rectCells = designParams_LF.pixelShapes(idx);
    patchShape = union_rectangles_batch(rectCells, 256);

    ant = create_antenna_model(patchShape, feed_rc_fixed, designParams_LF);

    try
        s = sparameters(ant, designParams_LF.freq_sweep);
        s11 = squeeze(s.Parameters(1,1,:));
        s11_db = 20*log10(abs(s11)+eps);

        s11_db_roi = s11_db(roi);
        f_roi = designParams_LF.freq_sweep(roi);

        [bw_hz, seg, nbands, extra_bw_hz] = longest_contiguous_bw(s11_db_roi, f_roi, cfg.thr_db);
        % --- 1) 主通带中心频率惩罚：希望中心在 10 GHz 附近---
        f0 = cfg.f0_GHz * 1e9;
        df = cfg.df_GHz * 1e9;

        if seg.valid
            f_center = 0.5 * (f_roi(seg.i1) + f_roi(seg.i2));   % 主通带中心频率
            pen_f0 = ((f_center - f0) / df)^2;                  % 平方惩罚（无量纲）
        else
            % 没有主通带：这里不给"中心惩罚"，只靠原来的 nbands==0 惩罚即可
            pen_f0 = 0;
        end
        roi_bw  = f_roi(end) - f_roi(1) + eps;
        bw_norm = bw_hz / roi_bw;

        % --- 2)多通带惩罚：段数 + 额外通带带宽
        pen_bands   = max(0, nbands - 1);
        pen_extraBW = extra_bw_hz / roi_bw;

        % --- 3) 深度项（限幅，防止单尖峰支配）---
        depth = min(s11_db_roi);
        depth_term = max(depth, cfg.depth_thr_db); % e.g. [-20 .. -inf] -> [-20 ..]
        % depth_term 越小越好（更负更好）

        % --- 4) 带宽奖励（饱和）---
        bw_reward = (1 - exp(-bw_norm/cfg.bw)); % 0.15 可调：15% 带宽就接近饱和

        % --- 5) 合成 cost（越小越好）---
        cost = depth_term*cfg.depth ...
             + cfg.w_multiband_count   * pen_bands ...
             + cfg.w_multiband_extraBW * pen_extraBW ...
             - (20*cfg.w_bw) * bw_reward...% 20 把奖励拉到 dB 量级
             + cfg.w_f0 * pen_f0;

        if nbands == 0
            cost = cost + 10; % 完全没通带的惩罚
        end 

        if isnan(cost) || isinf(cost), cost = 1e3; end
    catch
        cost = 1e3;
    end
end
function [bw_hz, seg, nbands, extra_bw_hz] = longest_contiguous_bw(s11_db, f_hz, thr_db)
    mask = (s11_db < thr_db);
    mask = mask | ([false; mask(1:end-1)] & [mask(2:end); false]);
    seg = struct('valid',false,'i1',1,'i2',0);
    bw_hz = 0;
    nbands = 0;
    extra_bw_hz = 0;

    if ~any(mask)
        return;
    end

    d = diff([false; mask(:); false]);
    starts = find(d==1);
    ends   = find(d==-1)-1;

    nbands = numel(starts);

    % 每段带宽
    bw_list = f_hz(ends) - f_hz(starts);

    % 最长段
    [bw_hz, k] = max(bw_list);
    seg.valid = true;
    seg.i1 = starts(k);
    seg.i2 = ends(k);

    % 主通带之外的匹配带宽总和
    extra_bw_hz = sum(bw_list) - bw_hz;
end


function idx = biased_select_indices(costs, M, alpha)
    c = costs(:);
    c = c - min(c);                % 平移，避免 exp 溢出
    p = exp(-alpha * c);
    p = p / (sum(p) + eps);
    idx = randsample(numel(c), M, true, p);
end

function children = gen_children(pool, costs, parent_idx, cfg, N, feed_rc_fixed)
    D = size(pool,2);
    M = numel(parent_idx);
    children = false(M, D);

    lin = feed_rc_fixed(1) + (feed_rc_fixed(2)-1)*N;

    for t = 1:M
        p1 = pool(parent_idx(t), :);

        % crossover
        if rand < cfg.pc && size(pool,1) >= 2
            p1_idx = parent_idx(t);

            if rand < cfg.pp2
                 % 70%：按 cost 偏置选
                max_retry = 5;
                p2_idx = p1_idx;
                for rr = 1:max_retry
                    p2_idx = biased_select_indices(costs, 1, cfg.beta);
                    if p2_idx ~= p1_idx
                        break;
                    end
                end
            else
                % 30%：完全随机选（保多样性）
                p2_idx = randi(size(pool,1));
            end

            p2 = pool(p2_idx, :);

            cp = randi([2, D-1]);
            child = [p1(1:cp), p2(cp+1:end)];
        else
            child = p1;
        end

        % mutation：翻转少量 bit
        flip = rand(1, D) < cfg.pm;
        child = xor(child, flip);

        % 强制馈电像素为金属
        child(lin) = true;

        children(t,:) = child;
    end
end
function [pool_u, costs_u] = dedup_pool(pool, costs)
    % 用 MD5/hash 去重（复用你 DataHash 也行）
    keys = cell(size(pool,1),1);
    for i = 1:size(pool,1)
        keys{i} = DataHash(uint8(pool(i,:)));
    end
    [~, ia, ic] = unique(keys, 'stable');

    % 对重复的，保留 cost 最小的那个
    best_cost = inf(numel(ia),1);
    best_row  = zeros(numel(ia),1);

    for i = 1:numel(ic)
        k = ic(i);
        if costs(i) < best_cost(k)
            best_cost(k) = costs(i);
            best_row(k)  = i;
        end
    end

    pool_u  = pool(best_row, :);
    costs_u = costs(best_row);
end

function [pool_k, costs_k] = keep_archive(pool, costs, keep_size, alpha_keep)
    % 两种策略都行：
    % 1) 直接按 cost 排序取前 keep_size（最稳定）
    % 2) 偏置抽样保留（更保多样性）

    if size(pool,1) <= keep_size
        pool_k = pool; costs_k = costs; return;
    end

    % 这里用"偏置抽样保留"
     c = costs(:) - min(costs);
    w = exp(-alpha_keep * c);      % 权重 w >= 0
    w = w + eps;                  % 防止出现 0 权重导致问题

    % ---- MATLAB旧版不支持: randsample(n,k,false,w) ----
    sel = weighted_sample_wo_replace(w, keep_size);

    pool_k = pool(sel,:);
    costs_k = costs(sel);

    % 可选：为了稳定性，再把其中最好的若干强制保留
    % [~,ord]=sort(costs,'ascend'); elite=ord(1:min(20,numel(ord)));
    % pool_k(1:numel(elite),:)=pool(elite,:); costs_k(1:numel(elite))=costs(elite);
end
% function [pool_k, costs_k] = keep_archive_diverse(pool, costs, keep_size, dmin)
%     if nargin<5, dmin=12; end
% 
%     % 先按 cost 由好到差排序
%     [~, ord] = sort(costs, 'ascend');
%     pool_s = pool(ord,:);
%     costs_s = costs(ord);
% 
%     sel = false(size(pool_s,1),1);
%     picked = [];
% 
%     for i = 1:size(pool_s,1)
%         if numel(picked) >= keep_size, break; end
%         if isempty(picked)
%             sel(i) = true;
%             picked = i;
%         else
%             % 计算与已选集合的最小海明距离
%             d = sum(xor(pool_s(i,:), pool_s(picked,:)), 2);
%             if min(d) >= dmin
%                 sel(i) = true;
%                 picked(end+1) = i; %#ok<AGROW>
%             end
%         end
%     end
% 
%     % 如果因为 dmin 太大导致不够 keep_size，就补充最优的
%     if numel(picked) < keep_size
%         remain = find(~sel);
%         need = keep_size - numel(picked);
%         add = remain(1:min(need, numel(remain)));
%         sel(add) = true;
%     end
% 
%     pool_k = pool_s(sel,:);
%     costs_k = costs_s(sel,:);
% end

% function pool = random_pixel_pool(B, N)
%     % 简单：每个像素 0/1 概率 0.5
%     pool = rand(B, N^2) < 0.6; % 0.6 只是例子，可调或随机化
% end
function pool = random_pixel_pool_diverse(B, N, fill_range)
    if nargin<3, fill_range=randomFillFactorRange ; end
    ff = linspace(fill_range(1), fill_range(2), B)';  % 每个样本一个填充率
    pool = rand(B, N^2) < ff;                         % 每行不同阈值
end

function sel = weighted_sample_wo_replace(w, k)
% 加权无放回抽样（Efraimidis-Spirakis 方法）
% 输入:
%   w: [n x 1] 权重，非负即可（越大越容易被选中）
%   k: 需要抽样数量，k<=n
% 输出:
%   sel: 被选中的索引（长度 k），无放回

    w = w(:);
    n = numel(w);
    k = min(k, n);

    % 权重必须非负，且不能全为0
    w(w < 0) = 0;
    if all(w == 0)
        sel = randperm(n, k);
        return;
    end

    % 生成随机key： key = -log(U) ./ w
    % w 越大，key 越小，越容易进入前 k
    U = rand(n,1);
    keys = -log(U) ./ (w + eps);

    [~, ord] = sort(keys, 'ascend');   % 取 key 最小的 k 个
    sel = ord(1:k);
end
