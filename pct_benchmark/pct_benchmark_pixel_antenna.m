function pct_benchmark_onepool_pixel_antenna()
% PCT_BENCHMARK_ONEPOOL_PIXEL_ANTENNA
% 使用随机生成的像素化贴片天线作为计算负载，测试并行加速效果。
%
% 主要特性:
% 1. 在脚本顶端提供集中的用户配置区。
% 2. 预先计算天线设计模板和像素“零件库”，提高 parfor 效率。
% 3. heavy_task 内部使用 rng('shuffle') 实现真正的随机天线生成。
% 4. 采用 pcbStack 建模，并实现像素重叠和强制馈电点机制。
% 5. 计算过程中不显示任何图形，只在命令行打印进度。

%% 0) User Configuration %%
% =========================================================================
% 在此区域修改所有测试参数

% --- 并行池配置 ---
useThreads = false;    % 池类型: false=多进程 (默认), true=多线程

% --- 负载配置 ---
tasksPerWorker      = 4;      % 每个 worker 分配的任务数 (建议 4~16)
centerFreq_GHz      = 2.45;   % 天线设计的中心频率 (GHz)
freqSpan_GHz        = 0.4;    % 扫频总带宽 (GHz)
numFreqPoints       = 21;     % 每个任务中计算的频点数量
meshLambdaFraction  = 10;     % 网格剖分精度 (lambda / N), N 在此定义

% --- 像素天线特有配置 ---
pixelResolution_N   = 16;     % 贴片分辨率 N x N
pixelFillFactor     = 0.6;    % 像素填充率 (0到1)，控制金属像素密度
overlap_mm          = 0.8;    % 像素间重叠的宽度 (mm)，确保电连接，这个值不应太小以免网格剖分错误

% --- 要测试的并行度 ---
testWorkers = unique([1, 2.^(0:10)]); % 例如: 1,2,4,8,16...

% =========================================================================


%% 1) 初始化环境与设计参数
assert(license('test','Distrib_Computing_Toolbox')==1, ...
    '未检测到 Parallel Computing Toolbox 许可证。');
assert(license('test','Antenna_Toolbox')==1, ...
    '未检测到 Antenna Toolbox 许可证。');
assert(license('test','RF_PCB_Toolbox')==1, ...
    '需要 RF PCB Toolbox 来支持 pcbStack。');

clc; close all; rng(1); % 主线程 rng 用于可复现的基准测试流程

% 调用独立函数，一次性计算好天线基准设计和仿真参数
designParams = design_pixel_antenna_parameters(centerFreq_GHz, freqSpan_GHz, numFreqPoints, meshLambdaFraction, pixelResolution_N, overlap_mm);
fprintf('像素天线基准参数设计完成。\n');
fprintf('  - 整体贴片区域: %.2f mm x %.2f mm\n', designParams.L*1e3, designParams.W*1e3);
fprintf('  - 电路板尺寸: %.2f mm x %.2f mm\n', designParams.ground.Length*1e3, designParams.ground.Width*1e3);
fprintf('  - 像素分辨率: %d x %d\n', pixelResolution_N, pixelResolution_N);
fprintf('  - 馈电点: (%.2f, %.2f) mm\n', designParams.finalFeedLocation(1)*1e3, designParams.finalFeedLocation(2)*1e3);
fprintf('  - 扫频范围: %.3f GHz 至 %.3f GHz (%d 个点)\n\n', min(designParams.freq_sweep)/1e9, max(designParams.freq_sweep)/1e9, numFreqPoints);

%% 2) 启动并行池
delete(gcp('nocreate'));
cleanupObj = onCleanup(@() finalize_env());

if useThreads
    pool = parpool("Threads", feature('numcores'), "IdleTimeout", 120);
    maxNumCompThreads(1);
else
    pc = parcluster('local');
    try pc.JobStorageLocation = fullfile(tempdir, 'pct_jobs'); end
    pool = parpool(pc, pc.NumWorkers, "IdleTimeout", 120);
    pctRunOnAll maxNumCompThreads(1)
end

maxWorkers = pool.NumWorkers;
testWorkers = testWorkers(testWorkers <= maxWorkers);
if isempty(testWorkers) || testWorkers(end) ~= maxWorkers
    testWorkers(end+1) = maxWorkers;
end
testWorkers = unique(testWorkers);

numTasks = max(tasksPerWorker * maxWorkers, 32);

fprintf('=== MATLAB PCT Benchmark (Pixel Antenna) ===\n');
fprintf('Pool Type: %s | Workers=%d\n', ternary(useThreads,'Threads','Processes'), maxWorkers);
fprintf('Total Tasks=%d (≈ %d per worker)\n\n', numTasks, ceil(numTasks/maxWorkers));

% 预热 (JIT/Antenna Toolbox)
fprintf('正在预热并行池...\n');
parfor (k = 1:min(4, maxWorkers), min(4, maxWorkers))
    heavy_task_pixel_antenna(designParams, pixelFillFactor);
end
dummy = heavy_task_pixel_antenna(designParams, pixelFillFactor); %#ok<NASGU>
fprintf('预热完成。\n\n');

%% 3) 串行基线
fprintf('Running serial baseline (for-loop)...\n');
tSerial = timeit(@() run_serial(numTasks, designParams, pixelFillFactor));
fprintf('  Serial time: %.3f s\n\n', tSerial);

%% 4) 并行测试
times = nan(size(testWorkers));
speedup = nan(size(testWorkers));
eff     = nan(size(testWorkers));

for i = 1:numel(testWorkers)
    P = testWorkers(i);
    if P == 1
        times(i) = tSerial;
    else
        fprintf('Running parallel with P=%d (parfor)...\n', P);
        t = timeit(@() run_parfor_P(numTasks, designParams, pixelFillFactor, P));
        times(i) = t;
    end
    
    speedup(i) = tSerial / times(i);
    eff(i)     = speedup(i) / P;
    fprintf('  P=%d -> Time: %.3f s | Speedup: %.2f | Efficiency: %.2f%%\n\n', ...
        P, times(i), speedup(i), 100*eff(i));
end

%% 5) 可视化 (全部计算完成后才显示)
fprintf('所有计算已完成，正在生成结果图表...\n');
fig = figure('Color','w','Position',[100 100 1000 650], 'Visible', 'off');

% ... (绘图部分代码与传统贴片版本完全相同) ...
subplot(3,1,1); xs = string(testWorkers); xs(1) = "1 (serial)"; plot(1:numel(testWorkers), times,'-o','LineWidth',1.8, 'MarkerSize',6); grid on; xticks(1:numel(xs)); xticklabels(xs); ylabel('Time (s)'); title('总耗时 vs. 并行度 P');
subplot(3,1,2); plot(1:numel(testWorkers), speedup,'-o','LineWidth',1.8, 'MarkerSize',6); hold on; plot(1:numel(testWorkers), testWorkers./testWorkers(1), '--','LineWidth',1.2); grid on; xticks(1:numel(xs)); xticklabels(xs); ylabel('Speedup'); title('加速比（含理想线）'); legend('实测','理想', 'Location','northwest');
subplot(3,1,3); plot(1:numel(testWorkers), 100*eff,'-o','LineWidth',1.8, 'MarkerSize',6); grid on; xticks(1:numel(xs)); xticklabels(xs); ylabel('Efficiency (%)'); ylim([0 110]); title('并行效率 = Speedup / P * 100%');
sgtitle(sprintf('Pixel Antenna Benchmark — %d Tasks, %dx%d grid, %.2f GHz', ...
    numTasks, pixelResolution_N, pixelResolution_N, centerFreq_GHz), 'FontWeight','bold');
set(fig, 'ToolBar', 'none', 'Visible', 'on'); try axtoolbar(gca, 'Visible', 'off'); catch; end

exportgraphics(fig, 'pct_benchmark_pixel_antenna.png', 'Resolution', 150);
fprintf('图表已保存：pct_benchmark_pixel_antenna.png\n\n');

%% 6) 结果表
T = table(testWorkers(:), times(:), speedup(:), eff(:)*100, ...
    'VariableNames', {'P','Time_s','Speedup','Efficiency_pct'});
disp('=== 结果表 ==='); disp(T);

end


%% ====== 辅助函数 ======

function run_serial(numTasks, designParams, fillFactor)
    acc = 0;
    for k = 1:numTasks
        acc = acc + heavy_task_pixel_antenna(designParams, fillFactor);
    end
    assert(isfinite(acc));
end

function run_parfor_P(numTasks, designParams, fillFactor, P)
    acc = 0;
    parfor (k = 1:numTasks, P)
        acc = acc + heavy_task_pixel_antenna(designParams, fillFactor);
    end
    assert(isfinite(acc));
end

% -------------------------------------------------------------------------
function params = design_pixel_antenna_parameters(freq_ghz, span_ghz, num_points, ...
                                                  mesh_lambda_frac, N, overlap_mm)
    % 预计算像素天线设计所需的所有固定参数和“零件库”
    f_center = freq_ghz * 1e9;
    c = physconst('LightSpeed');
    lambda0 = c / f_center;
    overlap = overlap_mm / 1000;

    % 1. 整体尺寸和基板
    params.L = lambda0 / 2;
    params.W = params.L * 1.5;
    params.h = lambda0 / 50;
    params.substrate = dielectric('Air');
    params.substrate.Thickness = params.h;

    extension = 12 * params.h; 
    board_L = params.L + extension;
    board_W = params.W + extension;
    params.ground = antenna.Rectangle('Length', board_L, 'Width', board_W, 'Center', [0 0]);

    % 2. 像素尺寸
    pixel_L = params.L / N;
    pixel_W = params.W / N;
    
    % 3. 预生成所有像素的 Shape 对象 (“零件库”)
    params.pixelShapes = cell(N, N);
    startX = -params.L/2 + pixel_L/2;
    startY = -params.W/2 + pixel_W/2;
    for r = 1:N
        for c = 1:N
            centerX = startX + (c-1)*pixel_L;
            centerY = startY + (r-1)*pixel_W;
            % “增肥”像素以实现重叠
            params.pixelShapes{r,c} = antenna.Rectangle('Length', pixel_L + overlap, ...
                'Width', pixel_W + overlap, 'Center', [centerX, centerY]);
        end
    end
    
    % 4. 馈电点位置
    % 步骤 1: 先计算出目标像素的索引
    initialFeedLocation = [params.L/4, 0];
    feed_c_idx = floor((initialFeedLocation(1) - (-params.L/2)) / pixel_L) + 1;
    feed_r_idx = floor((initialFeedLocation(2) - (-params.W/2)) / pixel_W) + 1;
    params.feedPixelIdx = [max(1, min(N, feed_r_idx)), max(1, min(N, feed_c_idx))];

    % 步骤 2: 获取目标像素的中心坐标作为最终馈电位置
    targetPixelShape = params.pixelShapes{params.feedPixelIdx(1), params.feedPixelIdx(2)};
    params.finalFeedLocation = targetPixelShape.Center;
    
    params.feedDiameter = min(pixel_L, pixel_W) / 10;
    

    % 5. 仿真参数
    f_start = (freq_ghz - span_ghz/2) * 1e9;
    f_stop = (freq_ghz + span_ghz/2) * 1e9;
    params.freq_sweep = linspace(f_start, f_stop, num_points);
    params.maxEdge = lambda0 / mesh_lambda_frac;
end

% -------------------------------------------------------------------------
function out = heavy_task_pixel_antenna(designParams, fillFactor)
    % 并行计算的实际负载任务：随机生成并仿真一个像素天线
    
    % 1. 确保每个 worker 的随机性
    rng('shuffle');
    
    % 2. 随机生成像素矩阵 (designVector)
    N = size(designParams.pixelShapes, 1);
    designVector = rand(N, N) < fillFactor;
    
    % 3. 强制放置馈电点像素
    designVector(designParams.feedPixelIdx(1), designParams.feedPixelIdx(2)) = 1;

    % 4. 根据 designVector 组合像素，构建贴片层
    pixel_indices = find(designVector);
    if isempty(pixel_indices)
        % 如果填充率为0且馈电点不在，创建一个最小天线避免错误
        patchShape = designParams.pixelShapes{designParams.feedPixelIdx(1), designParams.feedPixelIdx(2)};
    else
        % 用第一个金属像素作为基础
        patchShape = designParams.pixelShapes{pixel_indices(1)};
        % 将其他金属像素合并进去
        for i = 2:length(pixel_indices)
            patchShape = patchShape + designParams.pixelShapes{pixel_indices(i)};
        end
    end

    % 5. 使用 pcbStack 创建完整天线
    ant = pcbStack(); % 先创建一个空对象
    ant.BoardShape = designParams.ground;
    ant.BoardThickness = designParams.h;
    % 在设置 Layers 之前设置 FeedDiameter
    ant.FeedDiameter = designParams.feedDiameter; 
    ant.Layers = {patchShape, designParams.substrate, designParams.ground};
    ant.FeedLocations = [designParams.feedLocation, 1, 3]; % 最后设置 FeedLocations
    % 6. 执行计算
    mesh(ant, 'MaxEdgeLength', designParams.maxEdge);
    s = sparameters(ant, designParams.freq_sweep);

    % 7. 返回标量结果
    s11_db = 20*log10(abs(squeeze(s.Parameters(1,1,:))));
    out = sum(s11_db);
end

% -------------------------------------------------------------------------
function s = ternary(cond, a, b)
    if cond, s = a; else, s = b; end
end

function finalize_env()
    try delete(gcp('nocreate')); end
    try maxNumCompThreads('automatic'); end
    fprintf('并行池已关闭，环境已清理。\n');
end