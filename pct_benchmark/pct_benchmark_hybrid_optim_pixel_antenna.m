function pct_benchmark_hybrid_optim_pixel_antenna()
% PCT_BENCHMARK_PIXEL_ANTENNA
% 使用随机生成的像素贴片天线作为计算负载，测试并寻找 (进程数 x 每进程线程数) 
% 的最优组合，以最小化总计算时间。
%
% 主要特性:
% 1. 负载为动态生成的像素天线，包含建模、剖分和仿真全流程。
% 2. 确保每次生成的天线都不同 (基于 parfor 循环变量的随机种子)。
% 3. 实现了像素重叠、馈电点对齐和强制馈电像素等高级功能。
% 4. 自动检测CPU核心数作为参考。
% 5. 结果用条形图展示，直观对比不同组合的性能。

%% 0) User Configuration %%
% =========================================================================
% 在此区域修改所有测试参数

% --- 定义要测试的 [P, N] 组合 ---
% P = 进程数 (Workers), N = 每个进程的计算线程数 (Threads per Worker)
testCases = { ...
    [32, 1], ...  % 纯进程并行
    [32, 2], ...
    [16, 2], ...
    [16, 4]
};

% --- 像素天线负载配置 ---
pixelResolution_N  = 16;      % 贴片分辨率 (N x N)
pixelFillFactor    = 0.6;     % 金属像素填充率 (0 to 1)
overlap_mm         = 0.8;     % 像素间重叠距离 (mm)，这个值不应太小以免网格剖分错误

% --- 仿真参数配置 ---
tasksPerWorker_total = 8;     % 为最大worker数时，每个worker的目标任务数
centerFreq_GHz     = 2.45;
freqSpan_GHz       = 0.4;
numFreqPoints      = 21;
meshLambdaFraction = 20;

% =========================================================================


%% 1) 初始化环境与设计参数
assert(license('test','Distrib_Computing_Toolbox')==1, '未检测到 Parallel Computing Toolbox 许可证。');
assert(license('test','Antenna_Toolbox')==1, '未检测到 Antenna Toolbox 许可证。');

clc; close all;

% 获取CPU信息
maxNumPhysicalCores = feature('numcores');
try
    maxNumLogicalCores = parcluster('local').NumWorkers;
catch
    maxNumLogicalCores = maxNumPhysicalCores;
end
fprintf('检测到CPU信息:\n - 物理核心数: %d\n - 逻辑核心数: %d\n\n', ...
        maxNumPhysicalCores, maxNumLogicalCores);

% 计算天线共享基础参数
designParams = design_antenna_parameters(centerFreq_GHz, freqSpan_GHz, ...
                                          numFreqPoints, meshLambdaFraction, ...
                                          pixelResolution_N, overlap_mm);
fprintf('天线基准参数设计完成。\n\n');

%% 2) 启动并行池
max_P_to_test = 0;
for i = 1:numel(testCases)
    max_P_to_test = max(max_P_to_test, testCases{i}(1));
end
max_P_to_test = min(max_P_to_test, maxNumLogicalCores);

delete(gcp('nocreate'));
cleanupObj = onCleanup(@() finalize_env());

fprintf('正在以 %d 个Worker启动进程池...\n', max_P_to_test);
pool = parpool(max_P_to_test);
fprintf('并行池已启动。\n\n');

numTasks = max(tasksPerWorker_total * max_P_to_test, 32);
taskSeeds = 1:numTasks;

fprintf('=== MATLAB 混合并行性能测试 (像素天线负载) ===\n');
fprintf('总任务数 = %d\n\n', numTasks);

%% 3) 预热 (JIT/Antenna Toolbox)
fprintf('正在预热并行池 (使用 P=%d, N=1)...\n', max_P_to_test);
pctRunOnAll maxNumCompThreads(1);
parfor (k = 1:min(8, numTasks), max_P_to_test)
    heavy_task_antenna(k, designParams, pixelFillFactor);
end
fprintf('预热完成。\n\n');


%% 4) 循环测试不同组合
numCases = numel(testCases);
times = nan(1, numCases);
labels = cell(1, numCases);

for i = 1:numCases
    P = testCases{i}(1);
    N = testCases{i}(2);
    labels{i} = sprintf('%d P x %d T', P, N);
    
    if P > pool.NumWorkers
        fprintf('跳过测试: %s (需要 %d workers, 但池中只有 %d)\n\n', ...
                labels{i}, P, pool.NumWorkers);
        times(i) = NaN;
        continue;
    end
    
    fprintf('正在测试组合: %s (总计算线程 = %d)\n', labels{i}, P*N);
    
    command_to_run = sprintf('maxNumCompThreads(%d)', N);
    pctRunOnAll(command_to_run);
    
    t = timeit(@() run_parfor_P(taskSeeds, designParams, pixelFillFactor, P));
    times(i) = t;
    
    fprintf('  完成 -> 总耗时: %.3f s\n\n', t);
end

%% 5) 可视化与结果总结
fprintf('所有测试已完成，正在生成结果图表...\n');
fig = figure('Color','w', 'Position',[100 100 800 550]);
b = bar(times, 'FaceColor', [0, 114, 189] / 255);
grid on;
ax = gca; 
ax.GridColor = [0.8 0.8 0.8]; ax.GridAlpha = 0.7; ax.Box = 'on';
set(gca, 'xtick', 1:numel(labels), 'xticklabel', labels, 'FontSize', 10);
ylabel('总耗时 (秒)', 'FontSize', 11);
xlabel('并行组合 (进程数 P x 每进程线程数 T)', 'FontSize', 11);
title(sprintf('不同并行组合性能对比 (%d 个随机像素天线任务)', numTasks), ...
      'FontWeight', 'bold', 'FontSize', 13);
text(1:numel(labels), times, arrayfun(@(t) sprintf('%.2f s', t), times, 'UniformOutput', false), ...
    'HorizontalAlignment','center', 'VerticalAlignment','bottom', ...
    'FontSize', 10, 'Color', 'k', 'FontWeight', 'bold', 'Margin', 8);
set(fig, 'ToolBar', 'none');
exportgraphics(fig, 'pct_benchmark_pixel_antenna.png', 'Resolution', 150);
fprintf('图表已保存：pct_benchmark_pixel_antenna.png\n\n');

%% 6) 寻找最优解并显示
T = table((1:numCases)', labels(:), times(:), ...
    'VariableNames', {'Case_ID','Combination','Time_s'});
disp('=== 结果表 ===');
disp(T);

[minTime, minIdx] = min(times);
if ~isnan(minTime)
    bestConfig = labels{minIdx};
    fprintf('\n=== 最优组合 ===\n');
    fprintf('组合: %s\n', bestConfig);
    fprintf('最短时间: %.3f s\n', minTime);
else
    fprintf('\n未能完成任何有效测试。\n');
end

end


%% ====== 辅助函数 ======

function run_parfor_P(taskSeeds, designParams, pixelFillFactor, P)
    results = zeros(1, numel(taskSeeds));
    parfor (k = 1:numel(taskSeeds), P)
        results(k) = heavy_task_antenna(taskSeeds(k), designParams, pixelFillFactor);
    end
    
    % 在 parfor 结束后，可以进行如下分析
    total_tasks = numel(results);
    failed_tasks = sum(isnan(results)); % isnan会返回一个逻辑数组，sum会统计true(即NaN)的数量
    valid_tasks = total_tasks - failed_tasks;
    
    fprintf('  -> 分析: 总任务数 %d, 成功 %d, 失败 %d\n', ...
            total_tasks, valid_tasks, failed_tasks);
            
    % 确保至少有一个任务成功，否则断言失败
    assert(any(isfinite(results)));
end


function params = design_antenna_parameters(freq_ghz, span_ghz, num_points, ...
                                           mesh_lambda_frac, N, overlap_mm)
    % 此函数计算所有与随机性无关的、可被所有 worker 共享的参数
    f_center = freq_ghz * 1e9;
    c = physconst('LightSpeed');
    lambda0 = c / f_center;
    
    % 1. 整体尺寸和基板
    params.L = lambda0 / 2;
    params.W = params.L * 1.5;
    params.h = lambda0 / 50;
    params.substrateMaterial = dielectric('Air');
    extension = 12 * params.h;
    board_L = params.L + extension;
    board_W = params.W + extension;
    params.ground = antenna.Rectangle('Length', board_L, 'Width', board_W, 'Center', [0 0]);

    % 2. 像素相关参数
    params.pixelResolution_N = N;
    params.overlap = overlap_mm / 1000;
    pixel_L = params.L / N;
    pixel_W = params.W / N;
    
    % 3. 预先生成所有可能的像素几何体
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
    
    % 4. 计算固定的馈电点位置和索引
    initialFeedLocation = [params.L/4, 0];
    feed_c_idx = floor((initialFeedLocation(1) - (-params.L/2)) / pixel_L) + 1;
    feed_r_idx = floor((initialFeedLocation(2) - (-params.W/2)) / pixel_W) + 1;
    params.feedPixelIdx = [max(1, min(N, feed_r_idx)), max(1, min(N, feed_c_idx))];
    
    targetPixelShape = params.pixelShapes{params.feedPixelIdx(1), params.feedPixelIdx(2)};
    params.finalFeedLocation = targetPixelShape.Center; % 对齐到像素中心
    params.feedDiameter = min(pixel_L, pixel_W) / 10;
    
    % 5. 仿真参数
    f_start = (freq_ghz - span_ghz/2) * 1e9;
    f_stop = (freq_ghz + span_ghz/2) * 1e9;
    params.freq_sweep = linspace(f_start, f_stop, num_points);
    params.maxEdge = lambda0 / mesh_lambda_frac;
end


function out = heavy_task_antenna(seed, designParams, pixelFillFactor)
    % 此函数是每个 worker 执行的核心计算任务
    % 它为每个 'seed' 生成一个独一无二的随机天线并进行仿真
    
    % 1. 设置此任务的随机种子
    rng(seed);
    N = designParams.pixelResolution_N;
    
    % 2. 生成随机金属像素矩阵
    designVector = rand(N, N) < pixelFillFactor;
    
    % 3. 强制在馈电点位置放置一个像素
    designVector(designParams.feedPixelIdx(1), designParams.feedPixelIdx(2)) = 1;
    
    % 4. 【建模】通过布尔运算组合像素，这是计算密集型步骤
    pixel_indices = find(designVector);
    if isempty(pixel_indices) % 理论上不会发生，但作为保护
        patchShape = designParams.pixelShapes{designParams.feedPixelIdx(1), designParams.feedPixelIdx(2)};
    else
        patchShape = designParams.pixelShapes{pixel_indices(1)};
        for i = 2:length(pixel_indices)
            patchShape = patchShape + designParams.pixelShapes{pixel_indices(i)};
        end
    end
    
    % 5. 创建 pcbStack 天线对象
    ant = pcbStack();
    ant.BoardShape = designParams.ground;
    ant.BoardThickness = designParams.h;
    ant.Layers = {patchShape, designParams.substrateMaterial, designParams.ground};
    ant.FeedDiameter = designParams.feedDiameter;
    ant.FeedLocations = [designParams.finalFeedLocation, 1, 3];
    
    % 创建try-catch块以防不合理的天线几何导致并行池崩溃
    try
        % 6. 【剖分】进行网格剖分
        m = mesh(ant, 'MaxEdgeLength', designParams.maxEdge);
        
        % 7. 【仿真】计算 S 参数
        s = sparameters(ant, designParams.freq_sweep);
        
        % 8. 返回一个标量结果
        s11_db = 20*log10(abs(squeeze(s.Parameters(1,1,:))));
        out = sum(s11_db); % 使用一个简单的聚合值作为返回值
    catch ME
        % 如果剖分或仿真失败，打印错误信息并返回一个无效值 (如 NaN)
        % 这样 parfor 不会崩溃，而是会继续执行
        fprintf('任务 Seed %d 失败: %s\n', seed, ME.message);
        out = NaN;
    end
end

function finalize_env()
    try delete(gcp('nocreate')); end
    try maxNumCompThreads('automatic'); end % 恢复默认设置
    fprintf('并行池已关闭，环境已清理。\n');
end