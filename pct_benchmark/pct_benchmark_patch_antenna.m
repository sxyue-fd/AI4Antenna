function pct_benchmark_patch_antenna()
% PCT_BENCHMARK_PATCH_ANTENNA
% 使用 MATLAB Antenna Toolbox 作为计算负载，测试Parallel Computing Toolbox（PCT）的并行加速效果。
%
% 主要特性:
% 1. 在脚本顶端提供集中的用户配置区。
% 2. 预先计算天线设计参数，避免在 parfor 中重复计算。
% 3. 天线形式采用无限大地板、空气介质的探针馈电贴片天线，尽可能降低网格数。
% 4. 测试完成后，统一生成并显示性能分析图表。

%% 0) User Configuration %%
% =========================================================================
% 在此区域修改所有测试参数

% --- 并行池配置 ---
useThreads = false;    % 池类型: false=多进程 (默认), true=多线程

% --- 负载配置 ---
tasksPerWorker   = 4;      % 每个 worker 分配的任务数，总任务数=cpu核数*tasksPerWorker     
centerFreq_GHz   = 2.4;   % 天线设计的中心频率 (GHz)
freqSpan_GHz     = 0.4;    % 扫频总带宽 (GHz)
numFreqPoints    = 21;     % 每个任务中计算的频点数量
meshLambdaFraction = 10;    % 网格剖分精度 (lambda / N), N 在此定义

% --- 要测试的并行度 ---
testWorkers = unique([1, 2.^(0:10)]); % 例如: 1,2,4,8,16...

% =========================================================================


%% 1) 初始化环境与设计参数
assert(license('test','Distrib_Computing_Toolbox')==1, ...
    '未检测到 Parallel Computing Toolbox 许可证。');
assert(license('test','Antenna_Toolbox')==1, ...
    '未检测到 Antenna Toolbox 许可证。');

clc; close all; rng(106);

% 调用独立函数，一次性计算好天线基准设计和仿真参数
designParams = design_antenna_parameters(centerFreq_GHz, freqSpan_GHz, ...
                                        numFreqPoints, meshLambdaFraction);
fprintf('天线基准参数设计完成。\n');
fprintf('  - 贴片长度: %.2f mm, 宽度: %.2f mm\n', designParams.L*1e3, designParams.W*1e3);
fprintf('  - 扫频范围: %.3f GHz 至 %.3f GHz (%d 个点)\n', ...
        min(designParams.freq_sweep)/1e9, max(designParams.freq_sweep)/1e9, numFreqPoints);
fprintf('  - 网格最大边长: %.2f mm\n\n', designParams.maxEdge*1e3);

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
taskSeeds = 1:numTasks;

fprintf('=== MATLAB PCT Benchmark (Traditional Patch Antenna) ===\n');
fprintf('Pool Type: %s | Workers=%d\n', ternary(useThreads,'Threads','Processes'), maxWorkers);
fprintf('Total Tasks=%d (≈ %d per worker)\n\n', numTasks, ceil(numTasks/maxWorkers));

% 预热 (JIT/Antenna Toolbox)
fprintf('正在预热并行池...\n');
parfor (k = 1:min(4, maxWorkers), min(4, maxWorkers))
    heavy_task_antenna(k, designParams, numTasks);
end
dummy = heavy_task_antenna(taskSeeds(1), designParams, numTasks); %#ok<NASGU>
fprintf('预热完成。\n\n');

%% 3) 串行基线
fprintf('Running serial baseline (for-loop)...\n');
tSerial = timeit(@() run_serial(taskSeeds, designParams, numTasks));
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
        t = timeit(@() run_parfor_P(taskSeeds, designParams, numTasks, P));
        times(i) = t;
    end
    
    speedup(i) = tSerial / times(i);
    eff(i)     = speedup(i) / P;
    fprintf('  P=%d -> Time: %.3f s | Speedup: %.2f | Efficiency: %.2f%%\n\n', ...
        P, times(i), speedup(i), 100*eff(i));
end

%% 5) 可视化 (主题无关的稳健版本)
fprintf('所有计算已完成，正在生成结果图表...\n');
fig = figure('Color','w','Position',[100 100 900 750]);

% --- 1. 定义统一的、不受主题影响的颜色和样式 ---
textColor = 'k';                        % 文本颜色: 黑色
lineColor_Main = [0, 114, 189] / 255;  % 主曲线颜色: MATLAB 经典蓝色
lineColor_Ideal = [217, 83, 25] / 255; % 理想曲线颜色: MATLAB 经典橙色
gridColor = [0.8 0.8 0.8];            % 网格线颜色: 浅灰色
lineWidth = 1.8;
markerSize = 6;
fontSize_axes = 10;
fontSize_title = 11;
fontSize_superTitle = 14;

% --- 2. 绘制第一个子图: 总耗时 ---
ax1 = subplot(3,1,1);
xs = string(testWorkers); xs(1) = "1 (serial)";
plot(1:numel(testWorkers), times, '-o', 'LineWidth', lineWidth, ...
    'MarkerSize', markerSize, 'Color', lineColor_Main);
grid on;

% --- 对坐标轴 ax1 进行详细设置 ---
ax1.XColor = textColor;
ax1.YColor = textColor;
ax1.GridColor = gridColor;
ax1.Box = 'on';
xticks(1:numel(xs)); 
xticklabels(xs);
ylabel('Time (s)', 'Color', textColor, 'FontSize', fontSize_axes); 
title('总耗时 vs. 并行度 P', 'Color', textColor, 'FontSize', fontSize_title);

% --- 3. 绘制第二个子图: 加速比 ---
ax2 = subplot(3,1,2);
hold on;
plot(1:numel(testWorkers), speedup, '-o', 'LineWidth', lineWidth, ...
    'MarkerSize', markerSize, 'Color', lineColor_Main);
plot(1:numel(testWorkers), testWorkers./testWorkers(1), '--', ...
    'LineWidth', lineWidth, 'Color', lineColor_Ideal);
hold off;
grid on;

% --- 对坐标轴 ax2 进行详细设置 ---
ax2.XColor = textColor;
ax2.YColor = textColor;
ax2.GridColor = gridColor;
ax2.Box = 'on';
xticks(1:numel(xs)); 
xticklabels(xs);
ylabel('Speedup', 'Color', textColor, 'FontSize', fontSize_axes); 
title('加速比 (含理想线)', 'Color', textColor, 'FontSize', fontSize_title);
lgd = legend('实测','理想', 'Location','northwest');
lgd.TextColor = textColor; % 设置图例文字颜色

% --- 4. 绘制第三个子图: 并行效率 ---
ax3 = subplot(3,1,3);
plot(1:numel(testWorkers), 100*eff, '-o', 'LineWidth', lineWidth, ...
    'MarkerSize', markerSize, 'Color', lineColor_Main);
grid on;

% --- 对坐标轴 ax3 进行详细设置 ---
ax3.XColor = textColor;
ax3.YColor = textColor;
ax3.GridColor = gridColor;
ax3.Box = 'on';
xticks(1:numel(xs)); 
xticklabels(xs);
ylabel('Efficiency (%)', 'Color', textColor, 'FontSize', fontSize_axes); 
ylim([0 110]);
title('并行效率 = Speedup / P * 100%', 'Color', textColor, 'FontSize', fontSize_title);

% --- 5. 设置总标题和最终处理 ---
sgtitle(sprintf('Antenna Toolbox Benchmark — %d Tasks, %.2f GHz, %d Points/Task', ...
    numTasks, centerFreq_GHz, numFreqPoints), ...
    'FontWeight', 'bold', 'Color', textColor, 'FontSize', fontSize_superTitle);

set(fig, 'ToolBar', 'none', 'Visible', 'on');
exportgraphics(fig, 'pct_benchmark_antenna.png', 'Resolution', 150);
fprintf('图表已保存：pct_benchmark_antenna.png\n\n');

%% 6) 结果表
T = table(testWorkers(:), times(:), speedup(:), eff(:)*100, ...
    'VariableNames', {'P','Time_s','Speedup','Efficiency_pct'});
disp('=== 结果表 ==='); disp(T);

end


%% ====== 辅助函数 ======

function run_serial(taskSeeds, designParams, numTasks)
    acc = 0;
    for k = 1:numel(taskSeeds)
        acc = acc + heavy_task_antenna(taskSeeds(k), designParams, numTasks);
    end
    assert(isfinite(acc));
end

function run_parfor_P(taskSeeds, designParams, numTasks, P)
    acc = 0;
    parfor (k = 1:numel(taskSeeds), P)
        acc = acc + heavy_task_antenna(taskSeeds(k), designParams, numTasks);
    end
    assert(isfinite(acc));
end

% -------------------------------------------------------------------------
function params = design_antenna_parameters(freq_ghz, span_ghz, num_points, mesh_lambda_frac)
    % 使用经验公式计算天线设计参数和仿真设置
    f_center = freq_ghz * 1e9;
    c = physconst('LightSpeed');
    lambda0 = c / f_center;

    % 经验公式计算贴片尺寸 (空气介质, er=1)
    params.L = lambda0 / 2;
    params.W = params.L * 1.5;
    params.h = lambda0 / 50;
    
    % 创建天线对象模板
    d_air = dielectric('Air');
    d_air.Thickness = params.h;
    params.baseAntenna = patchMicrostrip( ...
        'Length', params.L, ...
        'Width', params.W, ...
        'Height', params.h, ...
        'Substrate', d_air, ...
        'GroundPlaneLength', Inf, ...
        'GroundPlaneWidth', Inf, ...
        'FeedOffset', [params.L/4, 0]); % 基准馈电点

    % 计算仿真参数
    f_start = (freq_ghz - span_ghz/2) * 1e9;
    f_stop = (freq_ghz + span_ghz/2) * 1e9;
    params.freq_sweep = linspace(f_start, f_stop, num_points);
    params.maxEdge = lambda0 / mesh_lambda_frac;
end

% -------------------------------------------------------------------------
function out = heavy_task_antenna(seed, designParams, numTasks)
    % 并行计算的实际负载任务
    % 复制基础天线对象
    ant = designParams.baseAntenna;
    
    % 使用 seed 对每个任务的天线参数进行微小扰动，确保计算的独立性
    offset_factor = (seed - numTasks/2) / numTasks; % from -0.5 to 0.5
    ant.FeedOffset(1) = ant.Length/4 * (1 + offset_factor*0.05);

    % 执行计算 (分为两步)
    % 1. 网格剖分
    m = mesh(ant, 'MaxEdgeLength', designParams.maxEdge);
    
    % 2. S参数计算
    s = sparameters(ant, designParams.freq_sweep);

    % 返回一个标量结果用于累加
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