function pct_benchmark_hybrid_optim_patch_antenna()
% PCT_OPTIMIZER_HYBRID_ANTENNA
% 使用 MATLAB Antenna Toolbox 作为计算负载，寻找 (进程数 x 每进程线程数) 的
% 最优组合，以最小化总计算时间。
%
% 主要特性:
% 1. 在用户配置区定义要测试的 [P, N] 组合。
% 2. 一次性启动并行池，循环测试不同组合，效率高。
% 3. 自动检测CPU核心数作为参考。
% 4. 结果用条形图展示，直观对比不同组合的性能。

%% 0) User Configuration %%
% =========================================================================
% 在此区域修改所有测试参数

% --- 定义要测试的 [P, N] 组合 ---
% P = 进程数 (Workers), N = 每个进程的计算线程数 (Threads per Worker)
% 这是一个示例，请根据您的CPU进行修改。
% 对于32核64线程CPU，以下是一些推荐的测试组合：[32,1],[32,2],[16,2],[16,4]
testCases = { ...
    [32, 1], ...  % 纯进程并行，每个进程1线程 (基准)
    [32, 2], ...  % 充分利用超线程
    [16, 2], ...  % 一半进程，每个进程2线程
    [16, 4], ...  % 进一步探索线程效益
};

% --- 负载配置 (与之前相同) ---
tasksPerWorker_total = 8; % 为最大worker数时，每个worker的目标任务数
centerFreq_GHz     = 2.4;
freqSpan_GHz       = 0.4;
numFreqPoints      = 21;
meshLambdaFraction = 10;

% =========================================================================


%% 1) 初始化环境与设计参数
assert(license('test','Distrib_Computing_Toolbox')==1, '未检测到 Parallel Computing Toolbox 许可证。');
assert(license('test','Antenna_Toolbox')==1, '未检测到 Antenna Toolbox 许可证。');

clc; close all; rng(1);

% 获取CPU信息
maxNumPhysicalCores = feature('numcores');
try
    % NumWorkers在local profile中通常对应逻辑核心数
    maxNumLogicalCores = parcluster('local').NumWorkers;
catch
    maxNumLogicalCores = maxNumPhysicalCores; % Fallback
end
fprintf('检测到CPU信息:\n - 物理核心数: %d\n - 逻辑核心数: %d\n\n', ...
        maxNumPhysicalCores, maxNumLogicalCores);

% 计算天线参数
designParams = design_antenna_parameters(centerFreq_GHz, freqSpan_GHz, ...
                                        numFreqPoints, meshLambdaFraction);
fprintf('天线基准参数设计完成。\n\n');

%% 2) 启动并行池
% 基于所有测试中所需的最大进程数来启动池
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

% 根据最大worker数计算总任务量
numTasks = max(tasksPerWorker_total * max_P_to_test, 32);
taskSeeds = 1:numTasks;

fprintf('=== MATLAB Hybrid Parallelism Optimizer (Antenna) ===\n');
fprintf('总任务数 = %d\n\n', numTasks);

%% 3) 预热 (JIT/Antenna Toolbox)
fprintf('正在预热并行池 (使用 P=%d, N=1)...\n', max_P_to_test);
pctRunOnAll maxNumCompThreads(1);
parfor (k = 1:min(8, numTasks), max_P_to_test)
    heavy_task_antenna(k, designParams, numTasks);
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
    
    % **核心步骤**: 为所有worker设置内部计算线程数
    command_to_run = sprintf('maxNumCompThreads(%d)', N);
    pctRunOnAll(command_to_run);
    
    % 使用timeit进行精确计时, 并限制只使用P个worker
    t = timeit(@() run_parfor_P(taskSeeds, designParams, numTasks, P));
    times(i) = t;
    
    fprintf('  完成 -> 总耗时: %.3f s\n\n', t);
end

%% 5) 可视化与结果总结
fprintf('所有测试已完成，正在生成结果图表...\n');
fig = figure('Color','w', 'Position',[100 100 800 550]);

% --- 核心修改部分 ---
% 定义颜色
barColor = [0, 114, 189] / 255; % MATLAB 经典的蓝色
textColor = 'k'; % 黑色文本
barTextColor = 'w'; % 条形图上的白色文本

% 绘制条形图
b = bar(times, 'FaceColor', barColor);
grid on;

% 获取当前坐标轴句柄(handle)以进行详细设置
ax = gca; 
ax.XColor = textColor; % X轴刻度和标签颜色
ax.YColor = textColor; % Y轴刻度和标签颜色
ax.GridColor = [0.8 0.8 0.8]; % 网格线颜色
ax.GridAlpha = 0.7; % 网格线透明度
ax.Box = 'on'; % 显示完整的坐标轴框

% 设置标签和标题，并明确指定颜色
set(gca, 'xtick', 1:numel(labels), 'xticklabel', labels, 'FontSize', 10);
ylabel('总耗时 (秒)', 'Color', textColor, 'FontSize', 11);
xlabel('并行组合 (进程数 P x 每进程线程数 T)', 'Color', textColor, 'FontSize', 11);
title(sprintf('不同并行组合性能对比 (%d 个天线仿真任务)', numTasks), ...
      'FontWeight', 'bold', 'Color', textColor, 'FontSize', 13);

% 在每个条形图上标注时间，并明确指定颜色为白色
text(1:numel(labels), times, arrayfun(@(t) sprintf('%.2f s', t), times, 'UniformOutput', false), ...
    'HorizontalAlignment','center', ...
    'VerticalAlignment','bottom', ...
    'FontSize', 10, ...
    'Color', barTextColor, ... % 关键修改：黑色改为白色
    'FontWeight', 'bold', ...
    'Margin', 8); % 增加一点边距，避免压线

set(fig, 'ToolBar', 'none');
exportgraphics(fig, 'pct_optimizer_hybrid_antenna_fixed.png', 'Resolution', 150);
fprintf('图表已保存：pct_optimizer_hybrid_antenna_fixed.png\n\n');

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

function run_parfor_P(taskSeeds, designParams, numTasks, P)
    acc = 0;
    parfor (k = 1:numel(taskSeeds), P)
        acc = acc + heavy_task_antenna(taskSeeds(k), designParams, numTasks);
    end
    assert(isfinite(acc));
end

function params = design_antenna_parameters(freq_ghz, span_ghz, num_points, mesh_lambda_frac)
    f_center = freq_ghz * 1e9;
    c = physconst('LightSpeed');
    lambda0 = c / f_center;
    params.L = lambda0 / 2;
    params.W = params.L * 1.5;
    params.h = lambda0 / 50;
    d_air = dielectric('Air');
    d_air.Thickness = params.h;
    params.baseAntenna = patchMicrostrip('Length', params.L, 'Width', params.W, 'Height', params.h, 'Substrate', d_air, 'GroundPlaneLength', Inf, 'GroundPlaneWidth', Inf, 'FeedOffset', [params.L/4, 0]);
    f_start = (freq_ghz - span_ghz/2) * 1e9;
    f_stop = (freq_ghz + span_ghz/2) * 1e9;
    params.freq_sweep = linspace(f_start, f_stop, num_points);
    params.maxEdge = lambda0 / mesh_lambda_frac;
end

function out = heavy_task_antenna(seed, designParams, numTasks)
    ant = designParams.baseAntenna;
    offset_factor = (seed - numTasks/2) / numTasks;
    ant.FeedOffset(1) = ant.Length/4 * (1 + offset_factor*0.05);
    m = mesh(ant, 'MaxEdgeLength', designParams.maxEdge);
    s = sparameters(ant, designParams.freq_sweep);
    s11_db = 20*log10(abs(squeeze(s.Parameters(1,1,:))));
    out = sum(s11_db);
end

function finalize_env()
    try delete(gcp('nocreate')); end
    try maxNumCompThreads('automatic'); end % 恢复默认设置
    fprintf('并行池已关闭，环境已清理。\n');
end