function pct_benchmark_onepool()
% PCT_BENCHMARK_ONEPOOL
% 主要特性：
% 1. 测试worker数量从1，2，4，直到cpu物理核数上限的并行加速效率
% 2. 自定义一些大规模矩阵操作作为计算负载
% 3. 只启动一次并行池，在同一池里扫描不同并行度 (P)，测试加速效果

%% 0) 环境与参数
assert(license('test','Distrib_Computing_Toolbox')==1, ...
    '未检测到 Parallel Computing Toolbox 许可证。');

clc; close all; rng(1);

% === 池类型开关：false=Processes, true=Threads ===
useThreads = false;     % ← 如需线程池改为 true

% 负载参数（可按机器性能调整）
tasksPerWorker = 12;    % 每个 worker 分配的任务数（建议 8~16 之间）
matrixN        = 512;   % 方阵维度（越大越吃 CPU/内存带宽）
repeats        = 2;     % 每任务内部重复次数（放大负载）
testWorkers    = unique([1, 2.^(0:10)]);  % 1,2,4,8,...

%% 1) 启动一次并行池 + 线程数限制 + 预热
delete(gcp('nocreate'));                     % 清理遗留池
cleanupObj = onCleanup(@() finalize_env());  % 运行结束自动收尾

if useThreads
    % Threads：一个 MATLAB 进程内多个线程
    pool = parpool("Processes", feature('numcores'), "IdleTimeout", 120);
    maxNumCompThreads(1);                    % 避免内层 BLAS/FFT 再并行
else
    % Processes：每个 worker 一个独立 MATLAB 进程
    pc = parcluster('local');
    % 将 JobStorage 放到本地临时目录（避免网络盘/杀软拖慢）
    try pc.JobStorageLocation = fullfile(tempdir, 'pct_jobs'); end
    pool = parpool(pc, pc.NumWorkers, "IdleTimeout", 120);
    pctRunOnAll maxNumCompThreads(1)         % 下发到每个 worker
end

maxWorkers = pool.NumWorkers;
testWorkers = testWorkers(testWorkers <= maxWorkers);
if testWorkers(end) ~= maxWorkers
    testWorkers(end+1) = maxWorkers;
end

% 任务数：保证每个 worker 有多任务，降低尾效应
numTasks = max(tasksPerWorker * maxWorkers, 32);
taskSeeds = 1:numTasks;

fprintf('=== MATLAB PCT Benchmark (One Pool) ===\n');
fprintf('Pool Type: %s | Workers=%d\n', ternary(useThreads,'Threads','Processes'), maxWorkers);
fprintf('Tasks=%d (≈ %d per worker), MatrixN=%d, Repeats/Task=%d\n\n', ...
    numTasks, ceil(numTasks/maxWorkers), matrixN, repeats);

% 预热（JIT/FFT/BLAS）
parfor (k = 1:4, min(4, maxWorkers))
    tmp = fft(rand(512));
end
dummy = heavy_task(taskSeeds(1), matrixN, 1); %#ok<NASGU>

%% 2) 串行基线
fprintf('Running serial baseline (for-loop)...\n');
tSerial = timeit(@() run_serial(taskSeeds, matrixN, repeats));
fprintf('  Serial time: %.3f s\n\n', tSerial);

%% 3) 在同一池中扫描不同并行度 P（不重启池）
times = nan(size(testWorkers));
speedup = nan(size(testWorkers));
eff     = nan(size(testWorkers));

for i = 1:numel(testWorkers)
    P = testWorkers(i);

    if P == 1
        times(i)   = tSerial;
        speedup(i) = 1;
        eff(i)     = 1;
        fprintf('P=%d -> time: %.3f s | speedup: %.2f | eff: %.2f%%\n', ...
            P, times(i), speedup(i), 100*eff(i));
        continue;
    end

    fprintf('Running parallel with P=%d (parfor)...\n', P);
    t = timeit(@() run_parfor_P(taskSeeds, matrixN, repeats, P));
    times(i)   = t;
    speedup(i) = tSerial / t;
    eff(i)     = speedup(i) / P;

    fprintf('  Time: %.3f s | Speedup: %.2f | Efficiency: %.2f%%\n\n', ...
        t, speedup(i), 100*eff(i));
end

%% 4) 可视化
fig = figure('Color','w','Position',[100 100 1000 650]);

% (a) 总耗时
subplot(3,1,1);
xs = string(testWorkers); xs(1) = "1 (serial)";
plot(1:numel(testWorkers), times,'-o','LineWidth',1.8, 'MarkerSize',6);
grid on; xticks(1:numel(xs)); xticklabels(xs);
ylabel('Time (s)'); title('总耗时 vs. 并行度 P');

% (b) 加速比（含理想线）
subplot(3,1,2);
plot(1:numel(testWorkers), speedup,'-o','LineWidth',1.8, 'MarkerSize',6); hold on;
plot(1:numel(testWorkers), testWorkers./testWorkers(1), '--','LineWidth',1.2);
grid on; xticks(1:numel(xs)); xticklabels(xs);
ylabel('Speedup'); title('加速比（含理想线）');
legend('实测','理想', 'Location','northwest');

% (c) 并行效率
subplot(3,1,3);
plot(1:numel(testWorkers), 100*eff,'-o','LineWidth',1.8, 'MarkerSize',6);
grid on; xticks(1:numel(xs)); xticklabels(xs);
ylabel('Efficiency (%)'); ylim([0 110]);
title('并行效率 = Speedup / P * 100%');

sgtitle(sprintf('PCT Benchmark — Tasks=%d, N=%d, Repeats/Task=%d', ...
    numTasks, matrixN, repeats), 'FontWeight','bold');

% 去掉导出时的工具栏提示
set(fig, 'ToolBar', 'none');
try axtoolbar(gca, 'Visible', 'off'); catch, end

exportgraphics(fig, 'pct_benchmark.png', 'Resolution', 150);
fprintf('图已保存：pct_benchmark.png\n\n');

%% 5) 结果表
T = table(testWorkers(:), times(:), speedup(:), eff(:)*100, ...
    'VariableNames', {'P','Time_s','Speedup','Efficiency_pct'});
disp('=== 结果表 ==='); disp(T);

end

%% ====== 辅助函数 ======

function run_serial(taskSeeds, N, repeats)
acc = 0;
for k = 1:numel(taskSeeds)
    acc = acc + heavy_task(taskSeeds(k), N, repeats);
end
assert(isfinite(acc));
end

function run_parfor_P(taskSeeds, N, repeats, P)
acc = 0;
parfor (k = 1:numel(taskSeeds), P)   % ★ 显式并行度，单池内切换
    acc = acc + heavy_task(taskSeeds(k), N, repeats);
end
assert(isfinite(acc));
end

function out = heavy_task(seed, N, repeats)
% 可重复、CPU 密集，无 I/O 与共享状态
out = 0;
[iIdx, jIdx] = ndgrid(1:N, 1:N);
A0 = sin(0.013*seed*iIdx) + cos(0.017*seed*jIdx);
B0 = sin(0.021*seed*jIdx) + cos(0.019*seed*iIdx);

A = A0; B = B0;
for r = 1:repeats
    C = A*B;                 % 矩阵乘（BLAS）
    F = fft(C, [], 2);       % 行向 FFT
    G = real(F.*conj(F));    % 功率谱

    % 数值稳健：QR 分解，避免 chol 正定性要求
    H = (G'*G) / (N*N) + eye(N)*1e-3;
    [Q, ~] = qr(H);
    out = out + sum(log(abs(diag(Q))) + eps);

    % 轻微变换，避免每轮完全一致
    A = 0.9*A + 0.1*B0;
    B = 0.9*B + 0.1*A0;
end
end

function s = ternary(cond, a, b)
if cond, s = a; else, s = b; end
end

function finalize_env()
% 统一收尾：关池 + 恢复线程策略
try delete(gcp('nocreate')); end
try maxNumCompThreads('automatic'); end
end
