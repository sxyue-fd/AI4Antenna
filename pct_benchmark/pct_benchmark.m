function pct_benchmark()
% PCT_BENCHMARK  简明 MATLAB Parallel Computing Toolbox 加速测试
% 主要特性：
% 1. 测试worker数量从1，2，4，直到cpu物理核数上限的并行加速效率
% 2. 自定义一些大规模矩阵操作作为计算负载
% 3. 每次测试后关闭并行池，下个测试再重启

%% 0) 基本环境与参数
assert(license('test','Distrib_Computing_Toolbox')==1, ...
    '未检测到 Parallel Computing Toolbox 许可证。');

clc; close all; rng(1);  % 固定随机种子确保可重复
pc = parcluster('local');
maxWorkers = pc.NumWorkers;

% 为了"简单可读"，按 1, 2, 4, 8, ..., <= maxWorkers 取点（去重、升序）
cands = unique([1, 2.^(0:10)]);
testWorkers = cands(cands <= maxWorkers);
if testWorkers(end) ~= maxWorkers
    testWorkers(end+1) = maxWorkers;
end

% 负载参数（可按机器性能调整）
numTasks = 32;    % 任务个数（越大统计越稳）
matrixN  = 600;   % 方阵维度（越大越吃 CPU）
repeats  = 2;     % 每任务内部重复次数（放大负载）

fprintf('=== MATLAB PCT Benchmark ===\n');
fprintf('Detected local workers: %d\n', maxWorkers);
fprintf('Tasks=%d, MatrixN=%d, Repeats/Task=%d\n\n', numTasks, matrixN, repeats);

%% 1) 预构建输入，避免测试期间的 I/O / 随机抖动
% 为每个任务构造一个小的、确定性的"种子"，任务之间相互独立
taskSeeds = 1:numTasks;

%% 2) 预热（JIT warm-up，避免首次调用开销干扰）
dummy = heavy_task(taskSeeds(1), matrixN, 1); %#ok<NASGU>

%% 3) 串行基线（workers=1 的 for 循环）
fprintf('Running serial baseline (for-loop)...\n');
tSerial = timeit(@() run_serial(taskSeeds, matrixN, repeats));
fprintf('  Serial time: %.3f s\n\n', tSerial);

%% 4) 并行测试（不同 worker 数）
times = nan(size(testWorkers));
speedup = nan(size(testWorkers));
eff     = nan(size(testWorkers));

for i = 1:numel(testWorkers)
    w = testWorkers(i);
    if w == 1
        % 为了公平，workers=1 下使用串行 for-loop 的结果作为基线
        times(i)   = tSerial;
        speedup(i) = tSerial / times(i);
        eff(i)     = speedup(i) / w;
        fprintf('Workers=%d -> time: %.3f s | speedup: %.2f | eff: %.2f%%\n', ...
            w, times(i), speedup(i), 100*eff(i));
        continue;
    end

    % 确保按需开关池；启动耗时不计入测量
    pool = gcp('nocreate');
    if ~isempty(pool) && pool.NumWorkers ~= w
        delete(pool);
        pool = [];
    end
    if isempty(pool)
        parpool("Threads", w);  % 启动池（不计时）
        maxNumCompThreads(1);
    end

    fprintf('Running parallel with %d workers (parfor)...\n', w);
    t = timeit(@() run_parfor(taskSeeds, matrixN, repeats));
    times(i)   = t;
    speedup(i) = tSerial / t;
    eff(i)     = speedup(i) / w;

    fprintf('  Time: %.3f s | Speedup: %.2f | Efficiency: %.2f%%\n\n', ...
        t, speedup(i), 100*eff(i));

    % 可选：保持池用于后续 w（如果下一轮 w 不同，会重建）
end

% 清理并行池（可选）
pool = gcp('nocreate');
if ~isempty(pool), delete(pool); end

%% 5) 可视化
figure('Color','w','Position',[100 100 1000 650]);

% (a) 总耗时
subplot(3,1,1);
xs = string(testWorkers); xs(1) = "1 (serial)";
plot(1:numel(testWorkers), times,'-o','LineWidth',1.8, 'MarkerSize',6);
grid on; xticks(1:numel(xs)); xticklabels(xs);
ylabel('Time (s)'); title('总耗时 vs. Workers');

% (b) 加速比（含理想线）
subplot(3,1,2);
plot(1:numel(testWorkers), speedup,'-o','LineWidth',1.8, 'MarkerSize',6); hold on;
plot(1:numel(testWorkers), testWorkers./testWorkers(1), '--','LineWidth',1.2); % 理想线：与 worker 成正比
grid on; xticks(1:numel(xs)); xticklabels(xs);
ylabel('Speedup'); title('加速比（含理想线）');
legend('实测','理想', 'Location','northwest');

% (c) 并行效率
subplot(3,1,3);
plot(1:numel(testWorkers), 100*eff,'-o','LineWidth',1.8, 'MarkerSize',6);
grid on; xticks(1:numel(xs)); xticklabels(xs);
ylabel('Efficiency (%)'); ylim([0 110]);
title('并行效率 = Speedup / Workers * 100%');

sgtitle(sprintf('PCT Benchmark — Tasks=%d, N=%d, Repeats/Task=%d', numTasks, matrixN, repeats), ...
    'FontWeight','bold');

exportgraphics(gcf, 'pct_benchmark.png', 'Resolution', 150);
fprintf('图已保存：pct_benchmark.png\n\n');

%% 6) 结果表
T = table(testWorkers(:), times(:), speedup(:), eff(:)*100, ...
    'VariableNames', {'Workers','Time_s','Speedup','Efficiency_pct'});
disp('=== 结果表 ==='); disp(T);

end

%% ====== 辅助函数 ======

function run_serial(taskSeeds, N, repeats)
% 串行版本（for）
acc = 0;
for k = 1:numel(taskSeeds)
    acc = acc + heavy_task(taskSeeds(k), N, repeats);
end
% 防止优化器把计算消掉
assert(isfinite(acc));
end

function run_parfor(taskSeeds, N, repeats)
% 并行版本（parfor）
acc = 0;
parfor k = 1:numel(taskSeeds)
    acc = acc + heavy_task(taskSeeds(k), N, repeats);
end
assert(isfinite(acc));
end

function out = heavy_task(seed, N, repeats)
% 计算密集的"可重复"任务，不依赖 I/O，不共享状态
% 内容：构造确定性矩阵 -> 矩阵乘、FFT、Cholesky -> 标量累加
% 这样既考 BLAS，也有频域操作与分解操作
out = 0;
% 以 seed 构造确定性矩阵（无随机）
[iIdx, jIdx] = ndgrid(1:N, 1:N);
A0 = sin(0.013*seed*iIdx) + cos(0.017*seed*jIdx);
B0 = sin(0.021*seed*jIdx) + cos(0.019*seed*iIdx);

A = A0; B = B0;
for r = 1:repeats
    C = A*B;                 % 矩阵乘（CPU/BLAS）
    F = fft(C, [], 2);       % 行向 FFT
    G = real(F.*conj(F));    % 功率谱
    % 构造对称正定矩阵做 Cholesky
    H = (G'*G) / (N*N) + eye(N)*1e-3;
    [Q, ~] = qr(H);             % QR分解
    out = out + sum(log(diag(Q)));  % 累加一个标量，避免被优化消除
    % 轻微变换以避免每轮完全一致
    A = 0.9*A + 0.1*B0; 
    B = 0.9*B + 0.1*A0;
end
end
