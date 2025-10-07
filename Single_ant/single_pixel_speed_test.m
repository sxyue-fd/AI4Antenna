function results = speed_benchmark_pixel_patch_10G()
% speed_benchmark_pixel_patch_10G.m
% 探针馈电、空气介质、像素化贴片天线的速度基准测试
% - f0 = 10 GHz, L = W = 15 mm, h = 1 mm
% - 扫描：地板尺寸（通过边缘裕量控制）、频点数、网格密度（meshLambdaFraction）
% - 固定随机种子，结果可复现
% - 计时：从建模开始到 S 参数仿真结束
% - 基准：最大地板 × 最密频点 × 最细网格
% - 误差：与基准的 S11（复数）按频率插值后计算 MSE
%
% 依赖：Antenna Toolbox, RF Toolbox, Partial Differential Equation Toolbox(网格)
%
% Haobo 的默认参数可按需修改：gndMarginsMM, numFreqVec, meshFracVec

clc; close all;

fprintf('=== Pixelated Probe-Fed Patch @10 GHz Benchmark ===\n');

%% ------------------- 全局参数 -------------------
rng(42, 'twister');                 % 固定随机种子
f0 = 10e9;                          % 中心频率
c  = physconst('LightSpeed');
lambda0 = c / f0;

L = 15e-3;                          % 贴片长 (m)
W = 15e-3;                          % 贴片宽 (m)
h = 1e-3;                           % 介质厚度 (m) 空气
Npix = 16;                          % 像素分辨率
fill = 0.7;                        % 像素填充率
overlap = 0.2e-3;                   % 像素重叠，增强连通性
feedGuess = [L/4, 0];               % 初始馈电点
feedDia   = min(L/Npix, W/Npix)/10; % 探针直径

% 频扫范围：±1 GHz around 10 GHz
fSpan = 4e9;                        % 总带宽
fmin  = f0 - fSpan/2;
fmax  = f0 + fSpan/2;

% 参数扫描集合（可自行调整）
gndMarginsMM = [7.5, 11, 15];       % 地板每边外扩裕量（mm）
numFreqVec   = [11, 21, 31];       % 频点数
meshFracVec  = [5, 10, 20]; % meshLambdaFraction（越大网格越细）

% 派生：基准配置索引（最大、最大、最大）
idxG_base = numel(gndMarginsMM);
idxF_base = numel(numFreqVec);
idxM_base = numel(meshFracVec);

%% ------------------- 构造随机像素并锁定馈电像素 -------------------
[pixelShapes, feedCenter] = build_pixel_grid(L, W, Npix, overlap, feedGuess);
designVector = rand(Npix, Npix) < fill;
% 强制馈电像素为导体
[fr, fc] = pixel_index_of_point(L, W, Npix, feedCenter);
designVector(fr, fc) = true;

% 组合贴片形状
patchShape = union_pixels(pixelShapes, designVector);

%% ------------------- 先跑"基准配置" -------------------
fprintf('\n-- Running baseline (largest ground, densest freq, finest mesh)...\n');
gndMargin_base = gndMarginsMM(idxG_base)*1e-3;
numFreq_base   = numFreqVec(idxF_base);
meshFrac_base  = meshFracVec(idxM_base);

[f_base, S11_base, t_base] = run_one_case( ...
    patchShape, L, W, h, feedCenter, feedDia, ...
    fmin, fmax, numFreq_base, gndMargin_base, meshFrac_base, lambda0);

fprintf('Baseline done: time = %.2f s, |S11|min=%.1f dB\n', ...
    t_base, 20*log10(min(abs(S11_base))+eps));

%% ------------------- 全部扫描并与基准比较 -------------------
rows = {};
header = {'gnd_margin_mm','num_freq','meshLambdaFrac','sim_time_s','MSE_vs_baseline'};

for ig = 1:numel(gndMarginsMM)
    for jf = 1:numel(numFreqVec)
        for km = 1:numel(meshFracVec)
            gmm   = gndMarginsMM(ig);
            nfreq = numFreqVec(jf);
            mfrac = meshFracVec(km);

            fprintf('\n>> Case: gnd+%d mm, Nf=%d, meshFrac=%d ...\n', gmm, nfreq, mfrac);

            [f_i, S11_i, t_i] = run_one_case( ...
                patchShape, L, W, h, feedCenter, feedDia, ...
                fmin, fmax, nfreq, gmm*1e-3, mfrac, lambda0);

            % 与基准插值对齐计算 MSE（对复数 S11）
            S11b_on_i = interp1(f_base, S11_base, f_i, 'linear', 'extrap'); % 线性
            mseVal = mean(abs(S11_i - S11b_on_i).^2);

            rows(end+1, :) = {gmm, nfreq, mfrac, t_i, mseVal}; %#ok<AGROW>
        end
    end
end

% 汇总表
results = cell2table(rows, 'VariableNames', header);

% 标记基准行（便于查看）
isBaseline = results.gnd_margin_mm == gndMarginsMM(idxG_base) & ...
             results.num_freq      == numFreqVec(idxF_base)   & ...
             results.meshLambdaFrac== meshFracVec(idxM_base);
results.MSE_vs_baseline{isBaseline} = 0;

% 排序（先看网格、再看频点、最后地板）
results = sortrows(results, {'meshLambdaFrac','num_freq','gnd_margin_mm'});

fprintf('\n=== Done. Summary (first 10 rows) ===\n');
disp(results(1:min(10,height(results)), :));

% 可选：保存
try
    writetable(results, 'benchmark_results_10G.csv');
    fprintf('Saved: benchmark_results_10G.csv\n');
catch
    % 忽略保存失败
end

end % main


%% ======================= 子函数 =======================

function [pixelShapes, feedCenter] = build_pixel_grid(L, W, N, overlap, feedGuess)
% 构建 N×N 像素网格的矩形元件（antenna.Rectangle），并把馈电点对齐到其像素中心
pL = L / N; pW = W / N;

pixelShapes = cell(N, N);
startX = -L/2 + pL/2;
startY = -W/2 + pW/2;
for r = 1:N
    for c = 1:N
        cx = startX + (c-1)*pL;
        cy = startY + (r-1)*pW;
        pixelShapes{r,c} = antenna.Rectangle( ...
            'Length', pL + overlap, 'Width', pW + overlap, ...
            'Center', [cx, cy]);
    end
end

% 找到 feedGuess 所在像素中心，作为最终馈电点
[ridx, cidx] = pixel_index_of_point(L, W, N, feedGuess);
feedCenter = pixelShapes{ridx, cidx}.Center;
end


function [ridx, cidx] = pixel_index_of_point(L, W, N, pt)
% 将平面点映射到 N×N 像素索引（行、列）
pL = L / N; pW = W / N;
x = pt(1); y = pt(2);

cidx = floor((x + L/2) / pL) + 1;
ridx = floor((y + W/2) / pW) + 1;

cidx = max(1, min(N, cidx));
ridx = max(1, min(N, ridx));
end


function shape = union_pixels(pixelShapes, mask)
% 将 mask 中为 true 的像素矩形做布尔并集
idx = find(mask(:));
if isempty(idx)
    % 避免空贴片：至少保留一个像素（不会用于辐射，但保证模型可建）
    idx = 1;
end
[r1, c1] = ind2sub(size(mask), idx(1));
shape = pixelShapes{r1, c1};
for k = 2:numel(idx)
    [rr, cc] = ind2sub(size(mask), idx(k));
    shape = shape + pixelShapes{rr, cc};
end
end


function [fvec, S11, t_elapsed] = run_one_case( ...
    patchShape, L, W, h, feedCenter, feedDia, ...
    fmin, fmax, Nf, gndMargin, meshLambdaFrac, lambda0)
% 构造 pcbStack、网格、S 参数；返回 freq、S11（复数）和耗时

ticCase = tic;

% 1) 板和地尺寸
board_L = L + 2*gndMargin;
board_W = W + 2*gndMargin;
ground  = antenna.Rectangle('Length', board_L, 'Width', board_W, 'Center', [0, 0]);

% 2) 介质（空气）——注意厚度单独设
substrateMaterial = dielectric('Air');
substrateMaterial.Thickness = h;

% 3) pcbStack
ant = pcbStack();
ant.BoardShape     = ground;
ant.BoardThickness = h;                          % 先设厚度
ant.FeedDiameter   = feedDia;
ant.Layers         = {patchShape, substrateMaterial, ground}; % 再设 Layers
ant.FeedLocations  = [feedCenter, 1, 3];         % 从上导体(1)打到地(3)

% 4) 网格
maxEdge = lambda0 / meshLambdaFrac;
mesh(ant, 'MaxEdgeLength', maxEdge);

% 5) 频扫 & S 参数
fvec = linspace(fmin, fmax, Nf);
sobj = sparameters(ant, fvec);
S11  = rfparam(sobj, 1, 1);     % 复数 S11（非 dB）

t_elapsed = toc(ticCase);
end
