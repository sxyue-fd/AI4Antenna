% eval_dataset_current_signed_log_clip_zscore_eval.m
% 评估 HDF5 数据集中的 /current 电流数据。
%
% 功能：
%   1) 随机抽样若干天线，统计 |current| 的全局分位数，并直接打印到终端；
%   2) 新增 tail ratio、zero ratio、signed raw current std、signed-log-scaled std；
%   3) 显示全局电流幅值直方图 global_histograms；
%   4) 显示 signed_log(J / p95_abs) 后的 signed histogram，并叠加 clip 前后曲线；
%   5) 新增 signed-log、signed-log + clip、signed-log + clip + z-score 的分位数统计；
%   6) 显示 signed-log + z-score histogram，并叠加 clip 前后曲线；
%   7) 显示不同频点、不同通道的 signed raw current std；
%   6) 显示中间频点的逐像素 q99 图 pixel_q99_mid_frequency；
%   7) 统计每张电流图的最大值 max(abs(current))，并按频点、通道显示直方图；
%   8) 随机选取 3 个天线样本，显示 8/10/12 GHz 下 Jx_real 的线性尺度和 signed-log 尺度电流图；
%   9) 不保存任何统计文件或图片。

clear; clc; close all;

%% 用户配置
h5path = '';                         % 留空：自动寻找最新的含 /current 的 HDF5 文件
sample_count_target = 100000;          % 用于统计的随机样本数
rng_seed = 42;                      % 随机种子，方便复现实验

target_freq_ghz = [8 10 12];         % 需要绘图的频率
plot_sample_count = 3;               % 随机绘制的天线数量
quantile_probs = [0 0.001 0.01 0.05 0.10 0.25 0.50 0.75 0.90 0.95 0.99 0.999 1];

num_bins = 120;                      % 全局电流幅值直方图箱数
hist_clip_percentile = 100;          % 百分数口径：100 表示直方图横轴不截断
edge_probe_sample_count = 64;        % 估计直方图边界时使用的样本数
pixel_q_prob = 0.99;                 % pixel_q99_mid_frequency 对应的分位数

zero_eps_factor = 1e-6;              % near-zero 阈值 = max(1e-12, p99(abs(J))*zero_eps_factor)
signed_log_alpha_prob = 0.95;             % signed-log 压缩尺度 alpha = p95(abs(J))
signed_log_clip_T = 4.0;                  % signed-log 域 clip 阈值：Y_clip = clip(Y, -T, T)
signed_log_hist_bins = 120;               % signed-log-scaled histogram 箱数
signed_log_hist_limit = 8;                % signed-log histogram 显示范围 [-limit, limit]，范围外单独计数

% signed-log / signed-log+clip / z-score 后的分位数统计。
transform_abs_quantile_probs = [0.50 0.90 0.95 0.99 0.999 0.9999 0.99995 0.99999 1];
zscore_abs_quantile_probs = transform_abs_quantile_probs;
zscore_hist_bins = 160;                   % z-score histogram 箱数
zscore_hist_limit = 12;                   % z-score histogram 显示范围 [-limit, limit]，范围外单独计数

max_hist_bins = 100;                 % 每张电流图最大值主体区间直方图箱数
max_hist_detail_limit = 100;        % 主体区间上限；超过该值的电流图最大值单独计数
channel_names = {'Jx_real', 'Jx_imag', 'Jy_real', 'Jy_imag'};
jx_real_channel = 1;
plot_color_percentile = 1;           % 概率口径：1 表示绘图色阶不截断

%% 定位数据集
script_dir = fileparts(mfilename('fullpath'));
if isempty(script_dir)
    script_dir = pwd;
end

if isempty(h5path)
    h5path = pick_latest_h5_with_current(script_dir);
elseif ~isfile(h5path)
    h5path_local = fullfile(script_dir, h5path);
    if isfile(h5path_local)
        h5path = h5path_local;
    end
end

assert(isfile(h5path), '找不到 HDF5 文件：%s', h5path);
assert(dataset_exists(h5path, '/current'), '文件中不存在 /current 数据集：%s', h5path);

fprintf('正在读取电流数据集：%s\n', h5path);

infoC = h5info(h5path, '/current');
csize = double(infoC.Dataspace.Size);
assert(numel(csize) == 5, '/current 必须是 [B,Fc,Cc,fineN,fineN]。');

B_all = csize(1);
Fc = csize(2);
Cc = csize(3);
fineN = csize(4);
fineN2 = csize(5);
assert(fineN == fineN2, '/current 的空间网格必须是方阵。');
assert(jx_real_channel <= Cc, 'Jx_real 通道索引超过 /current 的通道数。');

if Cc ~= numel(channel_names)
    channel_names = arrayfun(@(k) sprintf('ch%d', k), 1:Cc, 'UniformOutput', false);
end

if dataset_exists(h5path, '/current_freq_hz')
    current_freq_hz = double(h5read(h5path, '/current_freq_hz'));
    current_freq_hz = current_freq_hz(:);
else
    current_freq_hz = (1:Fc).';
end

fprintf('数据规模：B=%d, Fc=%d, Cc=%d, fineN=%d\n', B_all, Fc, Cc, fineN);
fprintf('电流频点(GHz)：%s\n\n', sprintf('%.6g ', current_freq_hz(:).' / 1e9));

%% 随机抽样
rng(rng_seed);
sample_count = min(sample_count_target, B_all);
sample_idx = sort(randperm(B_all, sample_count));
mid_f = max(1, round(Fc / 2));

fprintf('=== |current| 全局分位数统计 ===\n');
fprintf('随机样本数：%d / %d，随机种子：%d\n\n', sample_count, B_all, rng_seed);

%% 先用小样本估计全局直方图边界
probe_idx = sample_idx(round(linspace(1, sample_count, min(edge_probe_sample_count, sample_count))));
probe_values = [];

for kf = 1:Fc
    vals_probe = abs(read_current_frequency(h5path, probe_idx, kf, Cc, fineN));
    probe_values = [probe_values; single(vals_probe(:))]; %#ok<AGROW>
end

probe_values = double(probe_values(isfinite(probe_values)));
assert(~isempty(probe_values), '探测样本中没有有限的电流值。');

edge_hi = percentile_vector(probe_values, hist_clip_percentile / 100);
edge_hi = max(edge_hi, eps);
hist_edges = [linspace(0, edge_hi, num_bins), inf];
hist_bin_centers = 0.5 * (hist_edges(1:end-2) + hist_edges(2:end-1));

signed_log_hist_edges = linspace(-signed_log_hist_limit, signed_log_hist_limit, signed_log_hist_bins + 1);
signed_log_hist_bin_centers = 0.5 * (signed_log_hist_edges(1:end-1) + signed_log_hist_edges(2:end));

zscore_hist_edges = linspace(-zscore_hist_limit, zscore_hist_limit, zscore_hist_bins + 1);
zscore_hist_bin_centers = 0.5 * (zscore_hist_edges(1:end-1) + zscore_hist_edges(2:end));

fprintf('全局直方图范围：0 到 %.6g，最后一个箱用于统计超过该范围的值。\n', edge_hi);
fprintf('signed-log-scaled histogram 范围：[-%.6g, %.6g]，范围外单独计数。\n', ...
    signed_log_hist_limit, signed_log_hist_limit);
fprintf('signed-log clip 阈值：T = %.6g，即 Y_clip = min(max(Y, -T), T)。\n', signed_log_clip_T);
fprintf('signed-log + z-score histogram 范围：[-%.6g, %.6g]，范围外单独计数。\n\n', ...
    zscore_hist_limit, zscore_hist_limit);

%% 主统计
tic_eval = tic;
quantiles_global = zeros(numel(quantile_probs), Fc, Cc, 'single');
hist_counts_global = zeros(num_bins, Fc, Cc, 'uint64');
pixel_q_mid = zeros(Cc, fineN, fineN, 'single');
current_map_max = zeros(sample_count, Fc, Cc, 'single');

% 标准化相关统计：这些指标用于判断是否适合直接 z-score，或是否需要 signed-log + z-score。
raw_std_fc = zeros(Fc, Cc, 'single');
abs_p50_fc = zeros(Fc, Cc, 'single');
abs_p95_fc = zeros(Fc, Cc, 'single');
abs_p99_fc = zeros(Fc, Cc, 'single');
abs_p999_fc = zeros(Fc, Cc, 'single');
zero_ratio_fc = zeros(Fc, Cc, 'single');
tail_ratio_fc = zeros(Fc, Cc, 'single');
signed_log_alpha_fc = zeros(Fc, Cc, 'single');
signed_log_mean_fc = zeros(Fc, Cc, 'single');
signed_log_std_fc = zeros(Fc, Cc, 'single');
signed_log_hist_counts = zeros(signed_log_hist_bins, Fc, Cc, 'uint64');
signed_log_hist_underflow = zeros(Fc, Cc, 'uint64');
signed_log_hist_overflow = zeros(Fc, Cc, 'uint64');
signed_log_abs_quantiles = zeros(numel(transform_abs_quantile_probs), Fc, Cc, 'single');

% signed-log + clip 后的统计：clip 在 signed-log 域进行，后续 mu/std 应基于 clipped Y 计算。
signed_log_clip_T_fc = single(signed_log_clip_T * ones(Fc, Cc));
signed_log_clip_mean_fc = zeros(Fc, Cc, 'single');
signed_log_clip_std_fc = zeros(Fc, Cc, 'single');
signed_log_clip_abs_quantiles = zeros(numel(transform_abs_quantile_probs), Fc, Cc, 'single');
signed_log_clip_ratio_fc = zeros(Fc, Cc, 'single');
signed_log_clip_hist_counts = zeros(signed_log_hist_bins, Fc, Cc, 'uint64');
signed_log_clip_hist_underflow = zeros(Fc, Cc, 'uint64');
signed_log_clip_hist_overflow = zeros(Fc, Cc, 'uint64');

% signed-log + z-score 后的统计：分别保存 no clip 和 clip 后的版本。
zscore_abs_quantiles = zeros(numel(zscore_abs_quantile_probs), Fc, Cc, 'single');
zscore_hist_counts = zeros(zscore_hist_bins, Fc, Cc, 'uint64');
zscore_hist_underflow = zeros(Fc, Cc, 'uint64');
zscore_hist_overflow = zeros(Fc, Cc, 'uint64');
zscore_clip_abs_quantiles = zeros(numel(zscore_abs_quantile_probs), Fc, Cc, 'single');
zscore_clip_hist_counts = zeros(zscore_hist_bins, Fc, Cc, 'uint64');
zscore_clip_hist_underflow = zeros(Fc, Cc, 'uint64');
zscore_clip_hist_overflow = zeros(Fc, Cc, 'uint64');

for kf = 1:Fc
    raw = read_current_frequency(h5path, sample_idx, kf, Cc, fineN);        % [sample, channel, x, y], signed current
    vals = abs(raw);                                                       % 幅值，用于 tail/zero/pixel-q 统计
    map_max = squeeze(max(max(vals, [], 4), [], 3));                       % [sample, channel]
    current_map_max(:, kf, :) = reshape(single(map_max), [sample_count, 1, Cc]);

    for cc = 1:Cc
        x = double(raw(:, cc, :, :));
        x = x(isfinite(x));
        v = abs(x);

        quantiles_global(:, kf, cc) = single(percentile_vector(v, quantile_probs));
        hist_counts_global(:, kf, cc) = uint64(histcounts(v, hist_edges).');

        % signed raw current 的标准差：后续判断不同通道尺度是否相差很大。
        raw_std_fc(kf, cc) = single(std(x));

        % 幅值分位数和长尾指标。
        abs_p50 = percentile_vector(v, 0.50);
        abs_p95 = percentile_vector(v, signed_log_alpha_prob);
        abs_p99 = percentile_vector(v, 0.99);
        abs_p999 = percentile_vector(v, 0.999);

        abs_p50_fc(kf, cc) = single(abs_p50);
        abs_p95_fc(kf, cc) = single(abs_p95);
        abs_p99_fc(kf, cc) = single(abs_p99);
        abs_p999_fc(kf, cc) = single(abs_p999);

        zero_eps = max(1e-12, abs_p99 * zero_eps_factor);
        zero_ratio_fc(kf, cc) = single(mean(v < zero_eps));
        tail_ratio_fc(kf, cc) = single(abs_p999 / max(abs_p50, 1e-12));

        % signed-log 压缩后的 signed histogram 和 std。
        alpha = max(abs_p95, 1e-12);
        x_signed_log = signed_log_transform(x, alpha);

        signed_log_mu = mean(x_signed_log);
        signed_log_sigma = std(x_signed_log);
        signed_log_sigma = max(signed_log_sigma, eps);
        x_zscore = (x_signed_log - signed_log_mu) ./ signed_log_sigma;

        % 在 signed-log 域做 clip：Y_clip = clip(Y, -T, T)。
        x_signed_log_clip = clip_symmetric(x_signed_log, signed_log_clip_T);
        signed_log_clip_mu = mean(x_signed_log_clip);
        signed_log_clip_sigma = std(x_signed_log_clip);
        signed_log_clip_sigma = max(signed_log_clip_sigma, eps);
        x_zscore_clip = (x_signed_log_clip - signed_log_clip_mu) ./ signed_log_clip_sigma;

        signed_log_alpha_fc(kf, cc) = single(alpha);
        signed_log_mean_fc(kf, cc) = single(signed_log_mu);
        signed_log_std_fc(kf, cc) = single(signed_log_sigma);
        signed_log_abs_quantiles(:, kf, cc) = single(percentile_vector(abs(x_signed_log), transform_abs_quantile_probs));
        signed_log_hist_counts(:, kf, cc) = uint64(histcounts(x_signed_log, signed_log_hist_edges).');
        signed_log_hist_underflow(kf, cc) = uint64(nnz(x_signed_log < signed_log_hist_edges(1)));
        signed_log_hist_overflow(kf, cc) = uint64(nnz(x_signed_log > signed_log_hist_edges(end)));

        signed_log_clip_mean_fc(kf, cc) = single(signed_log_clip_mu);
        signed_log_clip_std_fc(kf, cc) = single(signed_log_clip_sigma);
        signed_log_clip_abs_quantiles(:, kf, cc) = single(percentile_vector(abs(x_signed_log_clip), transform_abs_quantile_probs));
        signed_log_clip_ratio_fc(kf, cc) = single(mean(abs(x_signed_log) > signed_log_clip_T));
        signed_log_clip_hist_counts(:, kf, cc) = uint64(histcounts(x_signed_log_clip, signed_log_hist_edges).');
        signed_log_clip_hist_underflow(kf, cc) = uint64(nnz(x_signed_log_clip < signed_log_hist_edges(1)));
        signed_log_clip_hist_overflow(kf, cc) = uint64(nnz(x_signed_log_clip > signed_log_hist_edges(end)));

        zscore_abs_quantiles(:, kf, cc) = single(percentile_vector(abs(x_zscore), zscore_abs_quantile_probs));
        zscore_hist_counts(:, kf, cc) = uint64(histcounts(x_zscore, zscore_hist_edges).');
        zscore_hist_underflow(kf, cc) = uint64(nnz(x_zscore < zscore_hist_edges(1)));
        zscore_hist_overflow(kf, cc) = uint64(nnz(x_zscore > zscore_hist_edges(end)));

        zscore_clip_abs_quantiles(:, kf, cc) = single(percentile_vector(abs(x_zscore_clip), zscore_abs_quantile_probs));
        zscore_clip_hist_counts(:, kf, cc) = uint64(histcounts(x_zscore_clip, zscore_hist_edges).');
        zscore_clip_hist_underflow(kf, cc) = uint64(nnz(x_zscore_clip < zscore_hist_edges(1)));
        zscore_clip_hist_overflow(kf, cc) = uint64(nnz(x_zscore_clip > zscore_hist_edges(end)));
    end

    if kf == mid_f
        sorted_vals = sort(double(vals), 1);
        qvals = percentile_sorted_dim1(sorted_vals, pixel_q_prob);
        pixel_q_mid(:, :, :) = reshape(single(qvals), [Cc, fineN, fineN]);
    end

    print_quantile_block(kf, current_freq_hz, channel_names, quantile_probs, quantiles_global);
end

fprintf('分位数统计完成，耗时 %.1f 秒。\n\n', toc(tic_eval));

%% 显示统计图，但不保存
print_standardization_summary(current_freq_hz, channel_names, ...
    raw_std_fc, abs_p50_fc, abs_p95_fc, abs_p999_fc, ...
    zero_ratio_fc, tail_ratio_fc, signed_log_alpha_fc, signed_log_std_fc);

show_global_histograms(current_freq_hz, channel_names, hist_bin_centers, hist_counts_global);
print_signed_log_clip_summary(current_freq_hz, channel_names, transform_abs_quantile_probs, signed_log_clip_T_fc, ...
    signed_log_mean_fc, signed_log_std_fc, signed_log_abs_quantiles, ...
    signed_log_clip_mean_fc, signed_log_clip_std_fc, signed_log_clip_abs_quantiles, signed_log_clip_ratio_fc);
show_signed_log_scaled_histograms(current_freq_hz, channel_names, ...
    signed_log_hist_bin_centers, signed_log_hist_counts, signed_log_hist_underflow, signed_log_hist_overflow, ...
    signed_log_clip_hist_counts, signed_log_clip_hist_underflow, signed_log_clip_hist_overflow, signed_log_clip_T);
print_zscore_quantile_summary(current_freq_hz, channel_names, zscore_abs_quantile_probs, ...
    signed_log_mean_fc, signed_log_std_fc, zscore_abs_quantiles, ...
    zscore_hist_underflow, zscore_hist_overflow);
print_zscore_clip_quantile_summary(current_freq_hz, channel_names, zscore_abs_quantile_probs, ...
    signed_log_clip_mean_fc, signed_log_clip_std_fc, zscore_clip_abs_quantiles, ...
    zscore_clip_hist_underflow, zscore_clip_hist_overflow, signed_log_clip_T);
show_zscore_histograms(current_freq_hz, channel_names, zscore_hist_bin_centers, ...
    zscore_hist_counts, zscore_hist_underflow, zscore_hist_overflow, ...
    zscore_clip_hist_counts, zscore_clip_hist_underflow, zscore_clip_hist_overflow, signed_log_clip_T);
show_channel_std_by_frequency(current_freq_hz, channel_names, raw_std_fc, signed_log_std_fc);
show_pixel_q99_mid_frequency(current_freq_hz, channel_names, pixel_q_mid, mid_f, pixel_q_prob, plot_color_percentile);
print_current_map_max_summary(current_map_max, quantile_probs);
show_current_map_max_histogram(current_map_max, max_hist_bins, max_hist_detail_limit);

%% 随机选取 3 个天线，显示 Jx_real 电流图
plot_count = min(plot_sample_count, B_all);
plot_idx = sort(randperm(B_all, plot_count));
freq_idx = match_frequency_indices(current_freq_hz, target_freq_ghz);

fprintf('=== Jx_real 电流图 ===\n');
fprintf('随机绘图样本编号：%s\n', sprintf('%d ', plot_idx));
fprintf('目标频率(GHz)：%s\n', sprintf('%.6g ', target_freq_ghz));
fprintf('实际匹配频率(GHz)：%s\n\n', sprintf('%.6g ', current_freq_hz(freq_idx).' / 1e9));

show_jx_real_figures(h5path, plot_idx, freq_idx, current_freq_hz, ...
    jx_real_channel, fineN, plot_color_percentile);

%% 本脚本使用的局部函数
function h5path = pick_latest_h5_with_current(script_dir)
% 优先从 data_generation/dataset_out 找数据；如果没有，再从项目根目录 datasets 找。
candidate_dirs = {
    fullfile(script_dir, 'dataset_out')
    fullfile(fileparts(script_dir), 'datasets')
    };

for id = 1:numel(candidate_dirs)
    dataset_dir = candidate_dirs{id};
    if ~isfolder(dataset_dir)
        continue;
    end

    files = dir(fullfile(dataset_dir, '*.h5'));
    if isempty(files)
        continue;
    end

    [~, order] = sort([files.datenum], 'descend');
    files = files(order);

    for ii = 1:numel(files)
        candidate = fullfile(files(ii).folder, files(ii).name);
        if dataset_exists(candidate, '/current')
            h5path = candidate;
            return;
        end
    end
end

error('未找到包含 /current 的 .h5 文件。');
end

function vals = read_current_frequency(h5path, sample_idx, kf, Cc, fineN)
% 按连续样本块读取单个频点，避免随机索引导致大量小 IO。
sample_idx = sample_idx(:).';
sample_count = numel(sample_idx);
vals = zeros(sample_count, Cc, fineN, fineN, 'single');

[run_starts, run_lengths, out_starts] = contiguous_runs(sample_idx);
for ir = 1:numel(run_starts)
    b0 = run_starts(ir);
    nb = run_lengths(ir);
    out0 = out_starts(ir);

    block = h5read(h5path, '/current', ...
        double([b0 kf 1 1 1]), double([nb 1 Cc fineN fineN]));
    block = reshape(single(block), [nb, 1, Cc, fineN, fineN]);
    block = squeeze(block(:, 1, :, :, :));

    if nb == 1
        block = reshape(block, [1, Cc, fineN, fineN]);
    end

    vals(out0:(out0 + nb - 1), :, :, :) = block;
end
end

function [run_starts, run_lengths, out_starts] = contiguous_runs(idx)
% 把排序后的样本编号拆成若干连续区间，方便 HDF5 分块读取。
idx = idx(:);
breaks = [true; diff(idx) ~= 1];
out_starts = find(breaks);
run_starts = idx(out_starts);
run_ends = [out_starts(2:end) - 1; numel(idx)];
run_lengths = run_ends - out_starts + 1;
end

function q = percentile_vector(v, probs)
% 用线性插值计算分位数，避免依赖 Statistics Toolbox。
v = sort(v(:));
v = v(isfinite(v));
assert(~isempty(v), '无法对空数组计算分位数。');

probs = probs(:);
n = numel(v);
q = zeros(numel(probs), 1);
for ip = 1:numel(probs)
    p = min(max(probs(ip), 0), 1);
    pos = 1 + (n - 1) * p;
    lo = floor(pos);
    hi = ceil(pos);
    w = pos - lo;
    if lo == hi
        q(ip) = v(lo);
    else
        q(ip) = (1 - w) * v(lo) + w * v(hi);
    end
end
end

function q = percentile_sorted_dim1(sorted_vals, probs)
% 对已经沿第 1 维排序的数据计算分位数。
probs = probs(:);
n = size(sorted_vals, 1);
tail_size = size(sorted_vals);
tail_size = tail_size(2:end);

flat = reshape(sorted_vals, n, []);
qflat = zeros(numel(probs), size(flat, 2));

for ip = 1:numel(probs)
    p = min(max(probs(ip), 0), 1);
    pos = 1 + (n - 1) * p;
    lo = floor(pos);
    hi = ceil(pos);
    w = pos - lo;
    if lo == hi
        qflat(ip, :) = flat(lo, :);
    else
        qflat(ip, :) = (1 - w) * flat(lo, :) + w * flat(hi, :);
    end
end

q = reshape(qflat, [numel(probs), tail_size]);
end

function print_quantile_block(kf, current_freq_hz, channel_names, quantile_probs, quantiles_global)
% 把一个频点下所有通道的全局分位数直接打印到终端。
if numel(current_freq_hz) >= kf
    fprintf('频点 %d：%.6g GHz\n', kf, current_freq_hz(kf) / 1e9);
else
    fprintf('频点 %d\n', kf);
end

fprintf('%-12s', 'channel');
for iq = 1:numel(quantile_probs)
    fprintf(' q%-8.4g', quantile_probs(iq));
end
fprintf('\n');

for cc = 1:numel(channel_names)
    fprintf('%-12s', channel_names{cc});
    for iq = 1:numel(quantile_probs)
        fprintf(' %-9.4g', double(quantiles_global(iq, kf, cc)));
    end
    fprintf('\n');
end
fprintf('\n');
end

function print_standardization_summary(current_freq_hz, channel_names, raw_std_fc, abs_p50_fc, abs_p95_fc, abs_p999_fc, zero_ratio_fc, tail_ratio_fc, signed_log_alpha_fc, signed_log_std_fc)
% 打印标准化相关指标。
% tail_ratio = p99.9(abs(J)) / p50(abs(J))，数值越大，长尾越明显。
% zero_ratio = mean(abs(J) < max(1e-12, p99(abs(J))*zero_eps_factor))。
fprintf('=== 标准化相关统计：tail ratio / zero ratio / std ===\n');
fprintf('%-10s %-12s %-12s %-12s %-12s %-12s %-12s %-12s %-12s\n', ...
    'freqGHz', 'channel', 'raw_std', 'abs_p50', 'abs_p95', 'abs_p999', 'tail', 'zero', 'signed_log_std');

[Fc, Cc] = size(raw_std_fc);
for kf = 1:Fc
    if numel(current_freq_hz) >= kf
        freq_ghz = current_freq_hz(kf) / 1e9;
    else
        freq_ghz = kf;
    end

    for cc = 1:Cc
        fprintf('%-10.6g %-12s %-12.4g %-12.4g %-12.4g %-12.4g %-12.4g %-12.4g %-12.4g\n', ...
            freq_ghz, channel_names{cc}, ...
            double(raw_std_fc(kf, cc)), double(abs_p50_fc(kf, cc)), ...
            double(abs_p95_fc(kf, cc)), double(abs_p999_fc(kf, cc)), ...
            double(tail_ratio_fc(kf, cc)), double(zero_ratio_fc(kf, cc)), ...
            double(signed_log_std_fc(kf, cc)));
    end
end

fprintf('\n说明：tail > 50 通常说明长尾明显；zero > 0.3 说明近零区域较多；raw_std 可用于比较不同通道尺度。\n');
fprintf('signed-log alpha 使用 p95(abs(J))，后续若采用 signed-log + z-score，可把 alpha/mean/std 只在 train set 上重新拟合。\n\n');

fprintf('=== signed-log alpha = p95(abs(J)) ===\n');
fprintf('%-10s', 'freqGHz');
for cc = 1:Cc
    fprintf(' %-12s', channel_names{cc});
end
fprintf('\n');

for kf = 1:Fc
    if numel(current_freq_hz) >= kf
        freq_ghz = current_freq_hz(kf) / 1e9;
    else
        freq_ghz = kf;
    end
    fprintf('%-10.6g', freq_ghz);
    for cc = 1:Cc
        fprintf(' %-12.4g', double(signed_log_alpha_fc(kf, cc)));
    end
    fprintf('\n');
end
fprintf('\n');
end



function print_signed_log_clip_summary(current_freq_hz, channel_names, transform_abs_quantile_probs, clip_T_fc, ...
    signed_log_mean_fc, signed_log_std_fc, signed_log_abs_quantiles, ...
    signed_log_clip_mean_fc, signed_log_clip_std_fc, signed_log_clip_abs_quantiles, clip_ratio_fc)
% 打印 signed-log 以及 signed-log + clip 后 abs(Y) 的分位数。
% 这里的 Y = sign(J).*log(1+abs(J)/alpha)，clip 在 Y 域执行。
fprintf('=== signed-log 后 abs(Y) 分位数统计，无 clip ===\n');
fprintf('%-10s %-12s %-12s %-12s', 'freqGHz', 'channel', 'log_mean', 'log_std');
for iq = 1:numel(transform_abs_quantile_probs)
    fprintf(' absY_q%-8.5g', transform_abs_quantile_probs(iq));
end
fprintf('\n');

[Fc, Cc] = size(signed_log_std_fc);
for kf = 1:Fc
    if numel(current_freq_hz) >= kf
        freq_ghz = current_freq_hz(kf) / 1e9;
    else
        freq_ghz = kf;
    end

    for cc = 1:Cc
        fprintf('%-10.6g %-12s %-12.4g %-12.4g', ...
            freq_ghz, channel_names{cc}, ...
            double(signed_log_mean_fc(kf, cc)), double(signed_log_std_fc(kf, cc)));
        for iq = 1:numel(transform_abs_quantile_probs)
            fprintf(' %-15.4g', double(signed_log_abs_quantiles(iq, kf, cc)));
        end
        fprintf('\n');
    end
end
fprintf('\n');

fprintf('=== signed-log + clip 后 abs(Y_clip) 分位数统计，T=%.6g ===\n', double(clip_T_fc(1,1)));
fprintf('%-10s %-12s %-12s %-12s %-12s', 'freqGHz', 'channel', 'clip_T', 'clip_mean', 'clip_std');
for iq = 1:numel(transform_abs_quantile_probs)
    fprintf(' absYc_q%-8.5g', transform_abs_quantile_probs(iq));
end
fprintf(' clip_ratio\n');

for kf = 1:Fc
    if numel(current_freq_hz) >= kf
        freq_ghz = current_freq_hz(kf) / 1e9;
    else
        freq_ghz = kf;
    end

    for cc = 1:Cc
        fprintf('%-10.6g %-12s %-12.4g %-12.4g %-12.4g', ...
            freq_ghz, channel_names{cc}, double(clip_T_fc(kf, cc)), ...
            double(signed_log_clip_mean_fc(kf, cc)), double(signed_log_clip_std_fc(kf, cc)));
        for iq = 1:numel(transform_abs_quantile_probs)
            fprintf(' %-15.4g', double(signed_log_clip_abs_quantiles(iq, kf, cc)));
        end
        fprintf(' %-12.4g\n', double(clip_ratio_fc(kf, cc)));
    end
end

fprintf('\n说明：clip_ratio = mean(abs(Y)>T)。若 q99/q999 基本不变、q1 被压到 T、clip_ratio 很低，说明 clip 主要处理极端异常点。\n\n');
end

function print_zscore_quantile_summary(current_freq_hz, channel_names, zscore_abs_quantile_probs, signed_log_mean_fc, signed_log_std_fc, zscore_abs_quantiles, underflow_counts, overflow_counts)
% 打印 signed-log + z-score 后 abs(Z) 的分位数。
% 这里不做 clip，因此这些分位数用于判断 transformed target 是否仍存在过大的标准化值。
fprintf('=== signed-log + z-score 后 abs(Z) 分位数统计，无 clip ===\n');
fprintf('%-10s %-12s %-12s %-12s', 'freqGHz', 'channel', 'log_mean', 'log_std');
for iq = 1:numel(zscore_abs_quantile_probs)
    fprintf(' absZ_q%-8.5g', zscore_abs_quantile_probs(iq));
end
fprintf(' under%-8s over%-8s\n', '', '');

[Fc, Cc] = size(signed_log_std_fc);
for kf = 1:Fc
    if numel(current_freq_hz) >= kf
        freq_ghz = current_freq_hz(kf) / 1e9;
    else
        freq_ghz = kf;
    end

    for cc = 1:Cc
        fprintf('%-10.6g %-12s %-12.4g %-12.4g', ...
            freq_ghz, channel_names{cc}, ...
            double(signed_log_mean_fc(kf, cc)), double(signed_log_std_fc(kf, cc)));
        for iq = 1:numel(zscore_abs_quantile_probs)
            fprintf(' %-15.4g', double(zscore_abs_quantiles(iq, kf, cc)));
        end
        fprintf(' %-12g %-12g\n', double(underflow_counts(kf, cc)), double(overflow_counts(kf, cc)));
    end
end

fprintf('\n建议观察：abs(Z) q99 < 4~5，q99.9 < 6~8，q99.99 < 8~10；若 max 长期 > 12~15，后续可考虑在 signed-log 域做 clip。\n\n');
end


function print_zscore_clip_quantile_summary(current_freq_hz, channel_names, zscore_abs_quantile_probs, signed_log_clip_mean_fc, signed_log_clip_std_fc, zscore_clip_abs_quantiles, underflow_counts, overflow_counts, clip_T)
% 打印 signed-log + clip + z-score 后 abs(Z_clip) 的分位数。
% z-score 的 mean/std 基于 clipped Y 重新计算，不使用 no-clip 的 mean/std。
fprintf('=== signed-log + clip + z-score 后 abs(Z_clip) 分位数统计，T=%.6g ===\n', clip_T);
fprintf('%-10s %-12s %-12s %-12s', 'freqGHz', 'channel', 'clip_mean', 'clip_std');
for iq = 1:numel(zscore_abs_quantile_probs)
    fprintf(' absZc_q%-8.5g', zscore_abs_quantile_probs(iq));
end
fprintf(' under%-8s over%-8s\n', '', '');

[Fc, Cc] = size(signed_log_clip_std_fc);
for kf = 1:Fc
    if numel(current_freq_hz) >= kf
        freq_ghz = current_freq_hz(kf) / 1e9;
    else
        freq_ghz = kf;
    end

    for cc = 1:Cc
        fprintf('%-10.6g %-12s %-12.4g %-12.4g', ...
            freq_ghz, channel_names{cc}, ...
            double(signed_log_clip_mean_fc(kf, cc)), double(signed_log_clip_std_fc(kf, cc)));
        for iq = 1:numel(zscore_abs_quantile_probs)
            fprintf(' %-15.4g', double(zscore_clip_abs_quantiles(iq, kf, cc)));
        end
        fprintf(' %-12g %-12g\n', double(underflow_counts(kf, cc)), double(overflow_counts(kf, cc)));
    end
end

fprintf('\n目标：clip 后 abs(Z_clip) 的 q99/q999 应基本接近 no-clip；q1 应被限制到约 (T-mu)/std，通常约 12~13。\n\n');
end

function show_zscore_histograms(current_freq_hz, channel_names, hist_bin_centers, zscore_hist_counts, underflow_counts, overflow_counts, zscore_clip_hist_counts, clip_underflow_counts, clip_overflow_counts, clip_T)
% 显示 signed-log + z-score 后的 signed Z histogram。
% 每个子图叠加 no clip 和 clip 后的曲线；范围外点用 under/over 计数显示在标题中。
fig = figure('Color', 'w', 'Name', 'signed_log_zscore_histograms');
tiledlayout(fig, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

Cc = numel(channel_names);
for cc = 1:Cc
    nexttile;
    counts = double(sum(zscore_hist_counts(:, :, cc), 2));
    counts_clip = double(sum(zscore_clip_hist_counts(:, :, cc), 2));
    underflow = double(sum(underflow_counts(:, cc), 1));
    overflow = double(sum(overflow_counts(:, cc), 1));
    underflow_clip = double(sum(clip_underflow_counts(:, cc), 1));
    overflow_clip = double(sum(clip_overflow_counts(:, cc), 1));

    hold on;
    stairs(hist_bin_centers, counts, 'LineWidth', 1.3, 'DisplayName', 'no clip');
    stairs(hist_bin_centers, counts_clip, 'LineWidth', 1.3, 'DisplayName', sprintf('clip T=%.3g', clip_T));
    hold off;
    set(gca, 'YScale', 'log');
    grid on;
    xlabel('Z');
    ylabel('count');
    title(sprintf('%s, noClip u/o=%g/%g, clip u/o=%g/%g', ...
        channel_names{cc}, underflow, overflow, underflow_clip, overflow_clip), 'Interpreter', 'none');
    legend('Location', 'best', 'Interpreter', 'none');
end

sgtitle(sprintf('signed-log + z-score histograms, %.6g-%.6g GHz, no clip vs clip', ...
    min(current_freq_hz) / 1e9, max(current_freq_hz) / 1e9), 'Interpreter', 'none');
drawnow;
end

function print_current_map_max_summary(current_map_max, quantile_probs)
% 不区分频点和通道，打印所有电流图 max(abs(current)) 的总体分位数摘要。
v = double(current_map_max(:));
v = v(isfinite(v));
q = percentile_vector(v, quantile_probs);

fprintf('=== 所有电流图 max(abs(current)) 的总体分位数统计 ===\n');
fprintf('统计对象数量：%d 张电流图\n', numel(v));
fprintf('%-12s', 'all_maps');
for iq = 1:numel(quantile_probs)
    fprintf(' q%-8.4g', quantile_probs(iq));
end
fprintf('\n');

fprintf('%-12s', 'max_abs');
for iq = 1:numel(quantile_probs)
    fprintf(' %-9.4g', q(iq));
end
fprintf('\n\n');
end

function show_global_histograms(current_freq_hz, channel_names, hist_bin_centers, hist_counts_global)
% 显示所有频点合并后的全局直方图。
fig = figure('Color', 'w', 'Name', 'global_histograms');
tiledlayout(fig, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

Cc = numel(channel_names);
for cc = 1:Cc
    nexttile;
    counts = double(sum(hist_counts_global(1:end-1, :, cc), 2));
    overflow = double(sum(hist_counts_global(end, :, cc), 2));

    stairs(hist_bin_centers, counts, 'LineWidth', 1.3);
    set(gca, 'YScale', 'log');
    grid on;
    xlabel('|current component|');
    ylabel('count');
    title(sprintf('%s, overflow=%g', channel_names{cc}, overflow), 'Interpreter', 'none');
end

sgtitle(sprintf('global histograms, %.6g-%.6g GHz', ...
    min(current_freq_hz) / 1e9, max(current_freq_hz) / 1e9), 'Interpreter', 'none');
drawnow;
end

function show_signed_log_scaled_histograms(current_freq_hz, channel_names, hist_bin_centers, signed_log_hist_counts, underflow_counts, overflow_counts, signed_log_clip_hist_counts, clip_underflow_counts, clip_overflow_counts, clip_T)
% 显示 signed_log(J / p95_abs) 后的 signed histogram。
% 每个子图叠加 clip 前后的曲线，用于观察 T 是否只影响极端尾部。
fig = figure('Color', 'w', 'Name', 'signed_log_scaled_histograms');
tiledlayout(fig, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

Cc = numel(channel_names);
for cc = 1:Cc
    nexttile;
    counts = double(sum(signed_log_hist_counts(:, :, cc), 2));
    counts_clip = double(sum(signed_log_clip_hist_counts(:, :, cc), 2));
    underflow = double(sum(underflow_counts(:, cc), 1));
    overflow = double(sum(overflow_counts(:, cc), 1));
    underflow_clip = double(sum(clip_underflow_counts(:, cc), 1));
    overflow_clip = double(sum(clip_overflow_counts(:, cc), 1));

    hold on;
    stairs(hist_bin_centers, counts, 'LineWidth', 1.3, 'DisplayName', 'no clip');
    stairs(hist_bin_centers, counts_clip, 'LineWidth', 1.3, 'DisplayName', sprintf('clip T=%.3g', clip_T));
    hold off;
    set(gca, 'YScale', 'log');
    grid on;
    xlabel('signed_log(J / p95(abs(J)))');
    ylabel('count');
    title(sprintf('%s, noClip u/o=%g/%g, clip u/o=%g/%g', ...
        channel_names{cc}, underflow, overflow, underflow_clip, overflow_clip), 'Interpreter', 'none');
    legend('Location', 'best', 'Interpreter', 'none');
end

sgtitle(sprintf('signed-log-scaled signed histograms, %.6g-%.6g GHz, no clip vs clip', ...
    min(current_freq_hz) / 1e9, max(current_freq_hz) / 1e9), 'Interpreter', 'none');
drawnow;
end

function show_channel_std_by_frequency(current_freq_hz, channel_names, raw_std_fc, signed_log_std_fc)
% 显示不同频点、不同通道的标准差。raw_std 看原始尺度；signed_log_std 看压缩后的尺度。
freq_ghz = current_freq_hz(:) / 1e9;
Cc = numel(channel_names);

fig = figure('Color', 'w', 'Name', 'channel_std_by_frequency');
tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile;
hold on;
for cc = 1:Cc
    plot(freq_ghz, double(raw_std_fc(:, cc)), '-o', 'LineWidth', 1.2, 'DisplayName', channel_names{cc});
end
hold off;
grid on;
set(gca, 'YScale', 'log');
xlabel('frequency (GHz)');
ylabel('std of signed raw J');
title('raw current std by channel', 'Interpreter', 'none');
legend('Location', 'best', 'Interpreter', 'none');

nexttile;
hold on;
for cc = 1:Cc
    plot(freq_ghz, double(signed_log_std_fc(:, cc)), '-o', 'LineWidth', 1.2, 'DisplayName', channel_names{cc});
end
hold off;
grid on;
xlabel('frequency (GHz)');
ylabel('std of signed_log(J / p95_abs)');
title('signed-log-scaled std by channel', 'Interpreter', 'none');
legend('Location', 'best', 'Interpreter', 'none');

drawnow;
end

function show_pixel_q99_mid_frequency(current_freq_hz, channel_names, pixel_q_mid, mid_f, pixel_q_prob, color_prob)
% 显示中间频点的逐像素 q99 图，色阶按配置分位数设置。
fig = figure('Color', 'w', 'Name', 'pixel_q99_mid_frequency');
tiledlayout(fig, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

Cc = numel(channel_names);
color_hi = robust_positive_limit(pixel_q_mid(:), color_prob);

for cc = 1:Cc
    nexttile;
    img = squeeze(pixel_q_mid(cc, :, :));
    imagesc(img);
    axis image tight;
    set(gca, 'YDir', 'normal');
    use_preferred_colormap(gca);
    clim([0 color_hi]);
    colorbar;
    title(sprintf('%s, q%.3g, %.6g GHz', ...
        channel_names{cc}, pixel_q_prob, current_freq_hz(mid_f) / 1e9), 'Interpreter', 'none');
    xlabel('x 网格');
    ylabel('y 网格');
end

sgtitle('pixel q99 at middle frequency', 'Interpreter', 'none');
drawnow;
end

function show_current_map_max_histogram(current_map_max, max_hist_bins, detail_limit)
% 不区分频点和通道，合并显示每张电流图 max(abs(current)) 的主体分布。
% 极端尾部会拉大等宽 bin，所以这里细看 [0, detail_limit]，并单独报告超过上限的数量。
v = double(current_map_max(:));
v = v(isfinite(v) & v > 0);

figure('Color', 'w', 'Name', 'current_map_max_all');
overflow_count = 0;
if isempty(v)
    histogram(0);
else
    in_range = v(v <= detail_limit);
    overflow_count = nnz(v > detail_limit);
    edges = linspace(0, detail_limit, max_hist_bins + 1);

    histogram(in_range, edges);
    xlim([0 detail_limit]);
end

grid on;
xlabel('max(abs(current)) per map');
ylabel('map count');
title(sprintf('all maps max(abs(current)) in [0,%g], N=%d, >%g: %d', ...
    detail_limit, numel(v), detail_limit, overflow_count), 'Interpreter', 'none');
drawnow;
end

function freq_idx = match_frequency_indices(current_freq_hz, target_freq_ghz)
% 为 8/10/12 GHz 找到数据集中最接近的频点。
freq_ghz = current_freq_hz(:) / 1e9;
freq_idx = zeros(size(target_freq_ghz));

for ii = 1:numel(target_freq_ghz)
    [delta, idx] = min(abs(freq_ghz - target_freq_ghz(ii)));
    assert(delta < 1e-6, ...
        '未找到 %.6g GHz；最近频点为 %.6g GHz。', ...
        target_freq_ghz(ii), freq_ghz(idx));
    freq_idx(ii) = idx;
end
end

function show_jx_real_figures(h5path, plot_idx, freq_idx, current_freq_hz, jx_real_channel, fineN, color_prob)
% 每个随机样本开一个窗口：上排线性尺度，下排带符号的对数压缩尺度。
for is = 1:numel(plot_idx)
    sample_id = plot_idx(is);

    fig = figure('Color', 'w', 'Name', sprintf('sample %d Jx_real', sample_id));
    tiledlayout(fig, 2, numel(freq_idx), 'TileSpacing', 'compact', 'Padding', 'compact');

    jx_stack = zeros(numel(freq_idx), fineN, fineN, 'single');
    for ii = 1:numel(freq_idx)
        raw = h5read(h5path, '/current', ...
            double([sample_id freq_idx(ii) jx_real_channel 1 1]), ...
            double([1 1 1 fineN fineN]));
        jx_stack(ii, :, :) = reshape(single(raw), [fineN, fineN]);
    end

    linear_lim = robust_positive_limit(abs(jx_stack(:)), color_prob);
    log_ref = max(linear_lim, eps);
    log_stack = signed_log_image(jx_stack, log_ref);
    log_lim = robust_positive_limit(abs(log_stack(:)), color_prob);

    for ii = 1:numel(freq_idx)
        freq_ghz = current_freq_hz(freq_idx(ii)) / 1e9;
        img = squeeze(jx_stack(ii, :, :));

        nexttile(ii);
        imagesc(img);
        axis image tight;
        set(gca, 'YDir', 'normal');
        use_preferred_colormap(gca);
        clim([-linear_lim linear_lim]);
        colorbar;
        title(sprintf('样本 %d, %.6g GHz, 线性', sample_id, freq_ghz), 'Interpreter', 'none');
        xlabel('x 网格');
        ylabel('y 网格');

        nexttile(numel(freq_idx) + ii);
        imagesc(signed_log_image(img, log_ref));
        axis image tight;
        set(gca, 'YDir', 'normal');
        use_preferred_colormap(gca);
        clim([-log_lim log_lim]);
        colorbar;
        title(sprintf('样本 %d, %.6g GHz, signed log', sample_id, freq_ghz), 'Interpreter', 'none');
        xlabel('x 网格');
        ylabel('y 网格');
    end

    sgtitle(sprintf('Jx_real 电流分布：样本 %d', sample_id), 'Interpreter', 'none');
    drawnow;
end
end



function y = clip_symmetric(x, T)
% 对 transformed-domain 数据做对称硬截断。
% y = min(max(x, -T), T)。
y = min(max(x, -T), T);
end

function y = signed_log_transform(x, alpha)
% 对 signed current 做自然对数形式的 signed-log 压缩：
% y = sign(x) .* log(1 + abs(x) ./ alpha)。
% 这里使用 natural log：y = sign(x) .* log(1 + abs(x)/alpha)。
y = sign(x) .* log(1 + abs(x) ./ max(alpha, eps));
end

function img_log = signed_log_image(img, ref_value)
% 对有正负号的 Jx_real 做 signed-log 压缩，同时保留电流方向符号。
img_log = signed_log_transform(img, ref_value);
end

function lim = robust_positive_limit(v, prob)
% 用分位数作为图像色阶上限；prob=1 时等价于最大值，不做显示截断。
v = double(v(:));
v = abs(v(isfinite(v)));
v = v(v > 0);
if isempty(v)
    lim = 1;
else
    lim = percentile_vector(v, prob);
    if lim <= 0 || ~isfinite(lim)
        lim = max(v);
    end
    if lim <= 0 || ~isfinite(lim)
        lim = 1;
    end
end
end

function use_preferred_colormap(ax)
% turbo 在较新 MATLAB 中可用；不可用时退回 parula。
try
    colormap(ax, 'turbo');
catch
    colormap(ax, 'parula');
end
end

function tf = dataset_exists(h5path, dset_name)
% 安静地判断 HDF5 数据集是否存在。
try
    h5info(h5path, dset_name);
    tf = true;
catch
    tf = false;
end
end
