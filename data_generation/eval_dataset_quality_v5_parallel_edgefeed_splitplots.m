function eval_dataset_quality_v5_parallel_edgefeed_splitplots()
% =========================================================================
% eval_dataset_quality_v5_parallel_edgefeed_splitplots.m
%（该代码最后两张图不参考）
% 修改说明：
%   1) 支持并行加速（parfor）：
%      - 带宽统计
%      - 抽样 pair 距离
%      - 最近邻距离
%
%   2) /pattern 与 /current 改为延迟读取（lazy read）
%      - 只在最后真正需要展示 Top1 时读取单样本/单频点切片
%
%   3) 删除 hf_score / score_hf 排序逻辑
%      - Top 样本展示直接取前 num_show_top 个样本
%
%   4) 新增边缘馈电点筛选
%      - 仅对"馈电点靠近边缘"的样本绘制：
%           a) 中心频率 vs 带宽
%           b) 带宽统计图
%      - 并分别绘制 random / optimized 两组
%
% 必需字段：
%   /X           [B,1,N,N]   0=air, 1=metal, 2=feed
%   /Y           [B,F]
%   /freq_hz     [F,1]
%
% 可选字段：
%   /pattern           [B,Fp,P,T]
%   /pattern_freq_hz   [Fp,1]
%   /pattern_theta_deg [T,1]
%   /current           [B,Fc,4,fineN,fineN]
%   /current_freq_hz   [Fc,1]
%   /feed_rc           [B,2]
%   /sample_src        [B,1]
%   /cost              [B,1]
%   /lf_cost           [B,1]
%   /raw_fill          [B,1]
%   /final_fill        [B,1]
% =========================================================================

%% ------------------------------------------------------------------------
% 0) 配置
% -------------------------------------------------------------------------
h5path = 'dataset_out\antenna_dataset_20260408_150016.h5';
assert(isfile(h5path), '文件不存在：%s', h5path);

% 评估前 N 个样本；设为 inf 表示全量
N_eval = inf;

% -10 dB 带宽阈值
thr_db = -10;

% 近重复率抽样 pair 数
max_pairs = 20000;

% 近重复率阈值（加权海明距离）
tau = 0.10;

% 距离权重
w_pix  = 1.0;
w_feed = 1.0;

% Top 样本展示个数
num_show_top = 16;

% 最近邻距离精确计算阈值：若 B 太大，用抽样近邻替代
max_exact_nn = 2500;

% -------- 边缘馈电点配置 --------
% 判定规则：
%   馈电点到上/下/左/右任意一条边界的最小像素距离 <= edge_margin_pix
%   即认为是"比较边缘的位置"
edge_margin_pix = 8;

% -------- Top1 电流可视化配置 --------
% [] 表示自动使用 Top1 的谐振频率 f0 对应的最近 current 频点
current_target_freq_GHz = [];

% 自动读取失败时的后备几何参数（单位 mm）
default_geom.patch_L_mm = 14;
default_geom.patch_W_mm = 14;
default_geom.sub_thick_mm = 2.5;
default_geom.board_L_mm = 30;
default_geom.board_W_mm = 30;
default_geom.feed_diam_mm = min(default_geom.patch_L_mm/16, default_geom.patch_W_mm/16) / 10;
default_geom.substrate_name = 'Air';

% 并行配置
use_parallel = true;
num_workers = [];   % [] 表示默认 parpool 配置

% 大字段延迟读取
lazy_read_pattern = true;
lazy_read_current = true;

fprintf('读取数据：%s\n', h5path);

%% ------------------------------------------------------------------------
% 1) 读取数据
% -------------------------------------------------------------------------
freq = h5read(h5path, '/freq_hz');
freq = double(freq(:));
F = double(numel(freq));

infoX = h5info(h5path, '/X');
xsize = double(infoX.Dataspace.Size);

B_all = xsize(1);
C     = xsize(2);
H     = xsize(3);
W     = xsize(4);

assert(C == 1, 'X 通道数应为 1（0=air, 1=metal, 2=feed）');

if isinf(N_eval)
    B = B_all;
else
    B = min(B_all, double(N_eval));
end

% 必需字段先读入
X = h5read(h5path, '/X', double([1 1 1 1]), double([B C H W]));
Y = h5read(h5path, '/Y', double([1 1]),     double([B F]));

% 可选字段存在性
has_pattern           = dataset_exists(h5path, '/pattern');
has_pattern_freq_hz   = dataset_exists(h5path, '/pattern_freq_hz');
has_pattern_theta_deg = dataset_exists(h5path, '/pattern_theta_deg');
has_current           = dataset_exists(h5path, '/current');
has_current_freq_hz   = dataset_exists(h5path, '/current_freq_hz');
has_feed_rc           = dataset_exists(h5path, '/feed_rc');
has_sample_src        = dataset_exists(h5path, '/sample_src');
has_cost              = dataset_exists(h5path, '/cost');
has_lf_cost           = dataset_exists(h5path, '/lf_cost');
has_raw_fill          = dataset_exists(h5path, '/raw_fill');
has_final_fill        = dataset_exists(h5path, '/final_fill');
has_meta_json         = attribute_exists(h5path, '/', 'meta_json');

% pattern：默认只读频率和角度元信息，不整块读 pattern
pattern = [];
pattern_freq = [];
pattern_theta_deg = [];
Fp = 0; P = 0; T = 0;

if has_pattern
    infoP = h5info(h5path, '/pattern');
    psize = double(infoP.Dataspace.Size);   % [B, Fp, P, T]

    Bp = min(B, psize(1));
    Fp = psize(2);
    P  = psize(3);
    T  = psize(4);

    if has_pattern_freq_hz
        pattern_freq = double(h5read(h5path, '/pattern_freq_hz'));
        pattern_freq = pattern_freq(:);
    else
        pattern_freq = (1:Fp).';
    end

    if has_pattern_theta_deg
        pattern_theta_deg = double(h5read(h5path, '/pattern_theta_deg'));
        pattern_theta_deg = pattern_theta_deg(:);
    else
        pattern_theta_deg = (0:T-1).';
    end

    if ~lazy_read_pattern
        pattern = h5read(h5path, '/pattern', ...
            double([1 1 1 1]), double([Bp Fp P T]));
        pattern = double(pattern);
    end
else
    pattern = [];
    pattern_freq = [];
    pattern_theta_deg = [];
end

% current：默认只读频点元信息，不整块读 current
current_data = [];
current_freq = [];
fineN = NaN;
Fc = 0; Cc = 0;

if has_current
    infoC = h5info(h5path, '/current');
    csize = double(infoC.Dataspace.Size);   % [B, Fc, Cc, fineN, fineN]

    Bc    = min(B, csize(1));
    Fc    = csize(2);
    Cc    = csize(3);
    fineN = csize(4);
    fineN2 = csize(5);

    assert(Cc == 4, 'current 的 channel 数应为 4。');
    assert(fineN == fineN2, 'current 网格应为正方形。');

    if has_current_freq_hz
        current_freq = double(h5read(h5path, '/current_freq_hz'));
        current_freq = current_freq(:);
    else
        current_freq = (1:Fc).';
    end

    if ~lazy_read_current
        current_data = h5read(h5path, '/current', ...
            double([1 1 1 1 1]), double([Bc Fc Cc fineN fineN]));
        current_data = double(current_data);
    end
else
    current_data = [];
    current_freq = [];
    fineN = NaN;
end

if has_feed_rc
    feed_rc = double(h5read(h5path, '/feed_rc', double([1 1]), double([B 2])));
else
    feed_rc = [];
end

if has_sample_src
    sample_src = double(h5read(h5path, '/sample_src', double([1 1]), double([B 1])));
else
    sample_src = nan(B,1);
end

if has_cost
    cost = double(h5read(h5path, '/cost', double([1 1]), double([B 1])));
    cost_name = 'cost';
elseif has_lf_cost
    cost = double(h5read(h5path, '/lf_cost', double([1 1]), double([B 1])));
    cost_name = 'lf_cost';
else
    cost = nan(B,1);
    cost_name = 'cost';
end

if has_raw_fill
    raw_fill = double(h5read(h5path, '/raw_fill', double([1 1]), double([B 1])));
else
    raw_fill = nan(B,1);
end

if has_final_fill
    final_fill = double(h5read(h5path, '/final_fill', double([1 1]), double([B 1])));
else
    final_fill = nan(B,1);
end

meta = struct();
if has_meta_json
    meta_json = h5readatt(h5path, '/', 'meta_json');
    meta = safe_jsondecode(meta_json);
end

fprintf('读取完成：B=%d, F=%d, H=%d, W=%d\n', B, F, H, W);
fprintf('X 编码：单通道，0=air, 1=metal, 2=feed\n');

if has_current
    fprintf('检测到 current：Fc=%d, fineN=%d\n', numel(current_freq), fineN);
else
    fprintf('未检测到 /current，Top1 电流可视化将跳过。\n');
end

if has_cost
    fprintf('检测到统一代价字段：/cost\n');
elseif has_lf_cost
    fprintf('未检测到 /cost，回退使用旧字段：/lf_cost\n');
else
    fprintf('未检测到代价字段：/cost 或 /lf_cost\n');
end

% 启动并行池
if use_parallel
    p = gcp('nocreate');
    if isempty(p)
        if isempty(num_workers)
            parpool;
        else
            parpool(num_workers);
        end
    end
end

%% ------------------------------------------------------------------------
% 2) 从单通道 X 提取结构与馈电位置
%    X: [B,1,H,W], 0=air, 1=metal, 2=feed
% -------------------------------------------------------------------------
X1 = squeeze(X(:,1,:,:));   % [B,H,W]

% 金属区域：值为 1 或 2 都算金属
pix  = (X1 > 0.5);          % [B,H,W]

% 馈电位置：只有值为 2 的位置才算馈点
feed = (X1 > 1.5);          % [B,H,W]

pix_vec  = reshape(pix,  B, H*W);
feed_vec = reshape(feed, B, H*W);

feed_count = sum(feed_vec, 2);
if any(feed_count ~= 1)
    warning('检测到部分样本馈电标记数量不等于 1；将优先使用 /feed_rc（若存在），否则使用 X 中第一个最大值位置。');
end

[~, feed_lin] = max(feed_vec, [], 2);
feed_r_from_X = mod(feed_lin - 1, H) + 1;
feed_c_from_X = floor((feed_lin - 1) / H) + 1;

if isempty(feed_rc)
    feed_rc = [feed_r_from_X, feed_c_from_X];
end

%% ------------------------------------------------------------------------
% 2.5) 边缘馈电点样本筛选
% -------------------------------------------------------------------------
feed_r = feed_rc(:,1);
feed_c = feed_rc(:,2);

dist_to_top    = feed_r - 1;
dist_to_bottom = H - feed_r;
dist_to_left   = feed_c - 1;
dist_to_right  = W - feed_c;

dist_to_edge = min([dist_to_top, dist_to_bottom, dist_to_left, dist_to_right], [], 2);
idx_edge_feed = (dist_to_edge <= edge_margin_pix);

fprintf('\n[边缘馈电点筛选]\n');
fprintf('edge_margin_pix = %d\n', edge_margin_pix);
fprintf('边缘馈电点样本数 = %d / %d (%.2f%%)\n', ...
    sum(idx_edge_feed), B, 100 * mean(idx_edge_feed));

%% ------------------------------------------------------------------------
% 3) S11 基础指标
% -------------------------------------------------------------------------
Y_mag = double(Y);
Y_db  = 20 * log10(Y_mag + eps);

[min_db, idx_min] = min(Y_db, [], 2);
f0 = freq(idx_min);

bw10   = zeros(B,1);
f10_lo = nan(B,1);
f10_hi = nan(B,1);
num_bands = zeros(B,1);

if use_parallel
    parfor i = 1:B
        mask = (Y_db(i,:) <= thr_db);
        if any(mask)
            d = diff([false, mask, false]);
            run_start = find(d == 1);
            run_end   = find(d == -1) - 1;

            nb = numel(run_start);

            [~, idx] = max(run_end - run_start + 1);
            s1 = run_start(idx);
            e1 = run_end(idx);

            num_bands(i) = nb;
            f10_lo(i) = freq(s1);
            f10_hi(i) = freq(e1);
            bw10(i)   = f10_hi(i) - f10_lo(i);
        else
            num_bands(i) = 0;
            f10_lo(i) = nan;
            f10_hi(i) = nan;
            bw10(i) = 0;
        end
    end
else
    for i = 1:B
        mask = (Y_db(i,:) <= thr_db);
        if any(mask)
            d = diff([false, mask, false]);
            run_start = find(d == 1);
            run_end   = find(d == -1) - 1;

            num_bands(i) = numel(run_start);

            [~, idx] = max(run_end - run_start + 1);
            s1 = run_start(idx);
            e1 = run_end(idx);

            f10_lo(i) = freq(s1);
            f10_hi(i) = freq(e1);
            bw10(i)   = f10_hi(i) - f10_lo(i);
        end
    end
end

bw10_GHz = bw10 / 1e9;
f0_GHz   = f0   / 1e9;

%% ------------------------------------------------------------------------
% 4) 多样性评估：近重复率 + 最近邻距离
% -------------------------------------------------------------------------
fprintf('\n开始多样性评估...\n');

rng(42);

num_pairs = min(max_pairs, floor(B*(B-1)/2));
pairs_i = randi(B, num_pairs, 1);
pairs_j = randi(B-1, num_pairs, 1);
pairs_j = pairs_j + (pairs_j >= pairs_i);

d_list = zeros(num_pairs,1);

if use_parallel
    parfor t = 1:num_pairs
        i = pairs_i(t);
        j = pairs_j(t);

        dH_pix  = mean(xor(pix_vec(i,:),  pix_vec(j,:)));
        dH_feed = mean(xor(feed_vec(i,:), feed_vec(j,:)));
        d_list(t) = w_pix*dH_pix + w_feed*dH_feed;
    end
else
    for t = 1:num_pairs
        i = pairs_i(t);
        j = pairs_j(t);

        dH_pix  = mean(xor(pix_vec(i,:),  pix_vec(j,:)));
        dH_feed = mean(xor(feed_vec(i,:), feed_vec(j,:)));
        d_list(t) = w_pix*dH_pix + w_feed*dH_feed;
    end
end

near_cnt = sum(d_list < tau);
dup_ratio = near_cnt / max(num_pairs, 1);
min_d_est = min(d_list);
mean_d_est = mean(d_list);

if B <= max_exact_nn
    fprintf('  样本数不大，进行精确最近邻距离计算...\n');
    nn_dist = exact_nearest_neighbor_distance(pix_vec, feed_vec, w_pix, w_feed, use_parallel);
else
    fprintf('  样本数较大，采用抽样方式估计最近邻距离...\n');
    nn_dist = approximate_nearest_neighbor_distance(pix_vec, feed_vec, w_pix, w_feed, 256, use_parallel);
end

%% ------------------------------------------------------------------------
% 5) 馈电位置分布
% -------------------------------------------------------------------------
feed_map = accumarray(feed_rc, 1, [H, W]);
counts = feed_map(:);
p = counts / sum(counts);
p = p(p > 0);

feed_entropy = -sum(p .* log(p));
feed_entropy_max = log(numel(feed_map));
feed_entropy_ratio = feed_entropy / feed_entropy_max;
feed_KL_uniform = sum(p .* log(p * numel(feed_map)));

%% ------------------------------------------------------------------------
% 6) 分组统计：optimized vs random
% -------------------------------------------------------------------------
has_valid_group = any(~isnan(sample_src));
if has_valid_group
    idx_rand = (sample_src == 0);
    idx_opt  = (sample_src == 1);
else
    idx_rand = false(B,1);
    idx_opt  = false(B,1);
end

% 边缘馈电点下的 random / optimized 子集
if has_valid_group
    idx_rand_edge = idx_rand & idx_edge_feed;
    idx_opt_edge  = idx_opt  & idx_edge_feed;
else
    idx_rand_edge = false(B,1);
    idx_opt_edge  = false(B,1);
end

%% ------------------------------------------------------------------------
% 7) raw/final fill 分析
% -------------------------------------------------------------------------
if has_raw_fill && has_final_fill
    delta_fill = final_fill - raw_fill;
else
    delta_fill = nan(B,1);
end

%% ------------------------------------------------------------------------
% 8) cost 与 HF 指标一致性分析
% -------------------------------------------------------------------------
valid_cost = ~isnan(cost) & isfinite(cost);

if any(valid_cost)
    [rho_cost_minS11, p1] = corr(cost(valid_cost), min_db(valid_cost), ...
        'type','Spearman', 'rows','complete');
    [rho_cost_bw, p2] = corr(cost(valid_cost), bw10_GHz(valid_cost), ...
        'type','Spearman', 'rows','complete');
else
    rho_cost_minS11 = nan; p1 = nan;
    rho_cost_bw     = nan; p2 = nan;
end

%% ------------------------------------------------------------------------
% 9) 文本输出
% -------------------------------------------------------------------------
fprintf('\n============================================================\n');
fprintf('数据集总体评估结果\n');
fprintf('============================================================\n');
fprintf('样本数 B = %d, 频点数 F = %d, 尺寸 = %dx%d\n', B, F, H, W);
fprintf('freq range = [%.3f, %.3f] GHz\n', freq(1)/1e9, freq(end)/1e9);

fprintf('\n[总体 S11 / 带宽]\n');
fprintf('min(S11)_dB:\n');
fprintf('  mean = %.2f, median = %.2f, P10 = %.2f, P90 = %.2f, best = %.2f\n', ...
    mean(min_db), median(min_db), prctile(min_db,10), prctile(min_db,90), min(min_db));

fprintf('-10 dB longest BW (GHz):\n');
fprintf('  mean = %.3f, median = %.3f, P90 = %.3f, max = %.3f\n', ...
    mean(bw10_GHz), median(bw10_GHz), prctile(bw10_GHz,90), max(bw10_GHz));

fprintf('resonant f0 (GHz): mean = %.3f, median = %.3f\n', ...
    mean(f0_GHz), median(f0_GHz));

fprintf('min(S11) < -10 dB ratio = %.2f%%\n', 100 * mean(min_db < -10));
fprintf('nonzero BW ratio        = %.2f%%\n', 100 * mean(bw10 > 0));
fprintf('multiband ratio         = %.2f%%\n', 100 * mean(num_bands >= 2));

fprintf('\n[多样性]\n');
fprintf('dup_ratio (tau=%.3f, pairs=%d) = %.4f\n', tau, num_pairs, dup_ratio);
fprintf('sampled distance: mean = %.4f, min = %.4f\n', mean_d_est, min_d_est);
fprintf('nearest-neighbor distance:\n');
fprintf('  mean = %.4f, median = %.4f, min = %.4f, max = %.4f\n', ...
    mean(nn_dist), median(nn_dist), min(nn_dist), max(nn_dist));

fprintf('\n[馈电分布]\n');
fprintf('feed entropy = %.4f / %.4f (%.2f%% of max)\n', ...
    feed_entropy, feed_entropy_max, 100*feed_entropy_ratio);
fprintf('KL(p || uniform) = %.4f\n', feed_KL_uniform);

if has_valid_group
    fprintf('\n[按 sample_src 分组]\n');
    print_group_stats('optimized', idx_opt, min_db, bw10_GHz, nn_dist, raw_fill, final_fill);
    print_group_stats('random   ', idx_rand, min_db, bw10_GHz, nn_dist, raw_fill, final_fill);

    fprintf('\n[仅边缘馈电点样本分组]\n');
    print_group_stats('optimized_edge', idx_opt_edge, min_db, bw10_GHz, nn_dist, raw_fill, final_fill);
    print_group_stats('random_edge   ', idx_rand_edge, min_db, bw10_GHz, nn_dist, raw_fill, final_fill);
end

if has_raw_fill && has_final_fill
    fprintf('\n[fill 变化]\n');
    fprintf('raw_fill   : mean=%.4f, median=%.4f\n', mean(raw_fill), median(raw_fill));
    fprintf('final_fill : mean=%.4f, median=%.4f\n', mean(final_fill), median(final_fill));
    fprintf('delta_fill : mean=%.4f, median=%.4f, P10=%.4f, P90=%.4f\n', ...
        mean(delta_fill), median(delta_fill), prctile(delta_fill,10), prctile(delta_fill,90));
end

if any(valid_cost)
    fprintf('\n[cost vs HF 指标一致性]\n');
    fprintf('Spearman(%s, min_db)    = %.4f, p = %.3g\n', cost_name, rho_cost_minS11, p1);
    fprintf('Spearman(%s, bw10_GHz)  = %.4f, p = %.3g\n', cost_name, rho_cost_bw, p2);
    fprintf('注：若 cost 越小越好，则通常希望它与 min_db 正相关、与带宽负相关。\n');
end

fprintf('============================================================\n');

%% ------------------------------------------------------------------------
% 10) 可视化
% -------------------------------------------------------------------------
figure('Name','Min S11 Histogram','Color','w');
histogram(min_db, 30);
grid on;
xlabel('min S11 (dB)');
ylabel('count');
title(sprintf('Min S11 distribution | median = %.2f dB', median(min_db)));

% 带宽统计图：只展示边缘馈电点样本，并按 random / optimized 拆分
if has_valid_group
    if any(idx_rand_edge)
        figure('Name','Bandwidth Histogram - Random ','Color','w');
        histogram(bw10_GHz(idx_rand_edge), 30);
        grid on;
        xlabel('Longest -10 dB BW (GHz)');
        ylabel('count');
        title(sprintf('Random  | BW distribution | n=%d | median=%.3f GHz', ...
            sum(idx_rand_edge), median(bw10_GHz(idx_rand_edge))));
    end

    if any(idx_opt_edge)
        figure('Name','Bandwidth Histogram - Optimized','Color','w');
        histogram(bw10_GHz(idx_opt_edge), 30);
        grid on;
        xlabel('Longest -10 dB BW (GHz)');
        ylabel('count');
        title(sprintf('Optimized | BW distribution | n=%d | median=%.3f GHz', ...
            sum(idx_opt_edge), median(bw10_GHz(idx_opt_edge))));
    end

    if ~any(idx_rand_edge) && ~any(idx_opt_edge)
        warning('没有满足"边缘馈电点"条件的 random/optimized 样本，带宽图未绘制。');
    end
else
    warning('sample_src 不可用，无法区分 random / optimized。');
end

figure('Name','Min S11 vs BW','Color','w');
if has_valid_group
    hold on;
    scatter(min_db(idx_rand), bw10_GHz(idx_rand), 16, 'filled', 'MarkerFaceAlpha', 0.35);
    scatter(min_db(idx_opt),  bw10_GHz(idx_opt),  16, 'filled', 'MarkerFaceAlpha', 0.35);
    hold off;
    legend({'random','optimized'}, 'Location','best');
else
    scatter(min_db, bw10_GHz, 16, 'filled', 'MarkerFaceAlpha', 0.35);
end
grid on;
xlabel('min S11 (dB)');
ylabel('Longest -10 dB BW (GHz)');
title('HF performance map');

% 中心频率 vs 带宽：只展示边缘馈电点样本，并按 random / optimized 拆分
if has_valid_group
    if any(idx_rand_edge)
        figure('Name','Center Frequency vs BW - Random','Color','w');
        scatter(f0_GHz(idx_rand_edge), bw10_GHz(idx_rand_edge), 16, 'filled', 'MarkerFaceAlpha', 0.35);
        grid on;
        xlabel('Resonant f_0 (GHz)');
        ylabel('Longest -10 dB BW (GHz)');
        title(sprintf('Random | center frequency vs bandwidth | n=%d', ...
            sum(idx_rand_edge)));
    end

    if any(idx_opt_edge)
        figure('Name','Center Frequency vs BW - Optimized','Color','w');
        scatter(f0_GHz(idx_opt_edge), bw10_GHz(idx_opt_edge), 16, 'filled', 'MarkerFaceAlpha', 0.35);
        grid on;
        xlabel('Resonant f_0 (GHz)');
        ylabel('Longest -10 dB BW (GHz)');
        title(sprintf('Optimized | center frequency vs bandwidth | n=%d', ...
            sum(idx_opt_edge)));
    end

    if ~any(idx_rand_edge) && ~any(idx_opt_edge)
        warning('没有满足"边缘馈电点"条件的 random/optimized 样本，中心频率 vs 带宽图未绘制。');
    end
else
    warning('sample_src 不可用，无法区分 random / optimized。');
end

figure('Name','Feed Position Heatmap','Color','w');
imagesc(feed_map);
axis image;
set(gca, 'YDir', 'normal');
colorbar;
xlabel('column');
ylabel('row');
title('Feed Position Heatmap');

figure('Name','Distance Statistics','Color','w');
tiledlayout(1,2,'Padding','compact','TileSpacing','compact');

nexttile;
histogram(d_list, 30);
grid on;
xlabel('sampled pair distance');
ylabel('count');
title(sprintf('Pair distance | dup ratio = %.4f', dup_ratio));

nexttile;
histogram(nn_dist, 30);
grid on;
xlabel('nearest-neighbor distance');
ylabel('count');
title('Nearest-neighbor distance');

if has_raw_fill && has_final_fill
    figure('Name','Fill Analysis','Color','w');
    tiledlayout(1,3,'Padding','compact','TileSpacing','compact');

    nexttile;
    histogram(raw_fill, 25);
    grid on;
    xlabel('raw fill');
    ylabel('count');
    title('Raw fill');

    nexttile;
    histogram(final_fill, 25);
    grid on;
    xlabel('final fill');
    ylabel('count');
    title('Final fill');

    nexttile;
    histogram(delta_fill, 25);
    grid on;
    xlabel('final fill - raw fill');
    ylabel('count');
    title('Morphology fill shift');
end

if any(valid_cost)
    figure('Name','Cost vs HF','Color','w');
    tiledlayout(1,2,'Padding','compact','TileSpacing','compact');

    nexttile;
    scatter(cost(valid_cost), min_db(valid_cost), 16, 'filled', 'MarkerFaceAlpha', 0.35);
    grid on;
    xlabel(cost_name);
    ylabel('HF min S11 (dB)');
    title(sprintf('\\rho = %.3f', rho_cost_minS11));

    nexttile;
    scatter(cost(valid_cost), bw10_GHz(valid_cost), 16, 'filled', 'MarkerFaceAlpha', 0.35);
    grid on;
    xlabel(cost_name);
    ylabel('HF BW (GHz)');
    title(sprintf('\\rho = %.3f', rho_cost_bw));
end

if has_valid_group && any(idx_rand) && any(idx_opt)
    figure('Name','Optimized vs Random','Color','w');
    tiledlayout(1,2,'Padding','compact','TileSpacing','compact');

    nexttile;
    boxchart(categorical(sample_src), min_db);
    grid on;
    xlabel('sample src (0=random, 1=optimized)');
    ylabel('min S11 (dB)');
    title('Min S11 by source');

    nexttile;
    boxchart(categorical(sample_src), bw10_GHz);
    grid on;
    xlabel('sample src');
    ylabel('BW (GHz)');
    title('BW by source');
end

%% ------------------------------------------------------------------------
% 11) Top 样本索引（不排序，直接取前 num_show_top 个）
% -------------------------------------------------------------------------
top_idx = 1:min(num_show_top, B);

% 如需恢复 Top 结构图、S11 曲线图、pattern 图，可取消下面相关代码注释
% top_i = top_idx(1);

%% ------------------------------------------------------------------------
% 12) Top16 文本摘要（不排序）
% -------------------------------------------------------------------------
fprintf('\n============================================================\n');
fprintf('前 %d 个样本摘要（按原始顺序，不排序）\n', numel(top_idx));
fprintf('============================================================\n');

for k = 1:numel(top_idx)
    i = top_idx(k);

    if has_valid_group
        if sample_src(i) == 1
            src_str = 'optimized';
        elseif sample_src(i) == 0
            src_str = 'random';
        else
            src_str = 'unknown';
        end
    else
        src_str = 'unknown';
    end

    fprintf(['No.%2d | idx=%6d | src=%-9s | feed=(%2d,%2d) | ' ...
             'edgeDist=%2d | minS11=%7.2f dB | f0=%6.3f GHz | BW=%6.3f GHz'], ...
            k, i, src_str, feed_rc(i,1), feed_rc(i,2), dist_to_edge(i), ...
            min_db(i), f0_GHz(i), bw10_GHz(i));

    if ~isnan(cost(i))
        fprintf(' | %s=%8.3f', cost_name, cost(i));
    end
    if has_final_fill && ~isnan(final_fill(i))
        fprintf(' | fill=%.3f', final_fill(i));
    end
    fprintf('\n');
end

fprintf('============================================================\n');

end

%% =========================================================================
%% 辅助函数
%% =========================================================================
function tf = dataset_exists(h5path, dset_name)
tf = false;
info = h5info(h5path);
tf = search_group_for_dataset(info, dset_name);
end

function tf = attribute_exists(h5path, obj_name, attr_name)
tf = false;
try
    info = h5info(h5path, obj_name);
    attr_names = {info.Attributes.Name};
    tf = any(strcmp(attr_names, attr_name));
catch
    tf = false;
end
end

function tf = search_group_for_dataset(group_info, dset_name)
tf = false;

for k = 1:numel(group_info.Datasets)
    this_name = ['/' group_info.Datasets(k).Name];
    if strcmp(this_name, dset_name)
        tf = true;
        return;
    end
end

for g = 1:numel(group_info.Groups)
    tf = search_group_for_dataset(group_info.Groups(g), dset_name);
    if tf
        return;
    end
end
end

function nn_dist = exact_nearest_neighbor_distance(pix_vec, feed_vec, w_pix, w_feed, use_parallel)
B = size(pix_vec,1);
nn_dist = inf(B,1);

if nargin < 5
    use_parallel = false;
end

if use_parallel
    parfor i = 1:B
        dH_pix  = mean(xor(pix_vec,  pix_vec(i,:)), 2);
        dH_feed = mean(xor(feed_vec, feed_vec(i,:)), 2);
        d = w_pix*dH_pix + w_feed*dH_feed;
        d(i) = inf;
        nn_dist(i) = min(d);
    end
else
    for i = 1:B
        dH_pix  = mean(xor(pix_vec,  pix_vec(i,:)), 2);
        dH_feed = mean(xor(feed_vec, feed_vec(i,:)), 2);
        d = w_pix*dH_pix + w_feed*dH_feed;
        d(i) = inf;
        nn_dist(i) = min(d);
    end
end
end

function nn_dist = approximate_nearest_neighbor_distance(pix_vec, feed_vec, w_pix, w_feed, K, use_parallel)
B = size(pix_vec,1);
nn_dist = inf(B,1);

if nargin < 6
    use_parallel = false;
end

cand_mat = zeros(B, min(K, max(B-1,1)));

for i = 1:B
    cand = randperm(B, min(K+1, B));
    cand(cand == i) = [];
    cand = cand(1:min(K, numel(cand)));
    cand_mat(i,1:numel(cand)) = cand;
end

if use_parallel
    parfor i = 1:B
        cand = cand_mat(i,:);
        cand = cand(cand > 0);

        dH_pix  = mean(xor(pix_vec(cand,:),  pix_vec(i,:)), 2);
        dH_feed = mean(xor(feed_vec(cand,:), feed_vec(i,:)), 2);
        d = w_pix*dH_pix + w_feed*dH_feed;
        nn_dist(i) = min(d);
    end
else
    for i = 1:B
        cand = cand_mat(i,:);
        cand = cand(cand > 0);

        dH_pix  = mean(xor(pix_vec(cand,:),  pix_vec(i,:)), 2);
        dH_feed = mean(xor(feed_vec(cand,:), feed_vec(i,:)), 2);
        d = w_pix*dH_pix + w_feed*dH_feed;
        nn_dist(i) = min(d);
    end
end
end

function print_group_stats(name, idx, min_db, bw10_GHz, nn_dist, raw_fill, final_fill)
n = sum(idx);
if n == 0
    fprintf('%s: n = 0\n', name);
    return;
end

fprintf('%s: n = %d\n', name, n);
fprintf('  minS11: mean = %.2f, median = %.2f, best = %.2f\n', ...
    mean(min_db(idx)), median(min_db(idx)), min(min_db(idx)));
fprintf('  BW_GHz: mean = %.3f, median = %.3f, max = %.3f\n', ...
    mean(bw10_GHz(idx)), median(bw10_GHz(idx)), max(bw10_GHz(idx)));
fprintf('  NNdist: mean = %.4f, median = %.4f\n', ...
    mean(nn_dist(idx)), median(nn_dist(idx)));

if all(~isnan(raw_fill)) && all(~isnan(final_fill))
    fprintf('  raw_fill   mean = %.4f\n', mean(raw_fill(idx)));
    fprintf('  final_fill mean = %.4f\n', mean(final_fill(idx)));
    fprintf('  delta_fill mean = %.4f\n', mean(final_fill(idx) - raw_fill(idx)));
end
end

function meta = safe_jsondecode(meta_json)
try
    if isstring(meta_json) || ischar(meta_json)
        meta = jsondecode(char(meta_json));
    else
        meta = jsondecode(native2unicode(meta_json(:).'));
    end
catch
    meta = struct();
end
end

function geom_cfg = extract_geom_from_meta(meta, pixelN, fineN, default_geom)
geom_cfg = struct();

geom_cfg.patch_L_m = default_geom.patch_L_mm * 1e-3;
geom_cfg.patch_W_m = default_geom.patch_W_mm * 1e-3;
geom_cfg.sub_thick_m = default_geom.sub_thick_mm * 1e-3;
geom_cfg.board_L_m = default_geom.board_L_mm * 1e-3;
geom_cfg.board_W_m = default_geom.board_W_mm * 1e-3;
geom_cfg.feed_diam_m = default_geom.feed_diam_mm * 1e-3;
geom_cfg.substrate_name = default_geom.substrate_name;

try
    p = meta.params.geom;
    if isfield(p, 'patch_L_mm'),   geom_cfg.patch_L_m   = double(p.patch_L_mm) * 1e-3; end
    if isfield(p, 'patch_W_mm'),   geom_cfg.patch_W_m   = double(p.patch_W_mm) * 1e-3; end
    if isfield(p, 'sub_thick_mm'), geom_cfg.sub_thick_m = double(p.sub_thick_mm) * 1e-3; end
    if isfield(p, 'board_L_mm'),   geom_cfg.board_L_m   = double(p.board_L_mm) * 1e-3; end
    if isfield(p, 'board_W_mm'),   geom_cfg.board_W_m   = double(p.board_W_mm) * 1e-3; end
    if isfield(p, 'feed_diam_mm'), geom_cfg.feed_diam_m = double(p.feed_diam_mm) * 1e-3; end
    if isfield(p, 'substrate_name'), geom_cfg.substrate_name = char(p.substrate_name); end
catch
end

try
    if isfield(meta, 'params') && isfield(meta.params, 'overlap_mm')
        geom_cfg.overlap_m = double(meta.params.overlap_mm) * 1e-3;
    else
        geom_cfg.overlap_m = 1e-9;
    end
catch
    geom_cfg.overlap_m = 1e-9;
end

geom_cfg.pixelN = pixelN;
geom_cfg.fineN = fineN;
end

function [x_centers, y_centers] = build_current_grid_centers(L, W, fineN)
patch_origin = [-L/2, -W/2];
x_edges = linspace(patch_origin(1), patch_origin(1) + L, fineN + 1);
y_edges = linspace(patch_origin(2), patch_origin(2) + W, fineN + 1);
x_centers = (x_edges(1:end-1) + x_edges(2:end)) / 2;
y_centers = (y_edges(1:end-1) + y_edges(2:end)) / 2;
end

function ant = build_antenna_from_sample(designMatrix, feed_rc, geom_cfg, fineN)
if nargin < 4
    fineN = geom_cfg.fineN;
end

N = size(designMatrix, 1);
L = geom_cfg.patch_L_m;
W = geom_cfg.patch_W_m;
h = geom_cfg.sub_thick_m;
board_L = geom_cfg.board_L_m;
board_W = geom_cfg.board_W_m;
feedDiameter = geom_cfg.feed_diam_m;

substrateMaterial = dielectric(geom_cfg.substrate_name);
substrateMaterial.Thickness = h;

ground = antenna.Rectangle('Length', board_L, 'Width', board_W, 'Center', [0 0]);

patchShape = build_patch_shape_from_design_eval(designMatrix, geom_cfg);
assert(~isempty(patchShape), '设计金属区为空，无法重建天线。');

pixel_L = L / N;
pixel_W = W / N;
patch_origin = [-L/2, -W/2];
feed_x = patch_origin(1) + (feed_rc(2) - 0.5) * pixel_L;
feed_y = patch_origin(2) + (feed_rc(1) - 0.5) * pixel_W;
finalFeedLocation = [feed_x, feed_y];

ant = pcbStack();
ant.BoardShape     = ground;
ant.BoardThickness = h;
ant.FeedDiameter   = feedDiameter;
ant.Layers         = {patchShape, substrateMaterial, ground};
ant.FeedLocations  = [finalFeedLocation, 1, 3];
end

function patchShape = build_patch_shape_from_design_eval(designMatrix, geom_cfg)
N = size(designMatrix, 1);
L = geom_cfg.patch_L_m;
W = geom_cfg.patch_W_m;

pixel_L = L / N;
pixel_W = W / N;
origin_x = -L / 2;
origin_y = -W / 2;

if isfield(geom_cfg, 'overlap_m') && ~isempty(geom_cfg.overlap_m)
    overlap = geom_cfg.overlap_m;
else
    overlap = 1e-9;
end

rectCells = cell(nnz(designMatrix), 1);
idx = 0;

for rr = 1:N
    for cc = 1:N
        if ~designMatrix(rr, cc)
            continue;
        end

        idx = idx + 1;
        centerX = origin_x + (cc - 0.5) * pixel_L;
        centerY = origin_y + (rr - 0.5) * pixel_W;

        rectCells{idx} = antenna.Rectangle( ...
            'Length', pixel_L + overlap, ...
            'Width',  pixel_W + overlap, ...
            'Center', [centerX, centerY]);
    end
end

if idx == 0
    patchShape = [];
    return;
end

rectCells = rectCells(1:idx);
patchShape = union_rectangles_batch(rectCells, 64, false);
end

function shape = union_rectangles_batch(rectCells, batch, do_simplify)
if nargin < 2, batch = 64; end
if nargin < 3, do_simplify = true; end

ids = find(~cellfun('isempty', rectCells));
if isempty(ids)
    shape = [];
    return;
end

shape = rectCells{ids(1)};
cnt = 1;

for ii = 2:numel(ids)
    shape = shape + rectCells{ids(ii)};
    cnt = cnt + 1;

    if do_simplify && mod(cnt, batch) == 0
        try
            shape = shape.simplify;
        catch
        end
    end
end
end

function plot_current_component(ax, x_centers, y_centers, val, metalMask, ttl)
hImg = imagesc(ax, x_centers, y_centers, val);
set(hImg, 'AlphaData', double(metalMask));
set(ax, 'YDir', 'normal');
axis(ax, 'equal');
axis(ax, 'tight');
xlabel(ax, 'X (m)');
ylabel(ax, 'Y (m)');
title(ax, ttl);
colormap(ax, parula);
colorbar(ax);
end

function metalMask = build_metal_mask_from_design(designMatrix, geom_cfg, fineN)
N = size(designMatrix, 1);
L = geom_cfg.patch_L_m;
W = geom_cfg.patch_W_m;
overlap = geom_cfg.overlap_m;

[x_centers, y_centers] = build_current_grid_centers(L, W, fineN);
[Xc, Yc] = meshgrid(x_centers, y_centers);

pixel_L = L / N;
pixel_W = W / N;
origin_x = -L / 2;
origin_y = -W / 2;

metalMask = false(fineN, fineN);
tol = 1e-8;

for rr = 1:N
    for cc = 1:N
        if ~designMatrix(rr, cc)
            continue;
        end

        x0 = origin_x + (cc - 1) * pixel_L - overlap/2;
        x1 = origin_x + cc * pixel_L + overlap/2;
        y0 = origin_y + (rr - 1) * pixel_W - overlap/2;
        y1 = origin_y + rr * pixel_W + overlap/2;

        metalMask = metalMask | (Xc >= x0-tol & Xc <= x1+tol & Yc >= y0-tol & Yc <= y1+tol);
    end
end
end