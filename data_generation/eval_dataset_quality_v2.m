function eval_dataset_quality()
% =========================================================================
% 数据集质量与多样性评估
%
% 指标：
%   - 最小 S11 (dB)
%   - -10 dB 带宽（最长连续频段）
%   - 像素多样性（平均海明距离）
%   - f_0
%   - 馈电位置分布（热力图）
%
% 用法：
%   eval_dataset_quality()
% v2 更新：
%   前top16数据集为cost前16的数据集
%   新增谐振频率 vs -10 dB 带宽图像
%   可以只评估数据集前N个样本
%
% 注意：
%   默认从 h5path 指定的 HDF5 文件读取数据
% =========================================================================

%% ------------------------------------------------------------------------
% 0) 数据路径
% -------------------------------------------------------------------------

h5path = 'dataset_out\antenna_dataset_20260118_164009.h5';
assert(isfile(h5path), '文件不存在：%s', h5path);

fprintf('读取数据：%s\n', h5path);

%% ------------------------------------------------------------------------
% 1) 读取数据
% -------------------------------------------------------------------------

% %% ------------------------------------------------------------------------
% 1) 读取数据（只读前 N 个样本）
% -------------------------------------------------------------------------

N_eval = 300;   % 你想评估的前 N 条

% 先读取 freq（通常不大）
freq = h5read(h5path, '/freq_hz');  % [S,1] or [1,S]
if size(freq,1) == 1, freq = freq(:); end
S = numel(freq);

% 用 h5info 获取 /X 和 /Y 的尺寸（避免把全量读进内存）
infoX = h5info(h5path, '/X');
infoY = h5info(h5path, '/Y');

% HDF5 里的维度顺序应是 [B, C, H, W] 和 [B, S]
B_all = infoX.Dataspace.Size(1);
C     = infoX.Dataspace.Size(2);
H     = infoX.Dataspace.Size(3);
W     = infoX.Dataspace.Size(4);

% 只取前 N_eval 个
B = min(N_eval, B_all);

% --- 只读取前 B 个样本 ---
X = h5read(h5path, '/X', [1 1 1 1], [B C H W]);
Y = h5read(h5path, '/Y', [1 1],     [B S]);

assert(C >= 2, 'X 通道数应 ≥ 2（像素 + 馈电掩码）');

%% ------------------------------------------------------------------------
% 2) S11 指标计算
% -------------------------------------------------------------------------

Y_mag = double(Y);
Y_db  = 20 * log10(Y_mag + eps);

[min_db, idx_min] = min(Y_db, [], 2);   % [B,1]
f0 = freq(idx_min);                     % [B,1]  % 谐振频率（Hz）

%% ------------------------------------------------------------------------
% 3) -10 dB 带宽（最长连续频段）
% -------------------------------------------------------------------------

bw10   = zeros(B,1);
f10_lo = nan(B,1);
f10_hi = nan(B,1);
thr    = -10;

for i = 1:B
    mask = (Y_db(i,:) <= thr);

    if any(mask)
        d = diff([false, mask, false]);
        run_start = find(d == 1);
        run_end   = find(d == -1) - 1;

        [~, idx] = max(run_end - run_start + 1);

        s1 = run_start(idx);
        e1 = run_end(idx);

        f10_lo(i) = freq(s1);
        f10_hi(i) = freq(e1);
        bw10(i)   = f10_hi(i) - f10_lo(i);
    else
        bw10(i) = 0;
    end
end

%% ------------------------------------------------------------------------
% 4) 多样性：dup_ratio（近重复率）
%   定义：随机抽样若干对(i,j)，若结构距离 d < tau 则认为 near-duplicate
%   d = w_pix * dH(pixels) + w_feed * dH(feedmask)
% -------------------------------------------------------------------------

% --- 展平像素与馈电通道 ---
pix  = squeeze(X(:,1,:,:)) > 0.5;      % [B,H,W] logical
feed = squeeze(X(:,2,:,:)) > 0.5;      % [B,H,W] logical

pix_vec  = reshape(pix,  B, H*W);      % [B,HW]
feed_vec = reshape(feed, B, H*W);      % [B,HW]

% --- 距离加权（你可调整） ---
w_pix  = 1.0;     % 像素结构权重
w_feed = 1.0;     % 馈电位置权重（如果你觉得馈电差异不该算太多，可设 0.2）

% --- near-duplicate 阈值（你可调整） ---
% tau 是"加权海明距离"的阈值：0.02 表示平均只有 2% bit 不同就算重复
tau = 0.1;

% --- 抽样 pair 数（避免 B^2） ---
max_pairs = 20000;
num_pairs = min(max_pairs, B*(B-1)/2);

rng(42);

near_cnt = 0;
d_list   = zeros(num_pairs,1);

for t = 1:num_pairs
    i = randi(B);
    j = randi(B-1);
    if j >= i, j = j + 1; end

    dH_pix  = mean(xor(pix_vec(i,:),  pix_vec(j,:)));
    dH_feed = mean(xor(feed_vec(i,:), feed_vec(j,:)));

    d = w_pix*dH_pix + w_feed*dH_feed;   % 加权结构距离
    d_list(t) = d;

    if d < tau
        near_cnt = near_cnt + 1;
    end
end

dup_ratio = near_cnt / num_pairs;

% （可选）也顺手给你一个"最近邻距离"的粗估，帮助你判断 tau 是否合适
min_d_est = min(d_list);


%% ------------------------------------------------------------------------
% 5) 馈电位置热力图
% -------------------------------------------------------------------------

feedmask = squeeze(X(:,2,:,:));              % [B,H,W]
[~, feed_lin] = max(reshape(feedmask, B, H*W), [], 2);

feed_r = mod(feed_lin - 1, H) + 1;
feed_c = floor((feed_lin - 1) / H) + 1;

feed_map = accumarray([feed_r, feed_c], 1, [H, W]);

%% ------------------------------------------------------------------------
% 6) 统计输出
% -------------------------------------------------------------------------

fprintf('\n=== 数据集质量指标 ===\n');
fprintf('样本数 B = %d，频点数 S = %d\n', B, S);

fprintf('min(S11)_dB:\n');
fprintf('  P50 = %.2f dB, P10 = %.2f dB, P90 = %.2f dB, 最好 = %.2f dB\n', ...
    prctile(min_db,50), ...
    prctile(min_db,10), ...
    prctile(min_db,90), ...
    min(min_db));

bwGHz = bw10 / 1e9;
fprintf('-10 dB 带宽（最大连续）：\n');
fprintf('  P50 = %.3f GHz, P90 = %.3f GHz, 最大 = %.3f GHz\n', ...
    prctile(bwGHz,50), ...
    prctile(bwGHz,90), ...
    max(bwGHz));

% --- 新指标：平均带宽 & dup_ratio ---
mean_bwGHz = mean(bw10) / 1e9;

% （可选）归一化平均带宽：除以扫频范围，便于不同 fmin/fmax 比较
bw_norm = bw10 / (freq(end) - freq(1) + eps);
mean_bw_norm = mean(bw_norm);

fprintf('平均 -10 dB 带宽：%.3f GHz\n', mean_bwGHz);
fprintf('平均归一化带宽：%.3f\n', mean_bw_norm);

fprintf('dup_ratio（近重复率，tau=%.3f, pairs=%d）：%.3f\n', tau, num_pairs, dup_ratio);
fprintf('抽样最小结构距离 min_d_est：%.4f\n', min_d_est);
fprintf('满足 min(S11) < -10 dB 的样本比例：%.1f%%\n', ...
    100 * mean(min_db < -10));
fprintf('拥有非零 -10 dB 带宽的样本比例：%.1f%%\n', ...
    100 * mean(bw10 > 0));

counts = feed_map(:);
p = counts / sum(counts);
p = p(p > 0);

H     = -sum(p .* log(p));
Hmax  = log(numel(feed_map));
KLuni = sum(p .* log(p * numel(feed_map)));

fprintf('Feed 分布熵: %.3f（相对 %.1f%%），KL 到均匀 = %.3f\n', ...
    H, 100*H/Hmax, KLuni);
valid_bw = bw10 > 0;

f0_GHz  = f0(valid_bw)  / 1e9;
bw_GHz  = bw10(valid_bw) / 1e9;
%% ------------------------------------------------------------------------
% 7) 可视化
% -------------------------------------------------------------------------

% --- 1) min S11 直方图 ---
figure('Name','Min S11 (dB)');
histogram(min_db, 20);
xlabel('min S11 (dB)');
ylabel('count');
grid on;
title(sprintf('Min S11: median = %.2f dB', median(min_db)));

% --- 2) -10 dB 带宽直方图 ---
figure('Name','-10 dB Bandwidth');
histogram(bwGHz, 20);
xlabel('BW at -10 dB (GHz)');
ylabel('count');
grid on;
title(sprintf('-10 dB BW: median = %.3f GHz', median(bwGHz)));

% --- 3) min S11 vs 带宽 ---
figure('Name','Min S11 vs BW');
scatter(min_db, bwGHz, 'filled');
xlabel('min S11 (dB)');
ylabel('BW at -10 dB (GHz)');
grid on;
title('Min S11 vs -10 dB Bandwidth');

% --- 4) 馈电位置热力图 ---
figure('Name','Feed Position Heatmap');
imagesc(feed_map);
axis image;
colorbar;
xlabel('column');
ylabel('row');
title('Feed Position Heatmap');

% --- 新图：谐振频率 vs -10 dB 带宽 ---
figure('Name','Resonant Frequency vs -10 dB Bandwidth');
scatter(f0_GHz, bw_GHz, 20, 'filled');
grid on;
xlabel('Resonant frequency f_0 (GHz)');
ylabel('-10 dB Bandwidth (GHz)');
title('Resonant Frequency vs -10 dB Bandwidth');

% % --- 5) Top-K 设计像素 ---
% K = min(8, B);
% [~, idx] = sort(min_db, 'ascend');
% best_idx = idx(1:K);
% 
% figure('Name','Top Designs: Pixels');
% for k = 1:K
%     subplot(2,4,k);
%     imagesc(squeeze(X(best_idx(k),1,:,:)));
%     axis image off;
%     title(sprintf('#%d: %.1f dB', best_idx(k), min_db(best_idx(k))));
% end
% 
% % --- 6) Top-K S11 曲线 ---
% figure('Name','Top Designs: S11 Curves');
% for k = 1:K
%     subplot(2,4,k);
%     plot(freq/1e9, Y_db(best_idx(k),:), '-');
%     hold on;
%     yline(-10, '--');
%     hold off;
%     grid on;
%     xlabel('GHz');
%     ylabel('S11 (dB)');
%     title(sprintf('#%d', best_idx(k)));
% end
%  K = min(16, B);
% first_idx = 1:K;
% figure('Name','First K Designs: Pixels');
% for k = 1:K
%     ii = first_idx(k);
%     subplot(4,4,k);
%     imagesc(squeeze(X(ii,1,:,:)));
%     axis image off;
%     title(sprintf('Sample #%d', ii));
% end
% 
% 
% figure('Name','First K Designs: S11 Curves');
% for k = 1:K
%     ii = first_idx(k);
%     subplot(4,4,k);
%     plot(freq/1e9, Y_db(ii,:), '-');
%     hold on; yline(-10,'--'); hold off;
%     grid on;
%     xlabel('GHz');
%     ylabel('S11 (dB)');
%     title(sprintf('Sample #%d', ii));
% end
idx_start = 1;
idx_end   = 16;

first_idx = idx_start : min(idx_end, B);  % 防止超过样本数
K = numel(first_idx);

ncol = 4;
nrow = ceil(K/ncol);

figure('Name','Designs: Pixels');
for k = 1:K
    ii = first_idx(k);
    subplot(nrow, ncol, k);
    imagesc(squeeze(X(ii,1,:,:)));
    axis image off;
    title(sprintf('Sample #%d', ii));
end

figure('Name','Designs: S11 Curves');
for k = 1:K
    ii = first_idx(k);
    subplot(nrow, ncol, k);
    plot(freq/1e9, Y_db(ii,:), '-');
    hold on; yline(-10,'--'); hold off;
    grid on;
    xlabel('GHz');
    ylabel('S11 (dB)');
    title(sprintf('Sample #%d', ii));
end

end
