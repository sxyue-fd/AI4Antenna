function eval_dataset_quality()
% 质量与多样性评估：
% - 最小 S11(dB)
% - -10 dB 带宽（最长连续区间）
% - 数据多样性（像素通道的平均海明距离）
% - 馈电位置热力图
%
% 用法：
%   eval_dataset_quality('dataset_out/antenna_dataset_20251007_123456.h5')

h5path = 'dataset_out/antenna_dataset_20251008_180003.h5';
assert(isfile(h5path), '文件不存在：%s', h5path);

fprintf('读取数据：%s\n', h5path);
X = h5read(h5path, '/X');        % [B,C,H,W], single
Y = h5read(h5path, '/Y');        % [B,S],    single
freq = h5read(h5path, '/freq_hz'); % [1,S] or [S,1]
if size(freq,1) == 1, freq = freq(:); end

[B,C,H,W] = size(X);
S = size(Y,2);
assert(C>=2, 'X 通道应≥2（像素与馈电掩码）');

% --- 指标计算 ---
Y_mag = double(Y);           % [B,S]
Y_db  = 20*log10(Y_mag + eps);
[min_db, ~] = min(Y_db, [], 2);   % [B,1]

bw10 = zeros(B,1);           % -10 dB 带宽（Hz）
f10_lo = nan(B,1);
f10_hi = nan(B,1);
thr = -10;

for i = 1:B
    mask = (Y_db(i,:) <= thr);  % 低于 -10 dB 的点
    if any(mask)
        % 找最长连续 True 段
        d = diff([false mask false]);
        run_start = find(d==1);
        run_end   = find(d==-1) - 1;
        [~, idx] = max(run_end - run_start + 1);
        s1 = run_start(idx); e1 = run_end(idx);
        f10_lo(i) = freq(s1);
        f10_hi(i) = freq(e1);
        bw10(i)   = f10_hi(i) - f10_lo(i);
    else
        bw10(i) = 0;
    end
end

% --- 多样性：像素海明距离（均值） ---
% 展平像素通道（C1）
pix = squeeze(X(:,1,:,:));              % [B,H,W]
pix = reshape(pix, B, H*W) > 0.5;       % logical

% 对于 B~100 以内可全对；若更大，随机抽取子集
if B <= 150
    % pairwise Hamming
    % 计算上三角 (i<j) 的 1 距比率
    M = B*(B-1)/2;
    cnt = 0; acc = 0;
    for i = 1:B-1
        xi = pix(i,:);
        for j = i+1:B
            acc = acc + nnz(xi ~= pix(j,:));
            cnt = cnt + numel(xi);
        end
    end
    mean_hamming = acc / cnt;
else
    % 子采样 2000 对
    P = 2000;
    rng(42);
    acc = 0; cnt = 0;
    for t = 1:P
        i = randi(B); j = randi(B); while j==i, j = randi(B); end
        acc = acc + nnz(pix(i,:) ~= pix(j,:));
        cnt = cnt + numel(pix(i,:));
    end
    mean_hamming = acc / cnt;
end

% --- 馈电位置热力图 ---
feedmask = squeeze(X(:,2,:,:));             % [B,H,W]
[~, feed_lin] = max(reshape(feedmask, B, H*W), [], 2);  % 每样本喂电位置
feed_r = mod(feed_lin-1, H) + 1;
feed_c = floor((feed_lin-1)/H) + 1;
feed_map = accumarray([feed_r, feed_c], 1, [H, W]);

% --- 输出统计 ---
fprintf('\n=== 数据集质量指标 ===\n');
fprintf('样本数 B = %d，频点数 S = %d\n', B, S);
fprintf('min(S11)_dB:  P50=%.2f, P10=%.2f, P90=%.2f, 最好=%.2f\n', ...
    prctile(min_db,50), prctile(min_db,10), prctile(min_db,90), min(min_db));
bwGHz = bw10 / 1e9;
fprintf('-10 dB 带宽(最大连续)：P50=%.3f GHz, P90=%.3f GHz, 最大=%.3f GHz\n', ...
    prctile(bwGHz,50), prctile(bwGHz,90), max(bwGHz));
fprintf('像素海明距离（平均）：%.3f\n', mean_hamming);
% 在输出统计部分，增加或修改这一行
fprintf('满足 min(S11) < -10 dB 的样本比例：%.1f%%\n', 100 * mean(min_db < -10));

% 同时，为了避免混淆，修改之前那行的标签
fprintf('拥有非零-10dB带宽的样本比例：%.1f%%\n', 100*mean(bw10>0));

% --- 可视化 ---
[~, ~, ~] = fileparts(h5path);

% 1) min S11 直方图
figure('Name','Min S11 (dB)');
histogram(min_db, 20);
xlabel('min S11 (dB)'); ylabel('count'); grid on;
title(sprintf('Min S11: median=%.2f dB', median(min_db)));


% 2) -10 dB 带宽直方图
figure('Name','-10 dB Bandwidth');
histogram(bw10/1e9, 20);
xlabel('BW at -10 dB (GHz)'); ylabel('count'); grid on;
title(sprintf('-10 dB BW: median=%.3f GHz', median(bw10/1e9)));


% 3) 散点：min S11 vs BW
figure('Name','Min S11 vs BW');
scatter(min_db, bw10/1e9, 'filled');
xlabel('min S11 (dB)'); ylabel('BW at -10 dB (GHz)'); grid on;
title('Min S11 vs -10 dB Bandwidth');


% 4) 馈电位置热力图
figure('Name','Feed Position Heatmap');
imagesc(feed_map); axis image; colorbar;
xlabel('col'); ylabel('row'); title('Feed Position Heatmap');


% 5) 示例：挑选 min S11 最好的前 8 个，画像素图 + 频响
K = min(8, B);
[~, idx] = sort(min_db, 'ascend');
best_idx = idx(1:K);

figure('Name','Top designs: pixels');
for k = 1:K
    subplot(2,4,k);
    im = squeeze(X(best_idx(k),1,:,:));
    imagesc(im); axis image off; title(sprintf('#%d: %.1f dB', best_idx(k), min_db(best_idx(k))));
end


figure('Name','Top designs: S11 curves');
for k = 1:K
    subplot(2,4,k);
    plot(freq/1e9, Y_db(best_idx(k),:), '-');
    hold on; yline(-10,'--'); hold off;
    xlabel('GHz'); ylabel('S11 (dB)'); grid on;
    title(sprintf('#%d', best_idx(k)));
end

end
