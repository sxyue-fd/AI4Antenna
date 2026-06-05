%% 基于高斯随机场 (GRF) 的 16x16 像素化图案生成
clear; clc; close all;

% 1. 参数设置
N = 16;             % 网格大小 (16x16)
correlation_len = 10; % 相关长度 (数值越大，图案越"丝滑"，斑块越大)

% 2. 构建坐标网格与频率网格
[X, Y] = meshgrid(1:N, 1:N);
[KX, KY] = meshgrid(fftshift(-N/2:N/2-1), fftshift(-N/2:N/2-1));
k_sq = KX.^2 + KY.^2;

% 3. 生成频率域的高斯滤波器 (指数衰减功率谱)
% 这里的滤波器决定了 GRF 的平滑程度
filter = exp(-k_sq * (correlation_len^2) / (2 * N^2));

% 4. 在频域生成随机场
% 生成复高斯白噪声
white_noise = randn(N, N) + 1i * randn(N, N);
% 频域卷积（等效于元素相乘）
field_fft = white_noise .* filter;

% 5. 转换回空间域
grf_continuous = real(ifft2(field_fft));

% 6. 标准化处理 (映射到 0-1 之间)
grf_continuous = (grf_continuous - min(grf_continuous(:))) / (max(grf_continuous(:)) - min(grf_continuous(:)));

% 7. 二值化处理 (生成 0/1 像素图案)
% 使用中位数作为阈值可以保证 0 和 1 的比例大致为 50/50
% threshold = median(grf_continuous(:));
threshold = 0.5;
grf_binary = grf_continuous > threshold;

% --- 可视化 ---
figure('Color', 'w', 'Position', [100, 100, 800, 350]);

subplot(1, 2, 1);
imagesc(grf_continuous);
colormap(gca, 'parula');
colorbar;
axis square;
title('连续高斯随机场 (Raw GRF)');
xlabel('像素'); ylabel('像素');

subplot(1, 2, 2);
imagesc(grf_binary);
colormap(gca, [1 1 1; 0 0 0]); % 0为白，1为黑
axis square;
grid on;
set(gca, 'XTick', 0.5:1:N+0.5, 'YTick', 0.5:1:N+0.5, 'XTickLabel', [], 'YTickLabel', []);
title(['16x16 二值化图案 (Threshold = ', num2str(threshold, '%.2f'), ')']);

% 添加说明
sgtitle(['GRF 像素图案生成 (相关长度 = ', num2str(correlation_len), ')']);