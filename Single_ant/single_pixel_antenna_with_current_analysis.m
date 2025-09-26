function single_pixel_antenna_with_current_analysis()
% single_pixel_antenna_with_current_analysis
% -------------------------------------------------------------------------
% Description:
% 仿真一个简单的像素贴片，包含电流分析

% 主要特性：
% 1. mesh图像
% 2. S参数
% 3. current函数直接绘制的原始电流图（只含幅值，不含方向和相位）
% 4. 从current函数接收数据，手动插值重构的电流图（定义在mesh格点上的高分辨率图像）
% 5. 将mesh格点分配给像素，平均化后输出像素级低分辨率电流图（未完成）

% Author:  sxyue
% Date:    2025-09-25
% -------------------------------------------------------------------------

clc;
clear;
close all;

%% 1. 用户配置
fprintf('--- 1. 开始天线配置 ---\n');
pixelFillFactor     = 1;      % 可在 0 到 1 之间调整
centerFreq_GHz      = 2.45;
pixelResolution_N   = 16;
overlap_mm          = 0.8;
meshLambdaFraction  = 20;
rng(1);

%% 2. 计算设计参数
fprintf('--- 2. 计算天线设计参数 ---\n');
f_center = centerFreq_GHz * 1e9;
c = physconst('LightSpeed');
lambda0 = c / f_center;
overlap = overlap_mm / 1000;

L = lambda0 / 2; 
W = L * 1.0; 
h = lambda0 / 50;
substrateMaterial = dielectric('Air');

board_L = L * 1.2;
board_W = W * 1.2;
ground = antenna.Rectangle('Length', board_L, 'Width', board_W, 'Center', [0 0]);

pixel_L = L / pixelResolution_N;
pixel_W = W / pixelResolution_N;
patch_origin = [-L/2, -W/2];

pixelShapes = cell(pixelResolution_N, pixelResolution_N);
pixel_centers_x = patch_origin(1) + pixel_L/2 + (0:pixelResolution_N-1)*pixel_L;
pixel_centers_y = patch_origin(2) + pixel_W/2 + (0:pixelResolution_N-1)*pixel_W;

for r = 1:pixelResolution_N
    for c = 1:pixelResolution_N
        pixelShapes{r,c} = antenna.Rectangle( ...
            'Length', pixel_L + overlap, ...
            'Width', pixel_W + overlap, ...
            'Center', [pixel_centers_x(c), pixel_centers_y(r)]);
    end
end

initialFeedLocation = [L/4, 0];
feed_c_idx = floor((initialFeedLocation(1) - patch_origin(1)) / pixel_L) + 1;
feed_r_idx = floor((initialFeedLocation(2) - patch_origin(2)) / pixel_W) + 1;
feedPixelIdx = [max(1, min(pixelResolution_N, feed_r_idx)), max(1, min(pixelResolution_N, feed_c_idx))];
finalFeedLocation = pixelShapes{feedPixelIdx(1), feedPixelIdx(2)}.Center;
feedDiameter = min(pixel_L, pixel_W) / 10;

%% 3. 创建天线几何
fprintf('--- 3. 创建天线几何模型 ---\n');
if pixelFillFactor >= 1
    designVector = ones(pixelResolution_N, pixelResolution_N);
else
    designVector = rand(pixelResolution_N, pixelResolution_N) < pixelFillFactor;
    designVector(feedPixelIdx(1), feedPixelIdx(2)) = 1;
end

pixel_indices = find(designVector);
patchShape = pixelShapes{pixel_indices(1)};
for i = 2:length(pixel_indices)
    patchShape = patchShape + pixelShapes{pixel_indices(i)};
end

ant = pcbStack();
ant.BoardShape = ground;
ant.BoardThickness = h;
ant.FeedDiameter = feedDiameter;
ant.Layers = {patchShape, substrateMaterial, ground}; 
ant.FeedLocations = [finalFeedLocation, 1, 3]; 

%% 4. 仿真
fprintf('--- 4. 开始电磁仿真 ---\n');
mesh(ant, 'MaxEdgeLength', lambda0 / meshLambdaFraction);
freq_sweep = linspace(f_center*0.9, f_center*1.1, 41);
s = sparameters(ant, freq_sweep);

figure('Name', 'S11 参数');
rfplot(s);
title('S11 仿真结果');

%% ========================================================================
%  5. 电流分析与可视化
% =========================================================================
fprintf('\n--- 5. 开始电流分析与可视化 ---\n');

% --- 5.1 图1：求解器"金标准"俯视图 ---
fprintf('  正在生成 [图1] 求解器2D电流参考图...\n');
figure('Name', '图1：求解器2D电流参考图');
current(ant, f_center);
view(0, 90);
axis equal;
xlim([patch_origin(1), patch_origin(1) + L]);
ylim([patch_origin(2), patch_origin(2) + W]);
title('图1：求解器直接生成的电流分布（金标准-俯视图）');

% --- 5.2 提取原始高精度电流数据 ---
[J_surface, triangle_centroids] = current(ant, f_center);
J_mag_original = vecnorm(J_surface);

% --- 5.3 图2：手动绘制高分辨率连续电流幅值图 ---
fprintf('  正在生成 [图2] 插值绘制的高分辨率幅值图...\n');
figure('Name', '图2：高分辨率电流幅值图 (插值)');

x_scatter = triangle_centroids(1, :);
y_scatter = triangle_centroids(2, :);
interp_resolution = 200;
xq = linspace(patch_origin(1), patch_origin(1) + L, interp_resolution);
yq = linspace(patch_origin(2), patch_origin(2) + W, interp_resolution);
[Xq, Yq] = meshgrid(xq, yq);
F = scatteredInterpolant(x_scatter', y_scatter', J_mag_original', 'natural');
J_interp_mag = F(Xq, Yq);

% 【核心修正】创建蒙版，将无金属区域的电流设为 NaN
% 步骤1: 计算插值网格上每个点对应的像素索引(0到N-1)
r_idx_raw = floor((Yq - patch_origin(2)) / pixel_W);
c_idx_raw = floor((Xq - patch_origin(1)) / pixel_L);
% 步骤2: 强制将索引约束在 [0, N-1] 的有效范围内，避免边界越界
r_idx_clamped = max(0, min(r_idx_raw, pixelResolution_N - 1));
c_idx_clamped = max(0, min(c_idx_raw, pixelResolution_N - 1));
% 步骤3: 计算安全的1D索引 (1到N*N)
pixel_map_interp = r_idx_clamped * pixelResolution_N + c_idx_clamped + 1;
% 步骤4: 使用安全的索引生成蒙版
mask = designVector(pixel_map_interp);
J_interp_mag(~mask) = NaN;


pcolor(Xq, Yq, J_interp_mag);
shading interp;
axis equal; grid on; colorbar;
xlim([patch_origin(1), patch_origin(1) + L]);
ylim([patch_origin(2), patch_origin(2) + W]);
title('图2：手动绘制的高分辨率电流幅值 (插值+蒙版)');
xlabel('X (m)'); ylabel('Y (m)');
colormap(parula);

% --- 5.4 计算像素化平均电流 ---
fprintf('  正在执行"分区求平均"操作...\n');
num_triangles = size(J_surface, 2);
pixel_current_sum = zeros(pixelResolution_N, pixelResolution_N, 3); 
pixel_mesh_count = zeros(pixelResolution_N, pixelResolution_N);
triangle_to_pixel_map = zeros(num_triangles, 2);

for i = 1:num_triangles
    centroid = triangle_centroids(:, i)'; 
    c_idx = floor((centroid(1) - patch_origin(1)) / pixel_L) + 1;
    r_idx = floor((centroid(2) - patch_origin(2)) / pixel_W) + 1;
    if (c_idx >= 1 && c_idx <= pixelResolution_N && r_idx >= 1 && r_idx <= pixelResolution_N)
        pixel_current_sum(r_idx, c_idx, :) = squeeze(pixel_current_sum(r_idx, c_idx, :)) + J_surface(:, i);
        pixel_mesh_count(r_idx, c_idx) = pixel_mesh_count(r_idx, c_idx) + 1;
        triangle_to_pixel_map(i, :) = [r_idx, c_idx];
    end
end
pixel_mesh_count(pixel_mesh_count == 0) = 1;
avg_pixel_current = pixel_current_sum ./ pixel_mesh_count;
avg_pixel_current_magnitude = vecnorm(avg_pixel_current, 2, 3);
avg_pixel_current_magnitude(designVector == 0) = NaN;

% --- 5.5 图3：绘制像素化（低分辨率）电流幅值图 ---
fprintf('  正在生成 [图3] 像素化低分辨率幅值图...\n');
figure('Name', '图3：像素化低分辨率电流幅值图');
pixel_x_edges = patch_origin(1) + (0:pixelResolution_N) * pixel_L;
pixel_y_edges = patch_origin(2) + (0:pixelResolution_N) * pixel_W;
pcolor(pixel_x_edges, pixel_y_edges, padarray(avg_pixel_current_magnitude', [1 1], NaN, 'post'));
shading flat; colorbar; colormap(parula);
axis equal;
xlim([patch_origin(1), patch_origin(1) + L]);
ylim([patch_origin(2), patch_origin(2) + W]);
set(gca, 'YDir', 'normal');
title('图3：像素化平均电流幅值 (最终输出)');
xlabel('X (m)'); ylabel('Y (m)');

% --- 5.6 计算均方根误差 (RMSE) ---
fprintf('  正在计算平均化操作引入的误差...\n');
J_mag_averaged_expanded = zeros(1, num_triangles);
for i = 1:num_triangles
    r_idx = triangle_to_pixel_map(i, 1);
    c_idx = triangle_to_pixel_map(i, 2);
    if r_idx > 0 && c_idx > 0
        J_mag_averaged_expanded(i) = avg_pixel_current_magnitude(r_idx, c_idx);
    end
end

mean_original_magnitude = mean(J_mag_original, 'omitnan');
squared_errors = (J_mag_original - J_mag_averaged_expanded).^2;
rmse = sqrt(mean(squared_errors, 'omitnan'));
nrmse = rmse / mean_original_magnitude;

% --- 5.7 打印误差结果 ---
fprintf('\n------------------------------------------------------\n');
fprintf('定量分析结果:\n');
fprintf('  - 原始电流幅值均值: %.4f A/m\n', mean_original_magnitude);
fprintf('  - 均方根误差 (RMSE): %.4f A/m\n', rmse);
fprintf('  - 归一化RMSE (NRMSE): %.4f (或 %.2f%%)\n', nrmse, nrmse * 100);
fprintf('------------------------------------------------------\n');

fprintf('\n--- 脚本运行结束 ---\n');
end