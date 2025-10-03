function single_pixel_antenna_with_current_analysis_2()
% single_pixel_antenna_with_current_analysis2
% -------------------------------------------------------------------------
% 仿真像素贴片 + 电流分析
% 修复点：
%  1) 图2蒙版与含 overlap 的真实几何一致（修复B）
%  2) 插值去重不再修改 J_mag_original，避免长度不一致
%  3) 像素平均使用"最近像素中心"分配；维度赋值安全
%  4) 图3用像素中心坐标；RMSE 矢量化且强制列向量
% -------------------------------------------------------------------------

clc; clear; close all;

%% 1. 用户配置
fprintf('--- 1. 开始天线配置 ---\n');
pixelFillFactor     = 0.8;      % 0~1
centerFreq_GHz      = 2.45;
pixelResolution_N   = 16;
overlap_mm          = 0.8;
meshLambdaFraction  = 30;
rng(1);

%% 2. 计算设计参数
fprintf('--- 2. 计算天线设计参数 ---\n');
f_center = centerFreq_GHz * 1e9;
c0 = physconst('LightSpeed');
lambda0 = c0 / f_center;
overlap = overlap_mm / 1000;

L = lambda0 / 2; 
W = L * 1.0; 
h = lambda0 / 50;
substrateMaterial = dielectric('Air');
substrateMaterial.Thickness = h;

board_L = L * 1.2;
board_W = W * 1.2;
ground = antenna.Rectangle('Length', board_L, 'Width', board_W, 'Center', [0 0]);

pixel_L = L / pixelResolution_N;
pixel_W = W / pixelResolution_N;
patch_origin = [-L/2, -W/2];

pixelShapes = cell(pixelResolution_N, pixelResolution_N);
pixel_centers_x = patch_origin(1) + pixel_L/2 + (0:pixelResolution_N-1)*pixel_L;
pixel_centers_y = patch_origin(2) + pixel_W/2 + (0:pixelResolution_N-1)*pixel_W;

for rr = 1:pixelResolution_N
    for cc = 1:pixelResolution_N
        pixelShapes{rr,cc} = antenna.Rectangle( ...
            'Length', pixel_L + overlap, ...
            'Width',  pixel_W + overlap, ...
            'Center', [pixel_centers_x(cc), pixel_centers_y(rr)]);
    end
end

initialFeedLocation = [L/4, 0];
feed_c_idx = floor((initialFeedLocation(1) - patch_origin(1)) / pixel_L) + 1;
feed_r_idx = floor((initialFeedLocation(2) - patch_origin(2)) / pixel_W) + 1;
feedPixelIdx = [max(1, min(pixelResolution_N, feed_r_idx)), ...
                max(1, min(pixelResolution_N, feed_c_idx))];
finalFeedLocation = pixelShapes{feedPixelIdx(1), feedPixelIdx(2)}.Center;
feedDiameter = min(pixel_L, pixel_W)/10 ;
fprintf('  - 【最终馈电点】: (%.3f, %.3f) mm\n\n', ...
    finalFeedLocation(1)*1e3, finalFeedLocation(2)*1e3);

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
plot(patchShape);

% 输入阻抗
Z = impedance(ant, f_center);
fprintf('输入阻抗Z = %.2f Ohm\n', Z);

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

% 5.1 求解器参考图
fprintf('  生成 [图1] 求解器2D电流参考图...\n');
figure('Name', '图1：求解器2D电流参考图');
current(ant, f_center);
view(0, 90); axis equal;
xlim([patch_origin(1), patch_origin(1) + L]);
ylim([patch_origin(2), patch_origin(2) + W]);
title('图1：求解器电流分布（俯视）');
colorbar;

% 5.2 原始电流数据
[J_surface, triangle_centroids] = current(ant, f_center);
if size(J_surface,1) == 3  % 每列一个三分量
    J_mag_original = vecnorm(J_surface, 2, 1);   % 1 x M
else
    J_mag_original = vecnorm(J_surface, 2, 2).'; % M x 1 -> 1 x M
end
num_triangles = size(J_surface, 2);              % 重要：M

% 5.3 插值绘制的高分辨率幅值图（插值用去重，但不改 J_mag_original）
fprintf('  生成 [图2] 插值绘制的高分辨率幅值图...\n');
figure('Name', '图2：高分辨率电流幅值图 (插值)');

% 散点坐标
if size(triangle_centroids,1) >= 2
    x_scatter = triangle_centroids(1, :);
    y_scatter = triangle_centroids(2, :);
else
    x_scatter = triangle_centroids(:,1)';
    y_scatter = triangle_centroids(:,2)';
end

% —— 去重仅用于插值；不要改动 J_mag_original —— %
XY = [x_scatter(:), y_scatter(:)];
[~, ia, ~] = unique(XY, 'rows', 'stable');
x_scatter_u       = x_scatter(ia);
y_scatter_u       = y_scatter(ia);
J_mag_for_interp  = J_mag_original(ia);   % 新变量，仅供插值使用

% 插值网格（这里保持原范围；如需包含外扩 overlap，可自行外扩）
interp_resolution = 200;
xq = linspace(patch_origin(1), patch_origin(1) + L, interp_resolution);
yq = linspace(patch_origin(2), patch_origin(2) + W, interp_resolution);
[Xq, Yq] = meshgrid(xq, yq);

% 插值
F = scatteredInterpolant(x_scatter_u(:), y_scatter_u(:), J_mag_for_interp(:), 'natural');
F.ExtrapolationMethod = 'none';
J_interp_mag = F(Xq, Yq);

% 【修复B】基于真实金属形状(含 overlap)生成蒙版
[rr_on, cc_on] = find(designVector == 1);
cx = pixel_centers_x(cc_on);
cy = pixel_centers_y(rr_on);
halfL = (pixel_L + overlap)/2;
halfW = (pixel_W + overlap)/2;

mask = false(size(Xq));
if ~isempty(cx)
    condX = abs(Xq - reshape(cx, 1, 1, [])) <= halfL; 
    condY = abs(Yq - reshape(cy, 1, 1, [])) <= halfW;
    mask  = any(condX & condY, 3);
end
J_interp_mag(~mask) = NaN;

pcolor(Xq, Yq, J_interp_mag);
shading interp; axis equal; grid on; colorbar;
xlim([patch_origin(1), patch_origin(1) + L]);
ylim([patch_origin(2), patch_origin(2) + W]);
title('图2：手动绘制的高分辨率电流幅值 (插值+蒙版)');
xlabel('X (m)'); ylabel('Y (m)');
colormap(parula);

%% 5.4 计算像素化平均电流
fprintf('  正在执行\"分区求平均\"操作...\n');

% ---------------- 原始矢量平均再取模 ---------------- %
pixel_current_sum   = zeros(pixelResolution_N, pixelResolution_N, 3); 
pixel_mesh_count    = zeros(pixelResolution_N, pixelResolution_N);
triangle_to_pixel_map = zeros(num_triangles, 2);  % 0 表示未映射

for i = 1:num_triangles
    ci = triangle_centroids(:, i).';
    x = ci(1); y = ci(2);

    % 最近像素中心
    c_idx = round((x - patch_origin(1)) / pixel_L + 0.5);
    r_idx = round((y - patch_origin(2)) / pixel_W + 0.5);

    if c_idx < 1 || c_idx > pixelResolution_N || r_idx < 1 || r_idx > pixelResolution_N
        continue;
    end
    if ~designVector(r_idx, c_idx)
        continue;
    end

    % 矢量累加
    pixel_current_sum(r_idx, c_idx, :) = pixel_current_sum(r_idx, c_idx, :) ...
                                       + reshape(J_surface(:, i), [1, 1, 3]);
    pixel_mesh_count(r_idx, c_idx) = pixel_mesh_count(r_idx, c_idx) + 1;

    triangle_to_pixel_map(i, :) = [r_idx, c_idx];
end

% 矢量平均
avg_pixel_current = NaN(size(pixel_current_sum));
has_data = pixel_mesh_count > 0;
for k = 1:3
    tmp_sum = pixel_current_sum(:,:,k);
    tmp_avg = NaN(size(tmp_sum));
    tmp_avg(has_data) = tmp_sum(has_data) ./ pixel_mesh_count(has_data);
    avg_pixel_current(:,:,k) = tmp_avg;
end

avg_pixel_current_magnitude_vec = vecnorm(avg_pixel_current, 2, 3);
avg_pixel_current_magnitude_vec(~designVector) = NaN;

% ---------------- 新增：取模后再平均 ---------------- %
pixel_current_mag_sum = zeros(pixelResolution_N, pixelResolution_N); 
pixel_mesh_count2     = zeros(pixelResolution_N, pixelResolution_N);

for i = 1:num_triangles
    ci = triangle_centroids(:, i).';
    x = ci(1); y = ci(2);

    % 最近像素中心
    c_idx = round((x - patch_origin(1)) / pixel_L + 0.5);
    r_idx = round((y - patch_origin(2)) / pixel_W + 0.5);

    if c_idx < 1 || c_idx > pixelResolution_N || r_idx < 1 || r_idx > pixelResolution_N
        continue;
    end
    if ~designVector(r_idx, c_idx)
        continue;
    end

    % 幅值累加（区别在这里）
    pixel_current_mag_sum(r_idx, c_idx) = pixel_current_mag_sum(r_idx, c_idx) ...
                                        + norm(J_surface(:, i));
    pixel_mesh_count2(r_idx, c_idx) = pixel_mesh_count2(r_idx, c_idx) + 1;
end

% 幅值平均
avg_pixel_current_magnitude_mod = NaN(size(pixel_current_mag_sum));
has_data2 = pixel_mesh_count2 > 0;
avg_pixel_current_magnitude_mod(has_data2) = ...
    pixel_current_mag_sum(has_data2) ./ pixel_mesh_count2(has_data2);
avg_pixel_current_magnitude_mod(~designVector) = NaN;

%% 5.5 图像比较
fprintf('  正在生成 [图3/图3b] 像素化幅值图...\n');

figure('Name', '图3：像素化低分辨率电流幅值对比');
subplot(1,2,1);
imagesc(pixel_centers_x, pixel_centers_y, avg_pixel_current_magnitude_vec);
set(gca,'YDir','normal'); axis equal tight;
colorbar; colormap(parula);
xlabel('X (m)'); ylabel('Y (m)');
title('矢量平均 → 再取模');

subplot(1,2,2);
imagesc(pixel_centers_x, pixel_centers_y, avg_pixel_current_magnitude_mod);
set(gca,'YDir','normal'); axis equal tight;
colorbar; colormap(parula);
xlabel('X (m)'); ylabel('Y (m)');
title('取模 → 再平均');

%% 5.6 计算 RMSE
fprintf('  正在计算平均化操作引入的误差...\n');

% --- 矢量平均再取模 ---
J_mag_averaged_expanded_vec = NaN(1, num_triangles);
valid_tri = triangle_to_pixel_map(:,1) > 0 & triangle_to_pixel_map(:,2) > 0;
lin_idx   = sub2ind([pixelResolution_N, pixelResolution_N], ...
                    triangle_to_pixel_map(valid_tri,1), ...
                    triangle_to_pixel_map(valid_tri,2));
J_mag_averaged_expanded_vec(valid_tri) = avg_pixel_current_magnitude_vec(lin_idx);

valid = isfinite(J_mag_original(:)) & isfinite(J_mag_averaged_expanded_vec(:));
rmse_vec  = sqrt(mean((J_mag_original(valid) - J_mag_averaged_expanded_vec(valid)).^2));
nrmse_vec = rmse_vec / mean(J_mag_original(valid));

% --- 取模后再平均 ---
J_mag_averaged_expanded_mod = NaN(1, num_triangles);
J_mag_averaged_expanded_mod(valid_tri) = avg_pixel_current_magnitude_mod(lin_idx);

valid2 = isfinite(J_mag_original(:)) & isfinite(J_mag_averaged_expanded_mod(:));
rmse_mod  = sqrt(mean((J_mag_original(valid2) - J_mag_averaged_expanded_mod(valid2)).^2));
nrmse_mod = rmse_mod / mean(J_mag_original(valid2));

%% 输出结果
fprintf('\n------------------------------------------------------\n');
fprintf('定量分析结果:\n');
fprintf('  - 矢量平均再取模: RMSE = %.4f A/m, NRMSE = %.2f%%\n', rmse_vec, nrmse_vec*100);
fprintf('  - 取模后再平均:   RMSE = %.4f A/m, NRMSE = %.2f%%\n', rmse_mod, nrmse_mod*100);
fprintf('------------------------------------------------------\n');

% % 5.7 打印误差结果
% fprintf('\n------------------------------------------------------\n');
% fprintf('定量分析结果:\n');
% fprintf('  - 原始电流幅值均值: %.4f A/m\n', mean_original_magnitude);
% fprintf('  - 均方根误差 (RMSE): %.4f A/m\n', rmse);
% fprintf('  - 归一化RMSE (NRMSE): %.4f (或 %.2f%%)\n', nrmse, nrmse * 100);
% fprintf('------------------------------------------------------\n');

fprintf('\n--- 脚本运行结束 ---\n');
end
