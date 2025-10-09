function single_pixel_antenna_with_current_analysis_2()
% single_pixel_antenna_with_current_analysis_2
% -------------------------------------------------------------------------
% 仿真像素贴片 + 电流分析
% 修复与增强：
%  1) 图2蒙版与含 overlap 的真实几何一致（修复B）
%  2) 像素平均使用"最近像素中心"分配；维度赋值安全
%  3) 图3 基于图2插值结果的像素平均计算（图3）
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
if size(J_surface,1) == 3
    J_mag_original = vecnorm(J_surface, 2, 1);
else
    J_mag_original = vecnorm(J_surface, 2, 2).';
end

% 5.3 插值绘制高分辨率电流幅值图
fprintf('  生成 [图2] 插值绘制的高分辨率幅值图...\n');
figure('Name', '图2：高分辨率电流幅值图 (插值)');

x_scatter = triangle_centroids(1, :);
y_scatter = triangle_centroids(2, :);

XY = [x_scatter(:), y_scatter(:)];
[~, ia, ~] = unique(XY, 'rows', 'stable');
x_scatter_u = x_scatter(ia);
y_scatter_u = y_scatter(ia);
J_mag_for_interp = J_mag_original(ia);

interp_resolution = 200;
xq = linspace(patch_origin(1), patch_origin(1) + L, interp_resolution);
yq = linspace(patch_origin(2), patch_origin(2) + W, interp_resolution);
[Xq, Yq] = meshgrid(xq, yq);

F = scatteredInterpolant(x_scatter_u(:), y_scatter_u(:), J_mag_for_interp(:), 'natural');
F.ExtrapolationMethod = 'none';
J_interp_mag = F(Xq, Yq);

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

%% 5.4 基于插值图的像素化平均电流 (新增)
fprintf('  基于插值图计算像素平均电流...\n');

pixel_N = pixelResolution_N;
dx = L / pixel_N; dy = W / pixel_N;
avg_pixel_current_interp = NaN(pixel_N, pixel_N);

for r = 1:pixel_N
    for c = 1:pixel_N
        x_min = patch_origin(1) + (c-1)*dx;
        x_max = patch_origin(1) + c*dx;
        y_min = patch_origin(2) + (r-1)*dy;
        y_max = patch_origin(2) + r*dy;
        in_pixel = (Xq >= x_min & Xq < x_max & ...
                    Yq >= y_min & Yq < y_max & mask);
        vals = J_interp_mag(in_pixel);
        if ~isempty(vals)
            avg_pixel_current_interp(r,c) = mean(vals,'omitnan');
        end
    end
end
avg_pixel_current_interp(~designVector) = NaN;

figure('Name','图3：基于插值图的像素平均电流');
imagesc(pixel_centers_x, pixel_centers_y, avg_pixel_current_interp);
set(gca,'YDir','normal'); axis equal tight;
colorbar; colormap(parula);
xlabel('X (m)'); ylabel('Y (m)');
title('图3：基于插值图的像素平均电流');

%% 5.5 可选：误差评估（插值像素 vs 原插值图）
J_recon = imresize(avg_pixel_current_interp, [interp_resolution, interp_resolution], 'nearest');
valid = isfinite(J_interp_mag) & isfinite(J_recon);
rmse_interp = sqrt(mean((J_interp_mag(valid) - J_recon(valid)).^2));
nrmse_interp = rmse_interp / mean(J_interp_mag(valid));
fprintf('  - 基于插值像素平均 RMSE = %.4f, NRMSE = %.2f%%\n', rmse_interp, nrmse_interp*100);

%% 输出
fprintf('\n--- 脚本运行结束 ---\n');
end

