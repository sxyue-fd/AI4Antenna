function single_pixel_antenna_with_current_analysis_3()
% single_pixel_antenna_with_current_analysis_3
% -------------------------------------------------------------------------
% Description:
% 仿真像素贴片，并进行电流分析，将不规则电流
% 插值到统一高分辨率规则网格

% 主要特性：
% 1. 从 HDF5 数据读取像素结构与馈电位置，自动重建 patch 几何
% 2. 支持多组 meshLambdaFraction 循环仿真
% 3. 提取 current() 返回的三角面片电流数据（复数 Jx/Jy）
% 4. 仅保留贴片顶层电流，去除地板与非金属区域
% 5. 使用 scatteredInterpolant 将电流插值到统一 fineN×fineN 网格
% 6. 构造金属掩膜，仅显示贴片区域电流
% 8. 计算不同网格剖分间的误差指标（relL2 / maxAbs / 相关系数）

% Author:  xyzhu
% Date:    2026-03-01
% -------------------------------------------------------------------------
%% 1. 用户配置
fprintf('--- 1. 开始天线配置 ---\n');
dataset_file = 'dataset_out/antenna_dataset_20260115_220237.h5';%从文件中读取天线结构

centerFreq_GHz      = 10.8;
 overlap_mm          = 0.2;
% 三种不同剖分精细度
meshLambdaFraction_list =[10,15,20];   % 越大网格越细（MaxEdgeLength = lambda0 / fraction）

% 输出均匀电流网格 64x64（16x16 每像素细分 4x4）
fineN =64;
%% 2. 读取第一个天线结构
fprintf('--- 2. 从文件读取第一个天线样本 ---\n');
X_all    = h5read(dataset_file, '/X');        % [B,2,N,N]
Y_all    = h5read(dataset_file, '/Y');        % [B,F]
feed_all = h5read(dataset_file, '/feed_rc');  % [B,2]
freq     = h5read(dataset_file, '/freq_hz');  % sweep 频率

first_idx = 1;  % === 目标1：固定读取第一个天线
X_1 = squeeze(X_all(first_idx,:,:,:));        % [2,N,N]
designVector = squeeze(X_1(1,:,:));           % 金属分布
feed_rc = squeeze(feed_all(first_idx,:));     % [row, col]

designVector = logical(designVector);
pixelResolution_N = size(designVector,1);

fprintf('  使用样本编号: %d, 像素分辨率: %dx%d\n', first_idx, pixelResolution_N, pixelResolution_N);

%% 3. 计算天线几何参数
fprintf('--- 3. 计算天线设计参数 ---\n');
f_center = centerFreq_GHz * 1e9;
c0 = physconst('LightSpeed');
lambda0 = c0 / f_center;


L = 14e-3;
W = 14e-3;
h = 2.5e-3;

board_L = 30e-3;
board_W = 30e-3;
%overlap = overlap_mm / 1000;
overlap = 2*L/fineN;%控制overlap大小为一个电流像素
substrateMaterial = dielectric('Air');
substrateMaterial.Thickness = h;

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

feedPixelIdx = feed_rc; % [row,col]
finalFeedLocation = pixelShapes{feedPixelIdx(1), feedPixelIdx(2)}.Center;
feedDiameter = min(pixel_L, pixel_W)/10;

fprintf('  - 馈电点: (%.3f, %.3f) mm\n', finalFeedLocation(1)*1e3, finalFeedLocation(2)*1e3);

%% 4. 创建像素拼接 patch + PCB stack
fprintf('--- 4. 创建天线几何模型 ---\n');

pixel_indices = find(designVector);
patchShape = pixelShapes{pixel_indices(1)};
for i = 2:length(pixel_indices)
    patchShape = patchShape + pixelShapes{pixel_indices(i)};
end

ant = pcbStack();
ant.BoardShape     = ground;
ant.BoardThickness = h;
ant.FeedDiameter   = feedDiameter;
ant.Layers         = {patchShape, substrateMaterial, ground};
ant.FeedLocations  = [finalFeedLocation, 1, 3];

figure('Name','Patch Geometry');
plot(patchShape);
title('像素化贴片几何');

%% ========================================================================
% 5. 三种网格剖分：仿真 + 电流插值到 64x64 + 对比
% =========================================================================
fprintf('--- 5. 三种剖分精细度的仿真与电流对比 ---\n');

% === 64x64 网格中心点坐标（每个小格中心）
x_edges = linspace(patch_origin(1)-overlap/2, patch_origin(1)+L+overlap/2, fineN+1);
y_edges = linspace(patch_origin(2)-overlap/2, patch_origin(2)+W+overlap/2, fineN+1);
x_centers = (x_edges(1:end-1) + x_edges(2:end)) / 2;
y_centers = (y_edges(1:end-1) + y_edges(2:end)) / 2;
[Xc, Yc] = meshgrid(x_centers, y_centers);

% === 64x64 金属蒙版（16x16 每像素细分 4x4 => kron）
% scaleFactor = fineN / pixelResolution_N;   % 64/16=4
% if abs(scaleFactor - round(scaleFactor)) > 0
%     error('fineN 必须是 pixelResolution_N 的整数倍，例如 16->64。');
% end
% metalMask64 = kron(designVector, true(scaleFactor));  % 64x64 logical
% === 更准确的金属蒙版：考虑 overlap（在 64x64 的网格中心上判断是否落在任一金属像素块的扩大矩形内）
metalMask64 = false(size(Xc));   % fineN x fineN

halfL = (pixel_L + overlap)/2;
halfW = (pixel_W + overlap)/2;

for rr = 1:pixelResolution_N
    for cc = 1:pixelResolution_N
        if ~designVector(rr,cc), continue; end

        cx = pixel_centers_x(cc);
        cy = pixel_centers_y(rr);

        inRect = (Xc >= (cx - halfL)) & (Xc <= (cx + halfL)) & ...
                 (Yc >= (cy - halfW)) & (Yc <= (cy + halfW));

        metalMask64 = metalMask64 | inRect;
    end
end
% 结果容器
numMeshes = numel(meshLambdaFraction_list);
J64_all  = cell(numMeshes,1);
Jx64_all = cell(numMeshes,1);
Jy64_all = cell(numMeshes,1);
S11_all = cell(numMeshes,1);

for k = 1:numMeshes
    meshLambdaFraction = meshLambdaFraction_list(k);
    fprintf('\n  [Case %d/%d] meshLambdaFraction = %g\n', k, numMeshes, meshLambdaFraction);

    % --- 5.1 网格剖分 + S参数
    % mesh(ant, 'MaxEdgeLength', lambda0 / meshLambdaFraction);
    % 生成网格但不绘图（关键：接收输出）
    msh = mesh(ant, 'MaxEdgeLength', lambda0 / meshLambdaFraction);
    figSolver = figure('Name', ...
    sprintf('Solver Current (meshFraction=%g)', meshLambdaFraction));

    current(ant, f_center);
    view(0,90);
    axis equal;
    title(sprintf('Solver 电流分布 meshFraction=%g', meshLambdaFraction));
    colorbar;
    colormap(parula);
    s = sparameters(ant, freq);
    S11_all{k} = s;

       % --- 5.2 取中心频点电流（不均匀三角面片）
    [J_surface, tri_centroids] = current(ant, f_center);
    
    % 统一维度
    if size(J_surface,1) ~= 3
        J_surface = J_surface.';   % 3xM
    end
    
    % 取分量
    Jx = J_surface(1,:).';   % Mx1
    Jy = J_surface(2,:).';   % Mx1
    
    % 三角形中心坐标
    x_scatter = tri_centroids(1,:).';
    y_scatter = tri_centroids(2,:).';
    z_scatter = tri_centroids(3,:).';
    
    % ===== 关键：只保留"贴片顶层"的三角面片 =====
    zPatch = h;
    %tolZ   = max(1e-6, h/200);   % 容差可按需要调大/调小
    tolZ   = 1e-8;
    isPatchTri = abs(z_scatter - zPatch) < tolZ;
    
    % （可选但推荐）再限制在贴片投影范围内，避免板边等位置干扰
    inPatchXY = (x_scatter >= patch_origin(1)-overlap/2) & (x_scatter <= patch_origin(1)+L+overlap/2) & ...
                (y_scatter >= patch_origin(2)-overlap/2) & (y_scatter <= patch_origin(2)+W+overlap/2);
    
    keep = isPatchTri & inPatchXY;
    
    x_scatter = x_scatter(keep);
    y_scatter = y_scatter(keep);
    Jx = Jx(keep);
    Jy = Jy(keep);
    
    % 去重（scatteredInterpolant 对重复点敏感）
    XY = [x_scatter, y_scatter];
    [~, ia] = unique(XY, 'rows', 'stable');
    
    x_u  = x_scatter(ia);
    y_u  = y_scatter(ia);
    Jx_u = Jx(ia);
    Jy_u = Jy(ia);
    
    % --- 5.3 分别插值 Jx、Jy 到 64x64 网格中心
    % Fx = scatteredInterpolant(x_u, y_u, Jx_u, 'natural', 'none');
    % Fy = scatteredInterpolant(x_u, y_u, Jy_u, 'natural', 'none');
    Fx_re = scatteredInterpolant(x_u, y_u, real(Jx_u), 'natural','none');%自然邻点插值外推能力弱
    Fx_im = scatteredInterpolant(x_u, y_u, imag(Jx_u), 'natural','none');
    Jx64  = Fx_re(Xc,Yc) + 1j*Fx_im(Xc,Yc);

    Fy_re = scatteredInterpolant(x_u, y_u, real(Jy_u), 'natural','none');
    Fy_im = scatteredInterpolant(x_u, y_u, imag(Jy_u), 'natural','none');
    Jy64  = Fy_re(Xc,Yc) + 1j*Fy_im(Xc,Yc);
    % Jx64 = Fx(Xc, Yc);
    % Jy64 = Fy(Xc, Yc);
    
    % 金属区域外置 NaN
    Jx64(~metalMask64) = NaN;
    Jy64(~metalMask64) = NaN;
    
    % 由分量计算幅值
    Jmag64 = sqrt( abs(Jx64).^2 + abs(Jy64).^2 );
    
    % 保存
    J64_all{k} = Jmag64;      % 幅值
    Jx64_all{k} = Jx64;       % 分量
    Jy64_all{k} = Jy64;
    
    % --- 5.4 可视化：画幅值图
    % --- 5.4 可视化：画幅值图（显式axes，避免被覆盖）
    fig = figure('Name', sprintf('|J| 64x64 - meshFraction=%g', meshLambdaFraction));
    ax  = axes('Parent', fig);   % 显式创建 axes
    hImg = imagesc(ax, x_centers, y_centers, Jmag64);

    alphaMask = double(metalMask64);
    alphaMask(~metalMask64) = 0;   % 非金属透明
    alphaMask(metalMask64)  = 1;   % 金属不透明

    set(hImg, 'AlphaData', alphaMask);
    set(ax,'YDir','normal'); axis(ax,'equal'); axis(ax,'tight');
    xlim(ax, [patch_origin(1)-overlap/2, patch_origin(1)+L+overlap/2]);
    ylim(ax, [patch_origin(2)-overlap/2, patch_origin(2)+W+overlap/2]);
    title(ax, sprintf('64x64 电流幅值 |J| (meshLambdaFraction=%g)', meshLambdaFraction));
    xlabel(ax,'X (m)'); ylabel(ax,'Y (m)');
    colorbar(ax); colormap(ax, parula);
    drawnow;
end

%% 6. 对比三种剖分差异（目标3）
fprintf('\n--- 6. 电流分布差异对比 ---\n');

% 选最细网格作为参考（通常 meshLambdaFraction 最大最细）
[~, refIdx] = max(meshLambdaFraction_list);
Jref = J64_all{refIdx};

% 计算差异指标 + 差分图
for k = 1:numMeshes
    if k == refIdx, continue; end
    Jk = J64_all{k};

    % 仅在两者都非 NaN 的金属区域比较
    valid = ~isnan(Jref) & ~isnan(Jk);

    diffAbs = zeros(size(Jref));
    diffAbs(valid) = abs(Jk(valid) - Jref(valid));
    diffAbs(~valid) = NaN;

    % 数值指标
    relL2 = norm(Jk(valid) - Jref(valid)) / max(norm(Jref(valid)), eps);%L2 误差
    maxAbs = max(abs(Jk(valid) - Jref(valid)));
    corrVal = corr(Jk(valid), Jref(valid), 'Type','Pearson');%计算 Pearson 相关系数，衡量两者在有效区域内的线性相似度，范围通常在 [-1, 1]

    fprintf('  Case meshFraction=%g vs REF=%g: relL2=%.4g, maxAbs=%.4g, corr=%.4g\n', ...
        meshLambdaFraction_list(k), meshLambdaFraction_list(refIdx), relL2, maxAbs, corrVal);

    figure('Name', sprintf('DiffAbs - %g vs REF %g', meshLambdaFraction_list(k), meshLambdaFraction_list(refIdx)));
    imagesc(x_centers, y_centers, diffAbs);
    set(gca,'YDir','normal'); axis equal tight;
    xlim([patch_origin(1), patch_origin(1)+L]);
    ylim([patch_origin(2), patch_origin(2)+W]);
    title(sprintf('差分图 |J_k - J_ref| (mesh=%g vs ref=%g)', ...
        meshLambdaFraction_list(k), meshLambdaFraction_list(refIdx)));
    xlabel('X (m)'); ylabel('Y (m)');
    colorbar; colormap(parula);
end

%% 7. S11 对比（可选）
fprintf('\n--- 7. S11 对比绘图 ---\n');
figure('Name','S11 Compare');
hold on; grid on;
for k = 1:numMeshes
    rfplot(S11_all{k}, 1, 1);
end
title('S11 对比（三种 meshLambdaFraction）');
legend(compose('meshFrac=%g', meshLambdaFraction_list), 'Location','best');
hold off;
end