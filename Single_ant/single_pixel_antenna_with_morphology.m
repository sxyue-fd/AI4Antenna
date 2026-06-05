function single_pixel_antenna_with_morphology()
% single_pixel_antenna_with_morphology.m
% 基于高斯随机场(GRF)与形态学操作的像素化天线生成与仿真脚本
% 采用 pcbStack 架构进行微带天线建模，完全解决 3D Boundary 报错问题

clc; clear; close all;
fprintf('--- 开始 GRF 像素化天线生成与仿真 ---\n');

%% =========================================================================
% 1. 用户配置参数区 (全局设定，方便统一修改)
% =========================================================================

% [频段与扫频参数]
centerFreq_GHz      = 10;      % 中心频率 (GHz)
freqSpan_GHz        = 4;      % 扫频带宽 (GHz)
numFreqPoints       = 21;       % 扫频点数

% [天线拓扑与网格参数]
pixelResolution_N   = 16;       % 像素网格维度 (N x N)
patchSize_mm        = 14;       % 贴片大小
pixelSize_mm        = patchSize_mm/pixelResolution_N;      % 单个离散像素物理边长 (mm)
metalFillFactor     = 0.6;      % 目标金属覆盖率 (0~1)
grfSigma            = 1.2;      % GRF高斯滤波平滑度 (越大金属斑块越连续)

% [PCB 基底物理参数]
substrateThickness  = 2.5;      % 介质基板厚度 (mm)
groundExtension_mm  = 4.0;      % 地板相比天线网格向外延伸的尺寸 (mm)
feedDiameter_mm     = 0.6;      % 馈电探针直径 (mm)

% [馈电点网格索引] (矩阵坐标：以左上角为 [Row=1, Col=1])
feedRow_idx         = 16;       % 馈电点所在行 (例如：放在最底部)
feedCol_idx         = 8;        % 馈电点所在列 (例如：放在中间)

% [仿真与控制参数]
meshLambdaFraction  = 15;       % 网格剖分精细度 (波长的几分之一)
randomSeed          = 42;       % 随机种子 (固定以保证可复现)

% =========================================================================

rng(randomSeed); 
c = physconst('LightSpeed');
f_center = centerFreq_GHz * 1e9;
lambda0 = c / f_center;

%% 2. 拓扑生成：高斯随机场 (GRF) 与二值化
fprintf('>>> 步骤 1: 生成 GRF 拓扑结构...\n');
R = randn(pixelResolution_N, pixelResolution_N); 
R_smooth = imgaussfilt(R, grfSigma);

% 根据目标覆盖率计算二值化阈值
thresh = quantile(R_smooth(:), 1 - metalFillFactor);
bw_matrix = R_smooth > thresh;

%% 3. 形态学约束：锚定、闭运算与连通域过滤
fprintf('>>> 步骤 2: 施加形态学操作与连通域物理约束...\n');

% 3.1 强制锚定馈电点 (必须为金属)
bw_matrix(feedRow_idx, feedCol_idx) = 1;
% 建议将馈电点上方的一个像素也强制赋 1，确保馈电处有稳固的物理连接
if feedRow_idx > 1, bw_matrix(feedRow_idx-1, feedCol_idx) = 1; end

% 3.2 形态学闭运算 (消除对角线仅顶点接触导致的奇异点)
se = strel('square', 2);
bw_closed = imclose(bw_matrix, se);

% 3.3 连通域过滤 (剥离无用的孤立碎片)
CC = bwconncomp(bw_closed, 4); % 4-连通保证严格物理接触
L = labelmatrix(CC);
feed_label = L(feedRow_idx, feedCol_idx); % 获取馈电点所在连通域
if feed_label > 0
    bw_final = (L == feed_label); 
else
    bw_final = bw_closed; % 异常保底
end

%% 4. 边界提取与物理坐标映射
fprintf('>>> 步骤 3: 提取平滑边界并映射到笛卡尔物理坐标...\n');
% 提取无内孔洞的绝对外边界
B = bwboundaries(bw_final, 'noholes');
boundary_pixels = B{1}; % 第一列是 Row (Y), 第二列是 Col (X)

% 坐标系转换逻辑
center_offset = (pixelResolution_N / 2) + 0.5;
x_coords_mm = (boundary_pixels(:, 2) - center_offset) * pixelSize_mm;
y_coords_mm = (center_offset - boundary_pixels(:, 1)) * pixelSize_mm;

% 【核心修复】：调用自定义函数，剔除所有共线冗余点，只保留 90度 拐角
[x_clean, y_clean] = clean_polygon(x_coords_mm, y_coords_mm);

feed_x_mm = (feedCol_idx - center_offset) * pixelSize_mm;
feed_y_mm = (center_offset - feedRow_idx) * pixelSize_mm;

%% 5. pcbStack 天线三维建模
fprintf('>>> 步骤 4: 构建 pcbStack 天线三维模型...\n');

% 5.1 创建天线顶层金属多边形 (传入清洗后的精简顶点)
patchShape = antenna.Polygon('Vertices', [x_clean, y_clean] * 1e-3);

% 5.2 定义介质基板
substrateMaterial = dielectric('FR4');
substrateMaterial.Thickness = substrateThickness * 1e-3;

% 5.3 定义底层地平面
gnd_size_mm = pixelResolution_N * pixelSize_mm + 2 * groundExtension_mm;
ground = antenna.Rectangle('Length', gnd_size_mm*1e-3, 'Width', gnd_size_mm*1e-3, 'Center', [0, 0]);

% 5.4 组装 pcbStack
ant = pcbStack;
ant.Name = 'GRF_Pixel_Antenna';
ant.BoardThickness = substrateMaterial.Thickness; 
ant.Layers = {patchShape, substrateMaterial, ground};
ant.FeedLocations = [feed_x_mm*1e-3, feed_y_mm*1e-3, 1, 3]; 
ant.FeedDiameter = feedDiameter_mm * 1e-3;

%% 6. 数据可视化输出
fprintf('>>> 步骤 5: 显示数据处理流水线可视化...\n');
figure('Position', [100, 200, 1400, 350], 'Name', 'GRF天线生成流水线');

% 子图 1: 原始 GRF 二值化
subplot(1, 4, 1);
imagesc(bw_matrix); colormap([1 1 1; 0 0.4 0.8]);
axis equal tight; grid on; set(gca,'YDir','reverse');
title('1. GRF 二值化 (含馈电锚定)');
hold on; plot(feedCol_idx, feedRow_idx, 'r*', 'MarkerSize', 10, 'LineWidth', 2); hold off;

% 子图 2: 闭运算
subplot(1, 4, 2);
imagesc(bw_closed);
axis equal tight; grid on; set(gca,'YDir','reverse');
title('2. 闭运算 (消除对角奇异点)');

% 子图 3: 连通域提取
subplot(1, 4, 3);
imagesc(bw_final);
axis equal tight; grid on; set(gca,'YDir','reverse');
title('3. 保留主连通域 (清除孤立碎片)');

% 子图 4: MATLAB 天线模型
subplot(1, 4, 4);
show(ant);
title('4. pcbStack 3D 模型');

%% 7. 网格剖分与电磁仿真
fprintf('>>> 步骤 6: 剖分网格并仿真 S11...\n');
figure('Position', [200, 150, 1000, 400], 'Name', '网格与仿真结果');

% 剖分网格并展示
subplot(1, 2, 1);
maxEdge = lambda0 / meshLambdaFraction;
mesh(ant, 'MaxEdgeLength', maxEdge);
title('结构网格剖分');

% S11 频域仿真
subplot(1, 2, 2);
freqs = linspace(centerFreq_GHz - freqSpan_GHz/2, centerFreq_GHz + freqSpan_GHz/2, numFreqPoints) * 1e9;
sp = sparameters(ant, freqs);
rfplot(sp);
title('S_{11} Return Loss (反射系数)');
grid on;

fprintf('--- 仿真执行完毕 ---\n');
end

% =========================================================================
% 局部辅助函数：清洗多边形共线冗余点
% =========================================================================
function [x_out, y_out] = clean_polygon(x_in, y_in)
    % 1. 移除相邻的完全重复点
    points = [x_in(:), y_in(:)];
    idx = [true; any(diff(points, 1, 1) ~= 0, 2)];
    points = points(idx, :);

    % 确保首尾闭合以便于循环处理
    if norm(points(1,:) - points(end,:)) > 1e-6
        points = [points; points(1,:)];
    end

    % 2. 移除共线点 (基于向量叉乘)
    keep = true(size(points, 1), 1);
    for i = 1:size(points, 1)-1
        if i == 1
            p_prev = points(end-1, :); % 首尾相接
        else
            p_prev = points(i-1, :);
        end
        p_curr = points(i, :);
        p_next = points(i+1, :);

        v1 = p_curr - p_prev;
        v2 = p_next - p_curr;
        cross_prod = v1(1)*v2(2) - v1(2)*v2(1);

        % 如果叉乘接近0，说明前、中、后三个点在一条直线上，中间点可安全删除
        if abs(cross_prod) < 1e-6
            keep(i) = false;
        end
    end
    points = points(keep, :);

    x_out = points(:, 1);
    y_out = points(:, 2);
end