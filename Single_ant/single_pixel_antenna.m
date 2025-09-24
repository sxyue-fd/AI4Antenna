function single_pixel_antenna()
% SINGLE_PIXEL_ANTENNA
% 一个用于调试、最简化的单个随机像素天线建模与仿真脚本。

clc; clear; close all;

fprintf('--- 开始单天线调试脚本 ---\n');

%% 1. 用户配置 (硬编码)
centerFreq_GHz      = 2.45;
pixelResolution_N   = 16;
pixelFillFactor     = 0.6;
overlap_mm          = 0.2;
meshLambdaFraction  = 10;
freqSpan_GHz        = 0.4;
numFreqPoints       = 41;

% 使用固定的随机种子
rng(1); 

%% 2. 设计天线固定参数
fprintf('正在计算天线设计参数...\n');
f_center = centerFreq_GHz * 1e9;
c = physconst('LightSpeed');
lambda0 = c / f_center;
overlap = overlap_mm / 1000;

L = lambda0 / 2; 
W = L * 1.5; 
h = lambda0 / 50;
substrateMaterial = dielectric('Air');

extension = 12 * h; 
board_L = L + extension;
board_W = W + extension;
ground = antenna.Rectangle('Length', board_L, 'Width', board_W, 'Center', [0 0]);

pixel_L = L / pixelResolution_N;
pixel_W = W / pixelResolution_N;

pixelShapes = cell(pixelResolution_N, pixelResolution_N);
startX = -L/2 + pixel_L/2;
startY = -W/2 + pixel_W/2;
for r = 1:pixelResolution_N
    for c = 1:pixelResolution_N
        centerX = startX + (c-1)*pixel_L;
        centerY = startY + (r-1)*pixel_W;
        pixelShapes{r,c} = antenna.Rectangle('Length', pixel_L+overlap, 'Width', pixel_W+overlap, 'Center', [centerX, centerY]);
    end
end

% 【核心修正】步骤 1: 先计算出目标像素的索引
initialFeedLocation = [L/4, 0];
feed_c_idx = floor((initialFeedLocation(1) - (-L/2)) / pixel_L) + 1;
feed_r_idx = floor((initialFeedLocation(2) - (-W/2)) / pixel_W) + 1;
feedPixelIdx = [max(1, min(pixelResolution_N, feed_r_idx)), max(1, min(pixelResolution_N, feed_c_idx))];

% 【核心修正】步骤 2: 获取目标像素的中心坐标作为最终馈电位置
targetPixelShape = pixelShapes{feedPixelIdx(1), feedPixelIdx(2)};
finalFeedLocation = targetPixelShape.Center;

feedDiameter = min(pixel_L, pixel_W) / 10;

fprintf('参数计算完成。\n');
fprintf('  - 初始馈电点: (%.2f, %.2f) mm\n', initialFeedLocation(1)*1e3, initialFeedLocation(2)*1e3);
fprintf('  - 目标像素索引: (Row %d, Col %d)\n', feedPixelIdx(1), feedPixelIdx(2));
fprintf('  - 【最终馈电点 (已对齐到像素中心)】: (%.2f, %.2f) mm\n\n', finalFeedLocation(1)*1e3, finalFeedLocation(2)*1e3);

%% 3. 生成并组合随机像素贴片
fprintf('正在生成随机像素矩阵 (designVector)...\n');
designVector = rand(pixelResolution_N, pixelResolution_N) < pixelFillFactor;
designVector(feedPixelIdx(1), feedPixelIdx(2)) = 1; % 强制馈电点

fprintf('正在根据 designVector 组合天线形状...\n');
pixel_indices = find(designVector);
if isempty(pixel_indices)
    patchShape = pixelShapes{feedPixelIdx(1), feedPixelIdx(2)};
else
    patchShape = pixelShapes{pixel_indices(1)};
    for i = 2:length(pixel_indices)
        patchShape = patchShape + pixelShapes{pixel_indices(i)};
    end
end
fprintf('天线贴片形状组合完成。\n\n');

%% 4. 创建 pcbStack 对象
fprintf('正在创建 pcbStack 对象...\n');
ant = pcbStack();
ant.BoardShape = ground;
ant.BoardThickness = h;
ant.FeedDiameter = feedDiameter;
ant.Layers = {patchShape, substrateMaterial, ground}; 
% 【核心修正】步骤 3: 使用对齐到中心的最终坐标
ant.FeedLocations = [finalFeedLocation, 1, 3]; 
fprintf('pcbStack 对象创建完成。\n\n');

%% 5. 显示天线几何
fprintf('>>> 步骤 1: 显示天线几何 (调用 show)...\n');
figure;
show(ant);
title('天线几何结构 (Meshing前)');
fprintf('>>> 您现在应该能看到馈电探针精确地位于其所在像素的中心。\n\n');

%% 6. 进行网格剖分
fprintf('>>> 步骤 2: 剖分网格 (调用 mesh)...\n');
maxEdge = lambda0 / meshLambdaFraction;
try
    mesh(ant, 'MaxEdgeLength', maxEdge);
    fprintf('>>> 网格剖分成功！\n\n');
    
    fprintf('>>> 步骤 3: 显示剖分后的网格...\n');
    figure;
    mesh(ant);
    title('剖分后的网格');
    
catch ME
    fprintf(2, '>>> 网格剖分失败！错误信息如下:\n');
    fprintf(2, '%s\n', ME.message);
    fprintf('\n如果依然失败，问题可能比预想的更深层。\n');
    return;
end

%% 7. (可选) 进行S参数仿真
fprintf('>>> 步骤 4: 计算 S 参数 (调用 sparameters)...\n');
f_start = (centerFreq_GHz - freqSpan_GHz/2) * 1e9;
f_stop = (centerFreq_GHz + freqSpan_GHz/2) * 1e9;
freq_sweep = linspace(f_start, f_stop, numFreqPoints);
s = sparameters(ant, freq_sweep);

fprintf('S 参数计算完成。\n\n');

%% 8. 可视化结果
fprintf('>>> 步骤 5: 绘制 S11 结果...\n');
figure;
rfplot(s);
title('S11 仿真结果');
grid on;

fprintf('--- 调试脚本运行结束 ---\n');

end