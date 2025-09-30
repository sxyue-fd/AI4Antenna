%% 脚本：分析地板尺寸对贴片天线性能及S参数误差的影响
%
%  本脚本旨在分析一个中心频率为10GHz的探针馈电微带贴片天线的
%  地板尺寸变化对其S参数、网格数量和计算速度的影响。
%
%  新增功能：以三倍地板尺寸的S11作为基准，计算其他尺寸的
%            S11幅度（magnitude）的均方误差（MSE）。
%
%  天线参数:
%   - 中心频率: 10 GHz
%   - 贴片尺寸 (L x W): 15mm x 15mm
%   - 基板高度 (h): 1mm
%   - 基板材料: FR-4 (典型值)

clear; close all; clc;

%% 1. 定义天线和仿真常量
% --- 物理常量
c = physconst('lightspeed');
centerFreq = 10e9; % 中心频率 (Hz)
lambda = c / centerFreq; % 波长 (m)

% --- 天线几何参数 (单位: m)
patchLength = 15e-3;
patchWidth = 15e-3;
substrateHeight = 1e-3;

% --- 基板材料
substrateMaterial = dielectric('Air');

% --- 馈电点设置
feedOffset = [patchLength/4, 0]; 

% --- 仿真频率范围
freqRange = (8:0.1:12) * 1e9;

%% 2. 设置要测试的地板尺寸
groundPlaneMultipliers = [1, 1.5, 2, 2.6, 3]; 
numSims = length(groundPlaneMultipliers);

% --- 初始化结果存储变量
results = struct();
legendInfo = cell(1, numSims);

fprintf('开始进行仿真分析...\n');
fprintf('------------------------------------\n');

%% 3. 循环仿真不同地板尺寸的天线
for i = 1:numSims
    multiplier = groundPlaneMultipliers(i);
    
    % 计算当前循环的地板尺寸
    gpLength = multiplier * patchLength;
    gpWidth = multiplier * patchWidth;
    
    fprintf('正在仿真: 地板尺寸 = %.2fmm x %.2fmm (乘数: %.1f)...\n', gpLength*1000, gpWidth*1000, multiplier);
    
    % --- 创建天线对象
    ant = patchMicrostrip;
    ant.Length = patchLength;
    ant.Width = patchWidth;
    ant.Height = substrateHeight;
    ant.Substrate = substrateMaterial;
    ant.GroundPlaneLength = gpLength;
    ant.GroundPlaneWidth = gpWidth;
    ant.FeedOffset = feedOffset;
    
    % --- 开始计时
    tic;
    
    % --- 计算S参数
    s = sparameters(ant, freqRange);
    
    % --- 记录计算时间
    elapsedTime = toc;
    
    % --- 获取网格信息
    m = mesh(ant, 'MaxEdgeLength', lambda/20); % 使用统一的网格剖分标准
    numMeshElements = m.NumTriangles;
    
    % --- 存储结果
    results(i).multiplier = multiplier;
    results(i).groundPlaneSize = [gpLength, gpWidth];
    results(i).s_params = s;
    results(i).simulationTime = elapsedTime;
    results(i).meshCount = numMeshElements;
    
    legendInfo{i} = sprintf('GP Multiplier = %.1f', multiplier);
    
    fprintf('完成! 用时: %.2f 秒, 网格数量: %d\n', elapsedTime, numMeshElements);
    fprintf('------------------------------------\n');
end

%% 4. 分析和可视化结果

% --- 绘制S参数对比图
figure('Name', 'S11参数对比');
hold on;
for i = 1:numSims
    rfplot(results(i).s_params, 1, 1);
end
hold off;
grid on;
title('不同地板尺寸下的S11参数对比');
xlabel('频率 (GHz)');
ylabel('S11 (dB)');
legend(legendInfo, 'Location', 'best');
ylim([-30, 0]);

% --- 绘制网格数量和计算速度随地板尺寸的变化
figure('Name', '网格与计算时间分析');
multipliers = [results.multiplier];
meshCounts = [results.meshCount];
simTimes = [results.simulationTime];

yyaxis left;
plot(multipliers, meshCounts, '-o', 'LineWidth', 2, 'DisplayName', '网格数量');
ylabel('网格三角形数量');

yyaxis right;
plot(multipliers, simTimes, '-s', 'LineWidth', 2, 'DisplayName', '计算时间');
ylabel('计算时间 (秒)');

grid on;
title('网格数量和计算时间随地板尺寸的变化');
xlabel('地板尺寸乘数 (相对于贴片尺寸)');
legend('Location', 'best');

fprintf('所有仿真已完成，开始计算MSE误差...\n');

%% 5. 计算并可视化S11幅度的MSE
% --- 设定基准
referenceMultiplier = 3;
ref_idx = find([results.multiplier] == referenceMultiplier);
if isempty(ref_idx)
    error('错误：找不到基准乘数 %d 在仿真结果中。', referenceMultiplier);
end
fprintf('基准: 地板尺寸乘数 = %.1f\n', referenceMultiplier);

% --- 提取基准S11幅度数据
ref_s_params = results(ref_idx).s_params;
% squeeze() 用于移除维度为1的维度，便于计算
ref_s11_mag = abs(squeeze(ref_s_params.Parameters(1,1,:)));

% --- 计算每个结果相对于基准的MSE
mse_values = zeros(1, numSims);
for i = 1:numSims
    current_s_params = results(i).s_params;
    current_s11_mag = abs(squeeze(current_s_params.Parameters(1,1,:)));
    
    % 计算MSE
    mse_values(i) = mean((current_s11_mag - ref_s11_mag).^2);
end

% --- 打印MSE结果到命令行窗口
fprintf('\n--- S11幅度MSE计算结果 (基准乘数 = %.1f) ---\n', referenceMultiplier);
for i = 1:numSims
    fprintf('乘数: %4.1f | MSE: %e\n', results(i).multiplier, mse_values(i));
end
fprintf('------------------------------------------------\n');


% --- 绘制MSE结果图
figure('Name', 'S11幅度MSE分析');
bar(multipliers, mse_values);
grid on;
title(sprintf('S11幅度MSE (以乘数 %.1f 为基准)', referenceMultiplier));
xlabel('地板尺寸乘数 (相对于贴片尺寸)');
ylabel('均方误差 (MSE)');
% 在柱状图上显示数值
xtips = multipliers;
ytips = mse_values;
labels = string(num2str(mse_values', '%1.2e'));
text(xtips, ytips, labels, 'HorizontalAlignment','center', 'VerticalAlignment','bottom');
set(gca, 'YScale', 'log'); % 使用对数坐标轴，以便更好地观察小误差
fprintf('分析完成。\n');