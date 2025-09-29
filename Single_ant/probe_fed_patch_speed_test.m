function probe_fed_patch_speed_test
% 比较探针馈电方形贴片在四种设置下的计算时间与网格规模：
% A) 有限地 + PTFE (patchMicrostrip)
% B) 有限地 + 空气 (patchMicrostrip)
% C) 无限地 + 空气 (patchMicrostrip)
% D) 有限地 + PTFE (pcbStack)  <-- 叠层PCB路线
% B1) 复用天线B，计算S11
% E) 复用天线B，计算S11+电流
% F) 复用天线B，计算S11+方向图

%% 基本参数
f0 = 2.45e9;
c0 = physconst('LightSpeed');
f  = linspace(0.9*f0, 1.1*f0, 21);   % 频扫点适中，便于公平对比

% 统一厚度与经验尺寸
h_ptfe = 1.6e-3;  er_ptfe = 2.1;
h_air  = 1.6e-3;

[W_ptfe, L_ptfe] = estimate_patch_dims(f0, er_ptfe, h_ptfe, c0);
Lsq = mean([W_ptfe, L_ptfe]);  % 方形化
Wsq = Lsq;

% 探针位置（粗匹配）
feed_x = 0.3*Lsq; feed_y = 0;

% 有限地板尺寸控制：在介质中的有效波长 λ_eff ~ λ0/sqrt(εeff)
eps_eff = (er_ptfe+1)/2 + (er_ptfe-1)/2 * 1/sqrt(1 + 12*h_ptfe/Wsq);
lambda_eff = c0/(f0*sqrt(eps_eff));
margin = max(0.25*lambda_eff, 6*h_ptfe); % 兼顾薄板与波导效应
Gx_finite = Wsq + 2*margin;
Gy_finite = Wsq + 2*margin;

% 统一网格限值（按最高频率）
lambda0_min = c0/max(f);
maxEdge_air  = lambda0_min/20;
maxEdge_ptfe = (lambda0_min/sqrt(er_ptfe))/15;  % 介质内稍严一点

%% A) patchMicrostrip: 有限地 + PTFE
d_ptfe = dielectric('Teflon'); d_ptfe.Thickness = h_ptfe;
antA = patchMicrostrip(Length=Lsq, Width=Wsq, Substrate=d_ptfe, ...
    GroundPlaneLength=Gx_finite, GroundPlaneWidth=Gy_finite, ...
    FeedOffset=[feed_x, feed_y]);
[tA, ZA, mA] = time_case(antA, f, maxEdge_ptfe, 'A) patch: Finite GND + PTFE');

%% B) patchMicrostrip: 有限地 + 空气
d_air = dielectric('Air'); d_air.Thickness = h_air;
antB = patchMicrostrip(Length=Lsq, Width=Wsq, Substrate=d_air, ...
    GroundPlaneLength=Gx_finite, GroundPlaneWidth=Gy_finite, ...
    FeedOffset=[feed_x, feed_y]);
[tB, ZB, mB] = time_case(antB, f, maxEdge_air, 'B) patch: Finite GND + Air');

%% C) patchMicrostrip: 无限地 + 空气（镜像理论）
antC = patchMicrostrip(Length=Lsq, Width=Wsq, Substrate=d_air, ...
    GroundPlaneLength=Inf, GroundPlaneWidth=Inf, ...
    FeedOffset=[feed_x, feed_y]);
[tC, ZC, mC] = time_case(antC, f, maxEdge_air, 'C) patch: Infinite GND + Air');

%% D) pcbStack: 有限地 + PTFE
% 三层：TopMetal - PTFE - Ground
% 过孔/探针：用 FeedLocations 指定（[x, y, layerID]），并设ViaDiameter
board = antenna.Rectangle('Length', Gx_finite, 'Width', Gy_finite);
top   = antenna.Rectangle('Length', Wsq, 'Width', Wsq, 'Center', [0,0]);
gnd   = antenna.Rectangle('Length', Gx_finite, 'Width', Gy_finite, 'Center', [0,0]); 

d_ptfe = dielectric('Teflon'); d_ptfe.Thickness = h_ptfe;
stack = pcbStack;
stack.BoardShape = board;
stack.BoardThickness = h_ptfe;
stack.Layers = {top, d_ptfe, gnd};
stack.ViaDiameter = 1.0e-3;                 % 探针/过孔直径（可按需求改）
% Feed 在顶层(layer=1)；参考地在底层；单端馈电即可
stack.FeedLocations = [feed_x, feed_y, 1, 3];   % [x y layerID viaConnectionLayerID]
stack.FeedDiameter  = 1.0e-3;

[tD, ZD, mD] = time_case(stack, f, maxEdge_ptfe, 'D) pcbStack: Finite GND + PTFE');

%% E) 有限地 + 空气: 计算S11+电流
% 复用天线B (antB)
fprintf('\n--- 测试S11 ---\n');
tagB1 = 'B1) patch: Finite GND + Air (S11)';
antB1 = clone(antB);
tic;
impedance(antB1, f);
tB1 = toc;
fprintf('[%s] 用时：%.3f s\n', tagB1, tB1);

%% E) 有限地 + 空气: 计算S11+电流
% 复用天线B (antB)
fprintf('\n--- 测试S11+电流 ---\n');
tagE = 'E) patch: Finite GND + Air (S11+Current)';
antE = clone(antB);
tic;
impedance(antE, f);
current(antE, f0); % 计算电流
tE = toc;
fprintf('[%s] 用时：%.3f s\n', tagE, tE);

%% F) 有限地 + 空气: 计算方向图
% 复用天线B (antB)
fprintf('\n--- 测试S11+方向图 ---\n');
tagF = 'F) patch: Finite GND + Air (S11+Pattern)';
antF = clone(antB);
tic;
impedance(antF, f);
pattern(antF, f0); % 计算方向图
tF = toc;
fprintf('[%s] 用时：%.3f s\n', tagF, tF);

%% 汇总
fprintf('\n=== 结果汇总（统一网格限值）===\n');
print_result('A 有限地+PTFE (patch)', tA, mA);
print_result('B 有限地+空气 (patch)', tB, mB);
print_result('C 无限地+空气 (patch)', tC, mC);
print_result('D 有限地+PTFE (pcbStack)', tD, mD);
fprintf('复用天线B计算S11，用时 = %7.3f s\n', tB1);
fprintf('复用天线B计算S11+电流，用时 = %7.3f s\n', tE);
fprintf('复用天线B计算S11+方向图，用时 = %7.3f s\n', tF);

%% 简要 |S11| 对比
figure('Name','|S11| 对比'); hold on; grid on;
plot(f/1e9, db(abs(z2s11(ZA,50))), 'LineWidth', 1.3);
plot(f/1e9, db(abs(z2s11(ZB,50))), 'LineWidth', 1.3);
plot(f/1e9, db(abs(z2s11(ZC,50))), 'LineWidth', 1.3);
plot(f/1e9, db(abs(z2s11(ZD,50))), 'LineWidth', 1.3);
xlabel('Frequency (GHz)'); ylabel('|S_{11}| (dB)');
legend('A patch+PTFE','B patch+Air','C patch+Air ∞GND','D pcbStack+PTFE','Location','best');
title('|S_{11}| (probe-fed square patch)');
end

%% ====== 辅助函数 ======

function [W, L] = estimate_patch_dims(f0, er, h, c0)
W = c0/(2*f0) * sqrt(2/(er+1));
eps_eff = (er+1)/2 + (er-1)/2 * 1/sqrt(1 + 12*h/W);
dL = 0.412*h * ((eps_eff+0.3)*(W/h+0.264))/((eps_eff-0.258)*(W/h+0.8));
Leff = c0/(2*f0*sqrt(eps_eff));
L = Leff - 2*dL;
end

function [t_used, Z, m] = time_case(ant, f, maxEdge, tag)
% 统一网格控制 + 计时
try
    mesh(ant, 'MaxEdgeLength', maxEdge);
catch
    % 老版本没有 MeshReader/参数也没关系，继续
end
% 网格信息（有的对象只会有三角形）
try
    m = mesh(ant);
catch
    m = [];
end

tic;
Z = impedance(ant, f);
t_used = toc;
fprintf('[%s] 用时：%.3f s\n', tag, t_used);
end

function print_result(name, t_used, m)
fprintf('%-28s | 用时 = %7.3f s', name, t_used);
if ~isempty(m) && isprop(m,'Triangles'), nTri = size(m.Triangles,2); else, nTri=NaN; end
if ~isempty(m) && isprop(m,'Tetrahedra') && ~isempty(m.Tetrahedra)
    nTet = size(m.Tetrahedra,2);
else
    nTet = 0;
end
fprintf(' | 三角形(金属)≈ %s | 四面体(介质)≈ %s\n', num2str(nTri), num2str(nTet));
end

function s11 = z2s11(Z, Z0)
s11 = (Z - Z0)./(Z + Z0);
end
