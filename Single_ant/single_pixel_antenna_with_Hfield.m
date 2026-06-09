function single_pixel_antenna_with_Hfield()
% single_pixel_antenna_with_Hfield
% -------------------------------------------------------------------------
% Single-antenna validation script for replacing surface current output with
% electric- and magnetic-field samples near the pixelated patch surface.
%
% Antenna construction follows generate_antenna_dataset_v10_with_current.m.
% The current() + scattered interpolation path is replaced by EHfields()
% evaluated directly on a 64x64 Cartesian grid.
% -------------------------------------------------------------------------
clear; clc; close all;

%% 0) Configuration copied from generate_antenna_dataset_v10_with_current.m
pixelResolution_N = 16;
fineN = 64;
overlap_mm = 5e-3;

geom.patch_L_mm = 14;
geom.patch_W_mm = 14;
geom.sub_thick_mm = 2.5;
geom.substrate_name = 'Air';
geom.board_L_mm = 30;
geom.board_W_mm = 30;
geom.feed_init_xy = [geom.patch_L_mm/4, 0];
geom.feed_diam_mm = min(geom.patch_L_mm/pixelResolution_N, ...
                        geom.patch_W_mm/pixelResolution_N) / 10;

sim_params.fmin_GHz = 8;
sim_params.fmax_GHz = 12;
sim_params.numFreqPoints = 41;
sim_params.pattern_fmin_GHz = 8.0;
sim_params.pattern_fmax_GHz = 12.0;
sim_params.pattern_step_GHz = 0.4;
sim_params.pattern_theta_deg = 1:3:360;
sim_params.pattern_phi_xoz = 0;
sim_params.pattern_phi_yoz = 90;

hfield_offset_m = -125e-3;
hfield_z_step_m = 0.1e-3;
rng(106);

assert(license('test','Antenna_Toolbox') == 1, ...
    'Antenna Toolbox license is required.');

fprintf('--- Build shared antenna parameters ---\n');
designParams = design_antenna_parameters(sim_params, geom, ...
    pixelResolution_N, fineN, overlap_mm);

%% 1) Build one test design and antenna
fprintf('--- Build one pixelated antenna ---\n');
feed_rc = designParams.feedPixelIdx;
designMatrix = generateSmoothDesignStable4Conn( ...
    pixelResolution_N, feed_rc, 0.80);
designMatrix(feed_rc(1), feed_rc(2)) = true;

pixel_indices = find(designMatrix);
patchShape = union_rectangles_batch(designParams.pixelShapes(pixel_indices), ...
    64, false);
ant = create_antenna_model(patchShape, feed_rc, designParams);

fprintf('  feed_rc = [%d, %d]\n', feed_rc(1), feed_rc(2));
fprintf('  metal fill = %.3f\n', nnz(designMatrix) / numel(designMatrix));
fprintf('  EH-field plane z = %.3f mm\n', ...
    (designParams.h + hfield_offset_m) * 1e3);

figure('Name','Single Pixel Antenna Geometry');
layout(ant);
title('Pixelated patch antenna');

%% 2) Optional S11 sweep, using the same full-wave setup as v10
fprintf('--- Run S11 sweep ---\n');
s = sparameters(ant, designParams.freq_sweep);
s11_mag = abs(squeeze(s.Parameters(1,1,:)));
[~, idx_min] = min(s11_mag);
f_eval = designParams.freq_sweep(idx_min);
fprintf('  minimum |S11| in sweep at %.3f GHz, |S11| = %.4g\n', ...
    f_eval / 1e9, s11_mag(idx_min));

%% 3) Direct EHfields sampling on 64x64 field grid
fprintf('--- Evaluate E/H fields with EHfields ---\n');
[E0, H0, gridInfo] = compute_ehfield_from_antenna(ant, f_eval, designParams, ...
    designParams.h + hfield_offset_m);

% Two extra z planes are used only for the 3-D Laplacian residual.
[Ezm, Hzm] = compute_ehfield_from_antenna(ant, f_eval, designParams, ...
    designParams.h + hfield_offset_m - hfield_z_step_m);
[Ezp, Hzp] = compute_ehfield_from_antenna(ant, f_eval, designParams, ...
    designParams.h + hfield_offset_m + hfield_z_step_m);

efieldData = pack_efield_channels(E0);
hfieldData = pack_hfield_channels(H0);
eLossTable = evaluate_field_physics_loss(E0, Ezm, Ezp, f_eval, ...
    gridInfo.dx, gridInfo.dy, hfield_z_step_m, 'E');
hLossTable = evaluate_field_physics_loss(H0, Hzm, Hzp, f_eval, ...
    gridInfo.dx, gridInfo.dy, hfield_z_step_m);

fprintf('\nE-field Helmholtz residual relative L2:\n');
disp(eLossTable);
fprintf('\nH-field Helmholtz residual relative L2:\n');
disp(hLossTable);

%% 4) Plot and save single-run validation output
Emag = sqrt(abs(E0.Ex).^2 + abs(E0.Ey).^2 + abs(E0.Ez).^2);
figure('Name','Electric Field Magnitude');
imagesc(gridInfo.x_centers * 1e3, gridInfo.y_centers * 1e3, Emag);
set(gca, 'YDir', 'normal');
axis equal tight;
xlabel('x (mm)');
ylabel('y (mm)');
title(sprintf('|E| at z=%.3f mm, f=%.3f GHz', gridInfo.z * 1e3, f_eval / 1e9));
colorbar;
colormap parula;

Hmag = sqrt(abs(H0.Hx).^2 + abs(H0.Hy).^2 + abs(H0.Hz).^2);
figure('Name','Magnetic Field Magnitude');
imagesc(gridInfo.x_centers * 1e3, gridInfo.y_centers * 1e3, Hmag);
set(gca, 'YDir', 'normal');
axis equal tight;
xlabel('x (mm)');
ylabel('y (mm)');
title(sprintf('|H| at z=%.3f mm, f=%.3f GHz', gridInfo.z * 1e3, f_eval / 1e9));
colorbar;
colormap parula;

outFile = fullfile(fileparts(mfilename('fullpath')), ...
    'single_pixel_antenna_Hfield_validation.mat');
save(outFile, 'designMatrix', 'feed_rc', 'designParams', 'f_eval', ...
    's11_mag', 'E0', 'H0', 'efieldData', 'hfieldData', 'gridInfo', ...
    'eLossTable', 'hLossTable', ...
    'hfield_offset_m', 'hfield_z_step_m');
fprintf('Saved validation output: %s\n', outFile);
end

%% ========================================================================
%%                       EH-field extraction and residuals
%% ========================================================================
function [E, H, gridInfo] = compute_ehfield_from_antenna(ant, f_now, designParams, z_now)
Xc = designParams.current_Xc;
Yc = designParams.current_Yc;
Zc = z_now * ones(size(Xc));
points = [Xc(:).'; Yc(:).'; Zc(:).'];

[e_raw, h_raw] = EHfields(ant, f_now, points);
if size(e_raw, 1) ~= 3
    e_raw = e_raw.';
end
if size(h_raw, 1) ~= 3
    h_raw = h_raw.';
end

fineN = designParams.fineN;
E.Ex = reshape(e_raw(1,:), fineN, fineN);
E.Ey = reshape(e_raw(2,:), fineN, fineN);
E.Ez = reshape(e_raw(3,:), fineN, fineN);
E.z = z_now;

H.Hx = reshape(h_raw(1,:), fineN, fineN);
H.Hy = reshape(h_raw(2,:), fineN, fineN);
H.Hz = reshape(h_raw(3,:), fineN, fineN);
H.z = z_now;

gridInfo.x_centers = designParams.current_x_centers;
gridInfo.y_centers = designParams.current_y_centers;
gridInfo.Xc = Xc;
gridInfo.Yc = Yc;
gridInfo.z = z_now;
gridInfo.dx = designParams.current_x_centers(2) - designParams.current_x_centers(1);
gridInfo.dy = designParams.current_y_centers(2) - designParams.current_y_centers(1);
end

function efieldData = pack_efield_channels(E)
efieldData = pack_field_channels(E, 'E');
end

function hfieldData = pack_hfield_channels(H)
hfieldData = pack_field_channels(H, 'H');
end

function fieldData = pack_field_channels(F, prefix)
% Channel order:
% 1:Fx_real, 2:Fx_imag, 3:Fy_real, 4:Fy_imag, 5:Fz_real, 6:Fz_imag.
Fx = F.([prefix 'x']);
Fy = F.([prefix 'y']);
Fz = F.([prefix 'z']);
fineN = size(Fx, 1);
fieldData = zeros(6, fineN, fineN, 'single');
fieldData(1,:,:) = single(real(Fx));
fieldData(2,:,:) = single(imag(Fx));
fieldData(3,:,:) = single(real(Fy));
fieldData(4,:,:) = single(imag(Fy));
fieldData(5,:,:) = single(real(Fz));
fieldData(6,:,:) = single(imag(Fz));
end

function lossTable = evaluate_field_physics_loss(F0, Fzm, Fzp, f_now, dx, dy, dz, prefix)
if nargin < 8
    prefix = 'H';
end
c0 = physconst('LightSpeed');
k0 = 2*pi*f_now / c0;

Fx = F0.([prefix 'x']);
Fy = F0.([prefix 'y']);
Fz = F0.([prefix 'z']);
Fxm = Fzm.([prefix 'x']);
Fym = Fzm.([prefix 'y']);
Fzm0 = Fzm.([prefix 'z']);
Fxp = Fzp.([prefix 'x']);
Fyp = Fzp.([prefix 'y']);
Fzp0 = Fzp.([prefix 'z']);

names = {sprintf('%sx_real', prefix); sprintf('%sx_imag', prefix); ...
         sprintf('%sy_real', prefix); sprintf('%sy_imag', prefix); ...
         sprintf('%sz_real', prefix); sprintf('%sz_imag', prefix)};
fields0 = {real(Fx), imag(Fx), real(Fy), imag(Fy), real(Fz), imag(Fz)};
fieldsm = {real(Fxm), imag(Fxm), real(Fym), imag(Fym), real(Fzm0), imag(Fzm0)};
fieldsp = {real(Fxp), imag(Fxp), real(Fyp), imag(Fyp), real(Fzp0), imag(Fzp0)};

loss2D = zeros(numel(names), 1);
loss3D = zeros(numel(names), 1);
for ii = 1:numel(names)
    psi0 = fields0{ii};
    psim = fieldsm{ii};
    psip = fieldsp{ii};
    loss2D(ii) = helmholtz_relative_l2_2d(psi0, k0, dx, dy);
    loss3D(ii) = helmholtz_relative_l2_3d(psi0, psim, psip, k0, dx, dy, dz);
end

lossTable = table(names, loss2D, loss3D, ...
    'VariableNames', {'component', 'relative_L2_2D', 'relative_L2_3D'});
end

function eps_l2 = helmholtz_relative_l2_2d(psi, k0, dx, dy)
center = psi(2:end-1, 2:end-1);
d2x = (psi(2:end-1, 3:end) - 2*center + psi(2:end-1, 1:end-2)) / dx^2;
d2y = (psi(3:end, 2:end-1) - 2*center + psi(1:end-2, 2:end-1)) / dy^2;
residual = d2x + d2y + k0^2 * center;
eps_l2 = relative_l2_from_residual(residual, k0^2 * center);
end

function eps_l2 = helmholtz_relative_l2_3d(psi0, psim, psip, k0, dx, dy, dz)
center = psi0(2:end-1, 2:end-1);
d2x = (psi0(2:end-1, 3:end) - 2*center + psi0(2:end-1, 1:end-2)) / dx^2;
d2y = (psi0(3:end, 2:end-1) - 2*center + psi0(1:end-2, 2:end-1)) / dy^2;
d2z = (psip(2:end-1, 2:end-1) - 2*center + psim(2:end-1, 2:end-1)) / dz^2;
residual = d2x + d2y + d2z + k0^2 * center;
eps_l2 = relative_l2_from_residual(residual, k0^2 * center);
end

function eps_l2 = relative_l2_from_residual(residual, reference)
valid = isfinite(residual) & isfinite(reference);
num = sum(abs(residual(valid)).^2, 'all');
den = sum(abs(reference(valid)).^2, 'all');
if den <= eps
    eps_l2 = NaN;
else
    eps_l2 = sqrt(num / den);
end
end

%% ========================================================================
%%                 Shared simulation / antenna-model helpers from v10
%% ========================================================================
function params = design_antenna_parameters(sim_params, geom, N, fineN, overlap_mm)
params.L = geom.patch_L_mm / 1e3;
params.W = geom.patch_W_mm / 1e3;
params.h = geom.sub_thick_mm / 1e3;

params.substrateMaterial = dielectric(geom.substrate_name);
params.substrateMaterial.Thickness = params.h;

params.ground = antenna.Rectangle('Length', geom.board_L_mm / 1e3, ...
                                  'Width',  geom.board_W_mm / 1e3, ...
                                  'Center', [0 0]);

params.pixelResolution_N = N;
params.fineN = fineN;
params.overlap = overlap_mm / 1e3;
params.patch_origin = [-params.L/2, -params.W/2];
pixel_L = params.L / N;
pixel_W = params.W / N;
params.pixel_L = pixel_L;
params.pixel_W = pixel_W;

params.pixelShapes = cell(N, N);
params.pixel_centers_x = params.patch_origin(1) + pixel_L/2 + (0:N-1) * pixel_L;
params.pixel_centers_y = params.patch_origin(2) + pixel_W/2 + (0:N-1) * pixel_W;
for r = 1:N
    for c = 1:N
        centerX = params.pixel_centers_x(c);
        centerY = params.pixel_centers_y(r);
        params.pixelShapes{r,c} = antenna.Rectangle( ...
            'Length', pixel_L + params.overlap, ...
            'Width',  pixel_W + params.overlap, ...
            'Center', [centerX, centerY]);
    end
end

feed_c_idx = floor((geom.feed_init_xy(1) - (-params.L/2)) / pixel_L) + 1;
feed_r_idx = floor((geom.feed_init_xy(2) - (-params.W/2)) / pixel_W) + 1;
params.feedPixelIdx = [max(1, min(N, feed_r_idx)), max(1, min(N, feed_c_idx))];

targetPixelShape = params.pixelShapes{params.feedPixelIdx(1), params.feedPixelIdx(2)};
params.finalFeedLocation = targetPixelShape.Center;
params.feedDiameter = geom.feed_diam_mm / 1e3;

f_start = sim_params.fmin_GHz * 1e9;
f_stop = sim_params.fmax_GHz * 1e9;
params.freq_sweep = linspace(f_start, f_stop, sim_params.numFreqPoints);
params.pattern_freqs = (sim_params.pattern_fmin_GHz : ...
                        sim_params.pattern_step_GHz : ...
                        sim_params.pattern_fmax_GHz) * 1e9;
params.current_freqs = params.pattern_freqs;

params.pattern_theta_deg = sim_params.pattern_theta_deg(:).';
params.pattern_phi_xoz = sim_params.pattern_phi_xoz;
params.pattern_phi_yoz = sim_params.pattern_phi_yoz;
params.pattern_num_channels = 4;
params.current_num_channels = 4;

x_edges = linspace(params.patch_origin(1) - params.overlap/2, ...
                   params.patch_origin(1) + params.L + params.overlap/2, fineN+1);
y_edges = linspace(params.patch_origin(2) - params.overlap/2, ...
                   params.patch_origin(2) + params.W + params.overlap/2, fineN+1);
params.current_x_centers = (x_edges(1:end-1) + x_edges(2:end)) / 2;
params.current_y_centers = (y_edges(1:end-1) + y_edges(2:end)) / 2;
[params.current_Xc, params.current_Yc] = meshgrid(params.current_x_centers, ...
    params.current_y_centers);
end

function ant = create_antenna_model(patchShape, feed_rc, params)
feed_rc = double(feed_rc);
N = params.pixelResolution_N;
pixel_L = params.L / N;
pixel_W = params.W / N;
startX = -params.L/2 + pixel_L/2;
startY = -params.W/2 + pixel_W/2;

cx = startX + (feed_rc(2)-1) * pixel_L;
cy = startY + (feed_rc(1)-1) * pixel_W;

ant = pcbStack( ...
    'BoardShape', params.ground, ...
    'BoardThickness', params.h, ...
    'Layers', {patchShape, params.substrateMaterial, params.ground}, ...
    'FeedDiameter', params.feedDiameter, ...
    'FeedLocations', [cx, cy, 1, 3]);
end

function shape = union_rectangles_batch(rectCells, batch, do_simplify)
if nargin < 2, batch = 64; end
if nargin < 3, do_simplify = true; end
ids = find(~cellfun('isempty', rectCells));
if isempty(ids)
    shape = [];
    return;
end

shape = rectCells{ids(1)};
cnt = 1;
for ii = 2:numel(ids)
    shape = shape + rectCells{ids(ii)};
    cnt = cnt + 1;
    if do_simplify && mod(cnt, batch) == 0
        try
            shape = shape.simplify;
        catch
        end
    end
end
end

function design = generateSmoothDesignStable4Conn(pixelN, feedPixelIdx, pMetal)
raw = rand(pixelN, pixelN) < pMetal;
[X, Y] = meshgrid(1:pixelN, 1:pixelN);
cx = (pixelN + 1) / 2;
cy = (pixelN + 1) / 2;
R2 = ((X - cx) / (0.42 * pixelN)).^2 + ((Y - cy) / (0.42 * pixelN)).^2;
maskCenter = R2 <= 1.0;

if pMetal < 0.45
    centerBoost = 0.02 + 0.04 * rand();
else
    centerBoost = 0.0;
end
tmp = rand(pixelN, pixelN) < centerBoost;
raw(maskCenter) = raw(maskCenter) | tmp(maskCenter);

r = feedPixelIdx(1);
c = feedPixelIdx(2);
raw(r, c) = 1;
bw = raw;

bw = imclose(bw, strel('square', 2));
bw = imopen(bw, strel('square', 2));
bw = bwareaopen(bw, 4, 4);
bw = bwmorph(bw, 'spur', 1);

bw(r, c) = 1;
bw = keepFeedConnectedComponent4(bw, feedPixelIdx);
bw(r, c) = 1;
design = logical(bw);
end

function bw2 = keepFeedConnectedComponent4(bw, feedPixelIdx)
r = feedPixelIdx(1);
c = feedPixelIdx(2);
bw(r, c) = 1;
CC = bwconncomp(bw, 4);
L = labelmatrix(CC);
lab = L(r, c);

bw2 = false(size(bw));
if lab > 0
    bw2(L == lab) = true;
end
bw2(r, c) = true;
end
