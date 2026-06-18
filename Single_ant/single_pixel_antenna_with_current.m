function single_pixel_antenna_with_current()
% single_pixel_antenna_with_current
% -------------------------------------------------------------------------
% One pixelated antenna sample using the same main parameters and current
% pipeline as data_generation/Ant_data_gen_v1p0.m.
%
% It computes:
%   1) S11 magnitude over 8-12 GHz, 41 points.
%   2) Radiation pattern over 8:1:12 GHz, theta = 1:3:360 deg.
%   3) Surface current over 8:1:12 GHz, interpolated to [5,4,64,64].
%   4) MoM triangular mesh visualization and per-metal-pixel mesh counts.
% -------------------------------------------------------------------------

clc; close all;

%% 1) Configuration aligned with Ant_data_gen_v1p0.m
fprintf('=== Single pixelated antenna with current ===\n');

pixelResolution_N = 16;
fineN = 16;

geom.patch_L_mm = 14;
geom.patch_W_mm = 14;
geom.sub_thick_mm = 2.5;
geom.substrate_name = 'Air';
geom.board_L_mm = 30;
geom.board_W_mm = 30;
geom.feed_init_xy = [geom.patch_L_mm / 4, 0];
geom.feed_diam_mm = min(geom.patch_L_mm / pixelResolution_N, ...
                        geom.patch_W_mm / pixelResolution_N) / 10;

sim_params.fmin_GHz = 8;
sim_params.fmax_GHz = 12;
sim_params.numFreqPoints = 41;
sim_params.pattern_fmin_GHz = 8.0;
sim_params.pattern_fmax_GHz = 12.0;
sim_params.pattern_step_GHz = 1;
sim_params.pattern_theta_deg = 1:3:360;
sim_params.pattern_phi_xoz = 20;
sim_params.pattern_phi_yoz = 90;

overlap_mm = 5e-3;
meshLambdaFraction = 20;
randomSeed = 100;
metalFillFactor = 0.6;
minFinalFillRateForDemo = 0.20;
maxDesignTries = 200;
designMode = "dataset_like_random";  % "dataset_like_random", "full", "feed_only", or "random_connected"

rng(randomSeed);
params = design_antenna_parameters(sim_params, geom, pixelResolution_N, fineN, overlap_mm);

fprintf('Patch: %.1f x %.1f mm, board: %.1f x %.1f mm, h=%.1f mm\n', ...
    geom.patch_L_mm, geom.patch_W_mm, geom.board_L_mm, geom.board_W_mm, geom.sub_thick_mm);
fprintf('Pixel grid: %d x %d, current grid: %d x %d\n', ...
    pixelResolution_N, pixelResolution_N, fineN, fineN);
fprintf('Feed pixel [row,col] = [%d,%d]\n', params.feedPixelIdx(1), params.feedPixelIdx(2));

%% 2) Build one pixelated design
designMatrix = make_single_design(designMode, pixelResolution_N, params.feedPixelIdx, ...
    metalFillFactor, minFinalFillRateForDemo, maxDesignTries);
designMatrix(params.feedPixelIdx(1), params.feedPixelIdx(2)) = true;
fprintf('Generated design mode: %s, metal pixels: %d/%d (fill=%.3f)\n', ...
    designMode, nnz(designMatrix), numel(designMatrix), nnz(designMatrix)/numel(designMatrix));

pixel_indices = find(designMatrix);
if isempty(pixel_indices)
    error('No metal pixels in design.');
end

rectCells = params.pixelShapes(pixel_indices);
patchShape = union_rectangles_batch(rectCells, 64, false);
ant = create_antenna_model(patchShape, params.feedPixelIdx, params);

figure('Name', 'Pixel Design');
imagesc(designMatrix);
axis equal tight;
set(gca, 'YDir', 'normal');
colormap(gca, [1 1 1; 0.1 0.35 0.85]);
hold on;
plot(params.feedPixelIdx(2), params.feedPixelIdx(1), 'rp', 'MarkerSize', 12, 'LineWidth', 2);
hold off;
title('16x16 Pixel Design');
xlabel('Column');
ylabel('Row');

figure('Name', 'Antenna Geometry');
show(ant);
title('pcbStack Geometry');

%% 3) MoM mesh plot and S11
fprintf('\n--- Meshing and S11 ---\n');
lambda_min = physconst('LightSpeed') / max(params.freq_sweep);
maxEdgeLength = lambda_min / meshLambdaFraction;
meshInfo = mesh(ant, 'MaxEdgeLength', maxEdgeLength);
numTriangles = get_num_triangles(meshInfo);
fprintf('Total mesh triangles reported by mesh(): %g\n', numTriangles);

figure('Name', 'MoM Triangular Mesh');
mesh(ant, 'MaxEdgeLength', maxEdgeLength);
view(0, 90);
axis equal;
title(sprintf('MoM Triangular Mesh, total triangles = %g', numTriangles));

fprintf('Computing S11 on %d points...\n', numel(params.freq_sweep));
s = sparameters(ant, params.freq_sweep);
s11_mag = abs(squeeze(s.Parameters(1, 1, :)));
s11_db = 20 * log10(s11_mag + eps);

figure('Name', 'S11');
plot(params.freq_sweep / 1e9, s11_db, 'LineWidth', 1.6);
grid on;
xlabel('Frequency (GHz)');
ylabel('|S11| (dB)');
title('S11, 8-12 GHz');

%% 4) Pattern, same channels as Ant_data_gen_v1p0.m
fprintf('\n--- Radiation pattern ---\n');
pattern_data = compute_pattern_from_antenna(ant, params);
fprintf('pattern_data size = [%s]\n', shape_to_text(size(pattern_data)));

midFreqIdx = ceil(numel(params.pattern_freqs) / 2);
plot_pattern_summary(pattern_data, params, midFreqIdx);

%% 5) Current interpolation to [Fc,4,64,64]
fprintf('\n--- Surface current ---\n');
[current_data, current_debug] = compute_current_from_antenna(ant, designMatrix, params);
fprintf('current_data size = [%s]\n', shape_to_text(size(current_data)));

plot_current_multires_summary(ant, current_debug, params, designMatrix, midFreqIdx);

%% 6) Mesh triangles per metal pixel
fprintf('\n--- Patch mesh count per metal pixel ---\n');
meshStats = compute_pixel_mesh_counts(current_debug.tri_centroids_by_freq{midFreqIdx}, ...
    designMatrix, params);

fprintf('Patch-layer triangles at %.1f GHz: %d\n', ...
    params.current_freqs(midFreqIdx) / 1e9, meshStats.totalPatchTriangles);
fprintf('Metal pixels: %d\n', meshStats.numMetalPixels);
fprintf('Average triangles per metal pixel: %.3f\n', meshStats.avgTrianglesPerMetalPixel);
fprintf('Median triangles per metal pixel: %.3f\n', meshStats.medianTrianglesPerMetalPixel);
fprintf('Min/Max triangles per metal pixel: %d / %d\n', ...
    meshStats.minTrianglesPerMetalPixel, meshStats.maxTrianglesPerMetalPixel);

disp('Per-metal-pixel triangle counts:');
disp(meshStats.table);

figure('Name', 'Triangles Per Metal Pixel');
countsForPlot = meshStats.counts;
countsForPlot(~designMatrix) = NaN;
imagesc(countsForPlot);
axis equal tight;
set(gca, 'YDir', 'normal');
colorbar;
title(sprintf('Patch-Layer Triangles per Metal Pixel @ %.1f GHz', ...
    params.current_freqs(midFreqIdx) / 1e9));
xlabel('Column');
ylabel('Row');

fprintf('\nDone.\n');

end

%% -------------------------------------------------------------------------
% Local helpers
%% -------------------------------------------------------------------------

function designMatrix = make_single_design(mode, N, feed_rc, fillFactor, minFinalFillRate, maxTries)
switch string(mode)
    case "full"
        designMatrix = true(N, N);
    case "feed_only"
        designMatrix = false(N, N);
        designMatrix(feed_rc(1), feed_rc(2)) = true;
    case "random_connected"
        raw = rand(N, N) < fillFactor;
        raw(feed_rc(1), feed_rc(2)) = true;
        designMatrix = keep_feed_connected_component(raw, feed_rc);
        if nnz(designMatrix) < 4
            r1 = max(1, feed_rc(1)-1); r2 = min(N, feed_rc(1)+1);
            c1 = max(1, feed_rc(2)-1); c2 = min(N, feed_rc(2)+1);
            designMatrix(r1:r2, c1:c2) = true;
        end
    case "dataset_like_random"
        designMatrix = [];
        targetMetalPixels = ceil(minFinalFillRate * N * N);
        cfg.min_area_keep = 4;

        for tries = 1:maxTries
            pMetal = 0.30 + (0.50 - 0.30) * rand();
            raw = rand(N, N) < pMetal;
            raw(feed_rc(1), feed_rc(2)) = true;
            candidate = morphology_repair_single(raw, feed_rc, cfg, N);

            if nnz(candidate) >= targetMetalPixels
                designMatrix = candidate;
                fprintf('Accepted morphology-repaired design after %d tries, raw p=%.3f.\n', ...
                    tries, pMetal);
                break;
            end
        end

        if isempty(designMatrix)
            warning('Could not reach fill target after %d tries. Using compact fallback block.', maxTries);
            designMatrix = compact_connected_fallback(N, feed_rc, targetMetalPixels);
        end
    otherwise
        error('Unknown designMode: %s', mode);
end
end

function txt = shape_to_text(shapeVec)
txt = strtrim(sprintf('%d ', shapeVec));
end

function pheno = morphology_repair_single(raw, feed_rc, cfg, pixelN)
bw = logical(raw);
[X, Y] = meshgrid(1:pixelN, 1:pixelN);
cx = (pixelN + 1) / 2;
cy = (pixelN + 1) / 2;

dist_to_center = sqrt((feed_rc(2)-cx)^2 + (feed_rc(1)-cy)^2);
if dist_to_center > 0.32 * pixelN
    R2 = ((X-cx)/(0.42*pixelN)).^2 + ((Y-cy)/(0.42*pixelN)).^2;
    maskCenter = R2 <= 1.0;
    centerBoost = 0.02 + 0.04 * rand();
    tmp = rand(pixelN, pixelN) < centerBoost;
    bw(maskCenter) = bw(maskCenter) | tmp(maskCenter);
end

r = feed_rc(1);
c = feed_rc(2);
bw(r, c) = true;
bw = imclose(bw, strel('square', 2));
bw = imopen(bw, strel('square', 2));
bw = bwareaopen(bw, cfg.min_area_keep, 4);
bw = bwmorph(bw, 'spur', 1);
bw(r, c) = true;
bw = keep_feed_connected_component(bw, feed_rc);
bw(r, c) = true;
pheno = logical(bw);
end

function designMatrix = compact_connected_fallback(N, feed_rc, targetMetalPixels)
designMatrix = false(N, N);
designMatrix(feed_rc(1), feed_rc(2)) = true;
[X, Y] = meshgrid(1:N, 1:N);
dist2 = (Y - feed_rc(1)).^2 + (X - feed_rc(2)).^2;
[~, order] = sort(dist2(:), 'ascend');
designMatrix(order(1:min(targetMetalPixels, numel(order)))) = true;
end

function out = keep_feed_connected_component(mask, feed_rc)
N = size(mask, 1);
out = false(size(mask));
if ~mask(feed_rc(1), feed_rc(2))
    return;
end

queue = zeros(numel(mask), 2);
head = 1;
tail = 1;
queue(tail, :) = feed_rc;
out(feed_rc(1), feed_rc(2)) = true;

dirs = [1 0; -1 0; 0 1; 0 -1];
while head <= tail
    p = queue(head, :);
    head = head + 1;
    for k = 1:4
        rr = p(1) + dirs(k, 1);
        cc = p(2) + dirs(k, 2);
        if rr < 1 || rr > N || cc < 1 || cc > N
            continue;
        end
        if mask(rr, cc) && ~out(rr, cc)
            tail = tail + 1;
            queue(tail, :) = [rr, cc];
            out(rr, cc) = true;
        end
    end
end
end

function params = design_antenna_parameters(sim_params, geom, N, fineN, overlap_mm)
params.L = geom.patch_L_mm / 1e3;
params.W = geom.patch_W_mm / 1e3;
params.h = geom.sub_thick_mm / 1e3;
params.substrateMaterial = dielectric(geom.substrate_name);
params.substrateMaterial.Thickness = params.h;
params.ground = antenna.Rectangle('Length', geom.board_L_mm / 1e3, ...
                                  'Width', geom.board_W_mm / 1e3, ...
                                  'Center', [0 0]);
params.pixelResolution_N = N;
params.fineN = fineN;
params.overlap = overlap_mm / 1e3;
params.patch_origin = [-params.L/2, -params.W/2];
params.pixel_L = params.L / N;
params.pixel_W = params.W / N;

params.pixel_centers_x = params.patch_origin(1) + params.pixel_L/2 + (0:N-1)*params.pixel_L;
params.pixel_centers_y = params.patch_origin(2) + params.pixel_W/2 + (0:N-1)*params.pixel_W;

params.pixelShapes = cell(N, N);
for rr = 1:N
    for cc = 1:N
        params.pixelShapes{rr, cc} = antenna.Rectangle( ...
            'Length', params.pixel_L + params.overlap, ...
            'Width', params.pixel_W + params.overlap, ...
            'Center', [params.pixel_centers_x(cc), params.pixel_centers_y(rr)]);
    end
end

feed_c_idx = floor((geom.feed_init_xy(1) - (-params.L/2)) / params.pixel_L) + 1;
feed_r_idx = floor((geom.feed_init_xy(2) - (-params.W/2)) / params.pixel_W) + 1;
params.feedPixelIdx = [max(1, min(N, feed_r_idx)), max(1, min(N, feed_c_idx))];
targetPixelShape = params.pixelShapes{params.feedPixelIdx(1), params.feedPixelIdx(2)};
params.finalFeedLocation = targetPixelShape.Center;
params.feedDiameter = geom.feed_diam_mm / 1e3;

params.freq_sweep = linspace(sim_params.fmin_GHz * 1e9, ...
                             sim_params.fmax_GHz * 1e9, ...
                             sim_params.numFreqPoints);
params.pattern_freqs = (sim_params.pattern_fmin_GHz : ...
                        sim_params.pattern_step_GHz : ...
                        sim_params.pattern_fmax_GHz) * 1e9;
params.current_freqs = params.pattern_freqs;
params.pattern_theta_deg = sim_params.pattern_theta_deg(:).';
params.pattern_phi_xoz = sim_params.pattern_phi_xoz;
params.pattern_phi_yoz = sim_params.pattern_phi_yoz;
params.current_num_channels = 4;

x_edges = linspace(params.patch_origin(1) - params.overlap/2, ...
                   params.patch_origin(1) + params.L + params.overlap/2, fineN+1);
y_edges = linspace(params.patch_origin(2) - params.overlap/2, ...
                   params.patch_origin(2) + params.W + params.overlap/2, fineN+1);
params.current_x_centers = (x_edges(1:end-1) + x_edges(2:end)) / 2;
params.current_y_centers = (y_edges(1:end-1) + y_edges(2:end)) / 2;
[params.current_Xc, params.current_Yc] = meshgrid(params.current_x_centers, params.current_y_centers);
end

function ant = create_antenna_model(patchShape, feed_rc, params)
feed_rc = double(feed_rc);
cx = params.pixel_centers_x(feed_rc(2));
cy = params.pixel_centers_y(feed_rc(1));

ant = pcbStack( ...
    'BoardShape', params.ground, ...
    'BoardThickness', params.h, ...
    'Layers', {patchShape, params.substrateMaterial, params.ground}, ...
    'FeedDiameter', params.feedDiameter, ...
    'FeedLocations', [cx, cy, 1, 3]);
end

function shape = union_rectangles_batch(rectCells, batch, do_simplify)
if nargin < 2, batch = 64; end
if nargin < 3, do_simplify = false; end

shape = rectCells{1};
for i = 2:numel(rectCells)
    shape = shape + rectCells{i};
    if do_simplify && mod(i, batch) == 0
        try
            shape = simplify(shape);
        catch
        end
    end
end
end

function nTri = get_num_triangles(meshInfo)
nTri = NaN;
try
    if isprop(meshInfo, 'NumTriangles')
        nTri = meshInfo.NumTriangles;
    elseif isprop(meshInfo, 'Triangles')
        nTri = size(meshInfo.Triangles, 2);
    end
catch
end
end

function pattern_data = compute_pattern_from_antenna(ant, params)
Fp = numel(params.pattern_freqs);
T = numel(params.pattern_theta_deg);
theta_deg = params.pattern_theta_deg;
pattern_data = zeros(Fp, 4, T, 'single');

for kf = 1:Fp
    f_pat = params.pattern_freqs(kf);
    pat_xoz_gth = pattern(ant, f_pat, params.pattern_phi_xoz, theta_deg, ...
        'Type', 'gain', 'Polarization', 'V');
    pat_xoz_gph = pattern(ant, f_pat, params.pattern_phi_xoz, theta_deg, ...
        'Type', 'gain', 'Polarization', 'H');
    pat_yoz_gth = pattern(ant, f_pat, params.pattern_phi_yoz, theta_deg, ...
        'Type', 'gain', 'Polarization', 'V');
    pat_yoz_gph = pattern(ant, f_pat, params.pattern_phi_yoz, theta_deg, ...
        'Type', 'gain', 'Polarization', 'H');

    pattern_data(kf, 1, :) = single(10.^(pat_xoz_gth(:) / 10));
    pattern_data(kf, 2, :) = single(10.^(pat_xoz_gph(:) / 10));
    pattern_data(kf, 3, :) = single(10.^(pat_yoz_gth(:) / 10));
    pattern_data(kf, 4, :) = single(10.^(pat_yoz_gph(:) / 10));
end
end

function plot_pattern_summary(pattern_data, params, freqIdx)
theta = params.pattern_theta_deg;
figure('Name', 'Pattern Summary');
tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
names = {'XOZ Gain theta', 'XOZ Gain phi', 'YOZ Gain theta', 'YOZ Gain phi'};
for ch = 1:4
    nexttile;
    plot(theta, squeeze(pattern_data(freqIdx, ch, :)), 'LineWidth', 1.3);
    grid on;
    title(sprintf('%s @ %.1f GHz', names{ch}, params.pattern_freqs(freqIdx)/1e9));
    xlabel('Theta (deg)');
    ylabel('Linear gain');
end
end

function [current_data, debug] = compute_current_from_antenna(ant, designMatrix, params)
Fc = numel(params.current_freqs);
fineN = params.fineN;
current_data = zeros(Fc, 4, fineN, fineN, 'single');
metalMask64 = build_current_metal_mask(designMatrix, params);
debug.metalMask64 = metalMask64;
debug.tri_centroids_by_freq = cell(Fc, 1);
debug.Jx_by_freq = cell(Fc, 1);
debug.Jy_by_freq = cell(Fc, 1);

for kf = 1:Fc
    f_now = params.current_freqs(kf);
    fprintf('  current() at %.1f GHz...\n', f_now / 1e9);
    [J_surface, tri_centroids] = current(ant, f_now);
    [Jx64, Jy64, tri_keep, Jx_keep, Jy_keep] = interpolate_current_to_grid( ...
        J_surface, tri_centroids, metalMask64, params);
    debug.tri_centroids_by_freq{kf} = tri_keep;
    debug.Jx_by_freq{kf} = Jx_keep;
    debug.Jy_by_freq{kf} = Jy_keep;

    current_data(kf, 1, :, :) = single(real(Jx64));
    current_data(kf, 2, :, :) = single(imag(Jx64));
    current_data(kf, 3, :, :) = single(real(Jy64));
    current_data(kf, 4, :, :) = single(imag(Jy64));
end
end

function metalMask64 = build_current_metal_mask(designMatrix, params)
Xc = params.current_Xc;
Yc = params.current_Yc;
metalMask64 = false(size(Xc));
halfL = (params.pixel_L + params.overlap) / 2;
halfW = (params.pixel_W + params.overlap) / 2;
N = params.pixelResolution_N;

for rr = 1:N
    for cc = 1:N
        if ~designMatrix(rr, cc)
            continue;
        end
        cx = params.pixel_centers_x(cc);
        cy = params.pixel_centers_y(rr);
        inRect = (Xc >= (cx - halfL)) & (Xc <= (cx + halfL)) & ...
                 (Yc >= (cy - halfW)) & (Yc <= (cy + halfW));
        metalMask64 = metalMask64 | inRect;
    end
end
end

function [Jx64, Jy64, tri_keep, Jx_keep, Jy_keep] = interpolate_current_to_grid( ...
    J_surface, tri_centroids, metalMask64, params)
if size(J_surface, 1) ~= 3
    J_surface = J_surface.';
end
if size(tri_centroids, 1) ~= 3
    tri_centroids = tri_centroids.';
end

Jx = J_surface(1, :).';
Jy = J_surface(2, :).';
x_scatter = tri_centroids(1, :).';
y_scatter = tri_centroids(2, :).';
z_scatter = tri_centroids(3, :).';

zPatch = params.h;
tolZ = 1e-8;
isPatchTri = abs(z_scatter - zPatch) < tolZ;
inPatchXY = (x_scatter >= params.patch_origin(1) - params.overlap/2) & ...
            (x_scatter <= params.patch_origin(1) + params.L + params.overlap/2) & ...
            (y_scatter >= params.patch_origin(2) - params.overlap/2) & ...
            (y_scatter <= params.patch_origin(2) + params.W + params.overlap/2);
keep = isPatchTri & inPatchXY;

x_scatter = x_scatter(keep);
y_scatter = y_scatter(keep);
Jx = Jx(keep);
Jy = Jy(keep);
tri_keep = [x_scatter.'; y_scatter.'; z_scatter(keep).'];
Jx_keep = Jx;
Jy_keep = Jy;

if isempty(x_scatter)
    Jx64 = zeros(params.fineN, params.fineN);
    Jy64 = zeros(params.fineN, params.fineN);
    return;
end

XY = [x_scatter, y_scatter];
[~, ia] = unique(XY, 'rows', 'stable');
x_u = x_scatter(ia);
y_u = y_scatter(ia);
Jx_u = Jx(ia);
Jy_u = Jy(ia);

Jx64 = interpolate_complex_scattered(x_u, y_u, Jx_u, params.current_Xc, params.current_Yc);
Jy64 = interpolate_complex_scattered(x_u, y_u, Jy_u, params.current_Xc, params.current_Yc);

Jx64(~metalMask64) = 0;
Jy64(~metalMask64) = 0;
Jx64(~isfinite(Jx64)) = 0;
Jy64(~isfinite(Jy64)) = 0;
end

function Vq = interpolate_complex_scattered(x, y, v, Xq, Yq)
if isscalar(x)
    Vq = complex(zeros(size(Xq)), zeros(size(Xq)));
    [~, idx] = min((Xq(:)-x(1)).^2 + (Yq(:)-y(1)).^2);
    Vq(idx) = v(1);
    return;
end

method = 'natural';
Fre = scatteredInterpolant(x, y, real(v), method, 'none');
Fim = scatteredInterpolant(x, y, imag(v), method, 'none');
Vq = Fre(Xq, Yq) + 1i * Fim(Xq, Yq);
Vq(~isfinite(Vq)) = 0;
end

function plot_current_multires_summary(ant, debug, params, designMatrix, freqIdx)
tri_centroids = debug.tri_centroids_by_freq{freqIdx};
Jx_tri = debug.Jx_by_freq{freqIdx};
Jy_tri = debug.Jy_by_freq{freqIdx};

if size(tri_centroids, 1) ~= 3
    tri_centroids = tri_centroids.';
end

Jmag_tri = sqrt(abs(Jx_tri).^2 + abs(Jy_tri).^2);

gridNs = [16, 32, 64];
gridResults = cell(numel(gridNs), 1);
allVals = Jmag_tri(isfinite(Jmag_tri));

for i = 1:numel(gridNs)
    gridParams = make_current_grid_params(params, gridNs(i));
    metalMask = build_current_metal_mask(designMatrix, gridParams);
    [Jx_grid, Jy_grid] = interpolate_kept_current_to_grid( ...
        tri_centroids, Jx_tri, Jy_tri, metalMask, gridParams);
    Jmag_grid = sqrt(abs(Jx_grid).^2 + abs(Jy_grid).^2);
    Jmag_grid(~metalMask) = NaN;

    gridResults{i} = struct( ...
        'N', gridNs(i), ...
        'x_centers', gridParams.current_x_centers, ...
        'y_centers', gridParams.current_y_centers, ...
        'Jmag', Jmag_grid);
    allVals = [allVals; Jmag_grid(isfinite(Jmag_grid))]; %#ok<AGROW>
end

if isempty(allVals)
    colorLim = [0, 1];
else
    finiteVals = allVals(isfinite(allVals));
    upperLim = percentile_value(finiteVals, 98);
    colorLim = [0, upperLim];
    if colorLim(1) == colorLim(2)
        colorLim(2) = colorLim(1) + eps;
    end
end

figure('Name', 'Fig6 Current Interpolation Resolution Compare');
t = tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
title(t, sprintf('Current |J| Comparison @ %.1f GHz', params.current_freqs(freqIdx)/1e9));

ax1 = nexttile;
tmpFig = figure('Visible', 'off');
try
    current(ant, params.current_freqs(freqIdx), direction='on');
catch
    current(ant, params.current_freqs(freqIdx));
end
axTmp = gca;
copiedChildren = copyobj(allchild(axTmp), ax1);
close(tmpFig);
scale_graphics_xy(copiedChildren, 1e3);
view(ax1, 0, 90);
axis(ax1, 'equal');
axis(ax1, 'tight');
xlim(ax1, (params.patch_origin(1) + [0, params.L]) * 1e3);
ylim(ax1, (params.patch_origin(2) + [0, params.W]) * 1e3);
clim(ax1, colorLim);
colormap(ax1, parula);
colorbar(ax1);
title(ax1, sprintf('Original MoM Current (%d patch centroids)', numel(Jmag_tri)));
xlabel(ax1, 'X (mm)');
ylabel(ax1, 'Y (mm)');

for i = 1:numel(gridResults)
    ax = nexttile;
    item = gridResults{i};
    imagesc(ax, item.x_centers * 1e3, item.y_centers * 1e3, item.Jmag);
    set(ax, 'YDir', 'normal');
    axis(ax, 'equal');
    axis(ax, 'tight');
    xlim(ax, (params.patch_origin(1) + [0, params.L]) * 1e3);
    ylim(ax, (params.patch_origin(2) + [0, params.W]) * 1e3);
    clim(ax, colorLim);
    colormap(ax, parula);
    colorbar(ax);
    title(ax, sprintf('%d x %d Regular Grid', item.N, item.N));
    xlabel(ax, 'X (mm)');
    ylabel(ax, 'Y (mm)');
end
end

function value = percentile_value(values, pct)
values = sort(values(:));
values = values(isfinite(values));
if isempty(values)
    value = 1;
    return;
end
idx = max(1, min(numel(values), ceil((pct / 100) * numel(values))));
value = values(idx);
if value <= 0
    value = max(values);
end
if value <= 0
    value = 1;
end
end

function scale_graphics_xy(objs, scaleFactor)
for i = 1:numel(objs)
    obj = objs(i);
    if isprop(obj, 'XData')
        try
            obj.XData = obj.XData * scaleFactor;
        catch
        end
    end
    if isprop(obj, 'YData')
        try
            obj.YData = obj.YData * scaleFactor;
        catch
        end
    end
    if isprop(obj, 'Children')
        scale_graphics_xy(obj.Children, scaleFactor);
    end
end
end

function gridParams = make_current_grid_params(params, gridN)
gridParams = params;
gridParams.fineN = gridN;
x_edges = linspace(params.patch_origin(1) - params.overlap/2, ...
                   params.patch_origin(1) + params.L + params.overlap/2, gridN+1);
y_edges = linspace(params.patch_origin(2) - params.overlap/2, ...
                   params.patch_origin(2) + params.W + params.overlap/2, gridN+1);
gridParams.current_x_centers = (x_edges(1:end-1) + x_edges(2:end)) / 2;
gridParams.current_y_centers = (y_edges(1:end-1) + y_edges(2:end)) / 2;
[gridParams.current_Xc, gridParams.current_Yc] = meshgrid( ...
    gridParams.current_x_centers, gridParams.current_y_centers);
end

function [Jx_grid, Jy_grid] = interpolate_kept_current_to_grid( ...
    tri_centroids, Jx, Jy, metalMask, params)
if size(tri_centroids, 1) ~= 3
    tri_centroids = tri_centroids.';
end

x_scatter = tri_centroids(1, :).';
y_scatter = tri_centroids(2, :).';

if isempty(x_scatter)
    Jx_grid = nan(size(metalMask));
    Jy_grid = nan(size(metalMask));
    return;
end

XY = [x_scatter, y_scatter];
[~, ia] = unique(XY, 'rows', 'stable');
x_u = x_scatter(ia);
y_u = y_scatter(ia);
Jx_u = Jx(ia);
Jy_u = Jy(ia);

Jx_grid = interpolate_complex_scattered(x_u, y_u, Jx_u, params.current_Xc, params.current_Yc);
Jy_grid = interpolate_complex_scattered(x_u, y_u, Jy_u, params.current_Xc, params.current_Yc);
Jx_grid(~metalMask) = NaN;
Jy_grid(~metalMask) = NaN;
Jx_grid(~isfinite(Jx_grid)) = NaN;
Jy_grid(~isfinite(Jy_grid)) = NaN;
end

function stats = compute_pixel_mesh_counts(tri_centroids, designMatrix, params)
if size(tri_centroids, 1) ~= 3
    tri_centroids = tri_centroids.';
end

N = params.pixelResolution_N;
counts = zeros(N, N);
x = tri_centroids(1, :).';
y = tri_centroids(2, :).';

c_idx = floor((x - params.patch_origin(1)) / params.pixel_L) + 1;
r_idx = floor((y - params.patch_origin(2)) / params.pixel_W) + 1;
valid = r_idx >= 1 & r_idx <= N & c_idx >= 1 & c_idx <= N;
r_idx = r_idx(valid);
c_idx = c_idx(valid);

for i = 1:numel(r_idx)
    if designMatrix(r_idx(i), c_idx(i))
        counts(r_idx(i), c_idx(i)) = counts(r_idx(i), c_idx(i)) + 1;
    end
end

metalCounts = counts(designMatrix);
[rows, cols] = find(designMatrix);
stats.counts = counts;
stats.totalPatchTriangles = sum(metalCounts);
stats.numMetalPixels = numel(metalCounts);
stats.avgTrianglesPerMetalPixel = mean(metalCounts);
stats.medianTrianglesPerMetalPixel = median(metalCounts);
stats.minTrianglesPerMetalPixel = min(metalCounts);
stats.maxTrianglesPerMetalPixel = max(metalCounts);
stats.table = table(rows, cols, metalCounts(:), ...
    'VariableNames', {'PixelRow', 'PixelCol', 'TriangleCount'});
end
