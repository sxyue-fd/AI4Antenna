% eval_dataset_current.m
% Evaluate current data stored in data_generation/dataset_out/*.h5.
%
% The /current dataset is expected to be [B,Fc,Cc,fineN,fineN].
% This script randomly samples samples from B, then computes histograms and
% quantiles of abs(current) for every frequency, channel and pixel.

clear; clc; close all;

%% User configuration
h5path = '';                         % Empty: use newest HDF5 in dataset_out that has /current.
sample_count_target = 1000;
rng_seed = 106;

num_bins = 100;
hist_clip_percentile = 99.9;         % Used only to choose finite histogram edges.
edge_probe_sample_count = 64;        % Samples used to estimate histogram edges.

quantile_probs = [0 0.001 0.01 0.05 0.10 0.25 0.50 0.75 0.90 0.95 0.99 0.999 1];
channel_names = {'Jx_real_abs','Jx_imag_abs','Jy_real_abs','Jy_imag_abs'};

save_figures = true;

%% Locate dataset
script_dir = fileparts(mfilename('fullpath'));
if isempty(script_dir)
    script_dir = pwd;
end

if isempty(h5path)
    dataset_dir = fullfile(script_dir, 'dataset_out');
    h5path = pick_latest_h5_with_current(dataset_dir);
elseif ~isfile(h5path)
    h5path_local = fullfile(script_dir, h5path);
    if isfile(h5path_local)
        h5path = h5path_local;
    end
end

assert(isfile(h5path), 'HDF5 file not found: %s', h5path);
assert(dataset_exists(h5path, '/current'), 'Dataset /current not found in: %s', h5path);

fprintf('Reading current dataset: %s\n', h5path);

infoC = h5info(h5path, '/current');
csize = double(infoC.Dataspace.Size);
assert(numel(csize) == 5, '/current must be [B,Fc,Cc,fineN,fineN].');

B_all = csize(1);
Fc = csize(2);
Cc = csize(3);
fineN = csize(4);
fineN2 = csize(5);
assert(fineN == fineN2, '/current spatial grid must be square.');

if Cc ~= numel(channel_names)
    channel_names = arrayfun(@(k) sprintf('ch%d_abs', k), 1:Cc, 'UniformOutput', false);
end

if dataset_exists(h5path, '/current_freq_hz')
    current_freq_hz = double(h5read(h5path, '/current_freq_hz'));
    current_freq_hz = current_freq_hz(:);
else
    current_freq_hz = (1:Fc).';
end

sample_count = min(sample_count_target, B_all);
rng(rng_seed);
sample_idx = sort(randperm(B_all, sample_count));

fprintf('Dataset size: B=%d, Fc=%d, Cc=%d, fineN=%d\n', B_all, Fc, Cc, fineN);
fprintf('Random sample count: %d (seed=%d)\n', sample_count, rng_seed);

%% Choose histogram edges from a small probe
probe_idx = sample_idx(round(linspace(1, sample_count, min(edge_probe_sample_count, sample_count))));
probe_values = [];

for kf = 1:Fc
    vals_probe = read_current_frequency(h5path, probe_idx, kf, Cc, fineN);
    probe_values = [probe_values; single(vals_probe(:))]; %#ok<AGROW>
end

probe_values = double(probe_values(isfinite(probe_values)));
assert(~isempty(probe_values), 'No finite current values were found in the probe.');

edge_hi = percentile_vector(probe_values, hist_clip_percentile / 100);
edge_hi = max(edge_hi, max(probe_values) * eps + eps);
hist_edges = [linspace(0, edge_hi, num_bins), inf];
num_hist_bins = numel(hist_edges) - 1;
hist_bin_centers = 0.5 * (hist_edges(1:end-1) + hist_edges(2:end));
hist_bin_centers(end) = hist_edges(end-1);

fprintf('Histogram edges: %d bins, finite range [0, %.6g], final bin catches overflow.\n', ...
    num_hist_bins, edge_hi);

%% Main evaluation
hist_counts = zeros(num_hist_bins, Fc, Cc, fineN, fineN, 'uint32');
quantiles = zeros(numel(quantile_probs), Fc, Cc, fineN, fineN, 'single');

hist_counts_global = zeros(num_hist_bins, Fc, Cc, 'uint64');
quantiles_global = zeros(numel(quantile_probs), Fc, Cc, 'single');

tic_eval = tic;
for kf = 1:Fc
    fprintf('Processing current frequency %d / %d', kf, Fc);
    if numel(current_freq_hz) >= kf
        fprintf(' (%.6g GHz)', current_freq_hz(kf) / 1e9);
    end
    fprintf(' ...\n');

    vals = read_current_frequency(h5path, sample_idx, kf, Cc, fineN);
    vals = abs(vals);  % [sample, channel, x, y]

    sorted_vals = sort(double(vals), 1);
    qvals = percentile_sorted_dim1(sorted_vals, quantile_probs);
    quantiles(:, kf, :, :, :) = reshape(single(qvals), [numel(quantile_probs), 1, Cc, fineN, fineN]);

    bin_idx = discretize(double(vals), hist_edges);
    for ib = 1:num_hist_bins
        bin_counts = squeeze(sum(bin_idx == ib, 1));
        hist_counts(ib, kf, :, :, :) = reshape(uint32(bin_counts), [1, 1, Cc, fineN, fineN]);
    end

    for cc = 1:Cc
        v = double(vals(:, cc, :, :));
        v = v(isfinite(v));
        quantiles_global(:, kf, cc) = single(percentile_vector(v, quantile_probs));

        b = discretize(v, hist_edges);
        b = b(isfinite(b));
        hist_counts_global(:, kf, cc) = uint64(accumarray(b(:), 1, [num_hist_bins 1], @sum, 0));
    end
end

fprintf('Evaluation completed in %.1f seconds.\n', toc(tic_eval));

%% Save outputs
[~, h5name] = fileparts(h5path);
out_dir = fullfile(script_dir, 'eval_current_out', h5name);
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

result_mat = fullfile(out_dir, 'current_hist_quantiles.mat');
save(result_mat, ...
    'h5path', 'sample_idx', 'rng_seed', 'sample_count', ...
    'current_freq_hz', 'channel_names', ...
    'hist_edges', 'hist_bin_centers', 'hist_counts', ...
    'hist_counts_global', 'quantile_probs', 'quantiles', 'quantiles_global', ...
    '-v7.3');

summary_csv = fullfile(out_dir, 'current_global_quantiles.csv');
write_global_quantile_csv(summary_csv, current_freq_hz, channel_names, quantile_probs, quantiles_global);

fprintf('Saved MAT result: %s\n', result_mat);
fprintf('Saved CSV summary: %s\n', summary_csv);

if save_figures
    save_overview_figures(out_dir, h5name, current_freq_hz, channel_names, ...
        hist_edges, hist_counts_global, quantile_probs, quantiles);
end

%% Local functions
function h5path = pick_latest_h5_with_current(dataset_dir)
assert(isfolder(dataset_dir), 'dataset_out folder not found: %s', dataset_dir);

files = dir(fullfile(dataset_dir, '*.h5'));
assert(~isempty(files), 'No .h5 files found in: %s', dataset_dir);

[~, order] = sort([files.datenum], 'descend');
files = files(order);

for ii = 1:numel(files)
    candidate = fullfile(files(ii).folder, files(ii).name);
    if dataset_exists(candidate, '/current')
        h5path = candidate;
        return;
    end
end

error('No .h5 file with /current was found in: %s', dataset_dir);
end

function vals = read_current_frequency(h5path, sample_idx, kf, Cc, fineN)
sample_idx = sample_idx(:).';
sample_count = numel(sample_idx);
vals = zeros(sample_count, Cc, fineN, fineN, 'single');

[run_starts, run_lengths, out_starts] = contiguous_runs(sample_idx);
for ir = 1:numel(run_starts)
    b0 = run_starts(ir);
    nb = run_lengths(ir);
    out0 = out_starts(ir);

    block = h5read(h5path, '/current', ...
        double([b0 kf 1 1 1]), double([nb 1 Cc fineN fineN]));
    block = reshape(single(block), [nb, 1, Cc, fineN, fineN]);
    block = squeeze(block(:, 1, :, :, :));

    if nb == 1
        block = reshape(block, [1, Cc, fineN, fineN]);
    end

    vals(out0:(out0 + nb - 1), :, :, :) = block;
end
end

function [run_starts, run_lengths, out_starts] = contiguous_runs(idx)
idx = idx(:);
breaks = [true; diff(idx) ~= 1];
out_starts = find(breaks);
run_starts = idx(out_starts);
run_ends = [out_starts(2:end) - 1; numel(idx)];
run_lengths = run_ends - out_starts + 1;
end

function q = percentile_sorted_dim1(sorted_vals, probs)
probs = probs(:);
n = size(sorted_vals, 1);
tail_size = size(sorted_vals);
tail_size = tail_size(2:end);

flat = reshape(sorted_vals, n, []);
qflat = zeros(numel(probs), size(flat, 2));

for ip = 1:numel(probs)
    p = min(max(probs(ip), 0), 1);
    pos = 1 + (n - 1) * p;
    lo = floor(pos);
    hi = ceil(pos);
    w = pos - lo;
    if lo == hi
        qflat(ip, :) = flat(lo, :);
    else
        qflat(ip, :) = (1 - w) * flat(lo, :) + w * flat(hi, :);
    end
end

q = reshape(qflat, [numel(probs), tail_size]);
end

function q = percentile_vector(v, probs)
v = sort(v(:));
v = v(isfinite(v));
assert(~isempty(v), 'Cannot compute percentile of an empty vector.');

probs = probs(:);
n = numel(v);
q = zeros(numel(probs), 1);
for ip = 1:numel(probs)
    p = min(max(probs(ip), 0), 1);
    pos = 1 + (n - 1) * p;
    lo = floor(pos);
    hi = ceil(pos);
    w = pos - lo;
    if lo == hi
        q(ip) = v(lo);
    else
        q(ip) = (1 - w) * v(lo) + w * v(hi);
    end
end
end

function write_global_quantile_csv(csv_path, current_freq_hz, channel_names, quantile_probs, quantiles_global)
Fc = size(quantiles_global, 2);
Cc = size(quantiles_global, 3);
num_rows = Fc * Cc;

freq_idx = zeros(num_rows, 1);
freq_hz = zeros(num_rows, 1);
channel_idx = zeros(num_rows, 1);
channel_name = strings(num_rows, 1);

qcols = zeros(num_rows, numel(quantile_probs));
rr = 0;
for kf = 1:Fc
    for cc = 1:Cc
        rr = rr + 1;
        freq_idx(rr) = kf;
        if numel(current_freq_hz) >= kf
            freq_hz(rr) = current_freq_hz(kf);
        else
            freq_hz(rr) = kf;
        end
        channel_idx(rr) = cc;
        channel_name(rr) = string(channel_names{cc});
        qcols(rr, :) = double(quantiles_global(:, kf, cc)).';
    end
end

T = table(freq_idx, freq_hz, channel_idx, channel_name);
for iq = 1:numel(quantile_probs)
    pct_name = sprintf('q_%g', quantile_probs(iq));
    pct_name = matlab.lang.makeValidName(strrep(pct_name, '.', 'p'));
    T.(pct_name) = qcols(:, iq);
end

writetable(T, csv_path);
end

function save_overview_figures(out_dir, h5name, current_freq_hz, channel_names, hist_edges, hist_counts_global, quantile_probs, quantiles)
finite_edges = hist_edges(1:end-1);
num_bins = numel(finite_edges);
Cc = numel(channel_names);
Fc = size(hist_counts_global, 2);

fig1 = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 1200 700]);
tiledlayout(fig1, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
for cc = 1:Cc
    nexttile;
    counts = double(sum(hist_counts_global(:, :, cc), 2));
    stairs(finite_edges, counts, 'LineWidth', 1.3);
    set(gca, 'YScale', 'log');
    grid on;
    xlabel('|current component|');
    ylabel('count');
    title(channel_names{cc}, 'Interpreter', 'none');
    xlim([finite_edges(1), finite_edges(max(2, num_bins - 1))]);
end
sgtitle(sprintf('%s: global current histograms', h5name), 'Interpreter', 'none');
export_or_save(fig1, fullfile(out_dir, 'global_histograms.png'));
close(fig1);

[~, q99_idx] = min(abs(quantile_probs - 0.99));
mid_f = max(1, round(Fc / 2));

fig2 = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 1200 900]);
tiledlayout(fig2, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
for cc = 1:Cc
    nexttile;
    img = squeeze(quantiles(q99_idx, mid_f, cc, :, :));
    imagesc(img);
    axis image;
    colorbar;
    title(sprintf('%s, q99, f-index=%d', channel_names{cc}, mid_f), 'Interpreter', 'none');
    if numel(current_freq_hz) >= mid_f
        xlabel(sprintf('%.6g GHz', current_freq_hz(mid_f) / 1e9));
    end
end
sgtitle(sprintf('%s: per-pixel q99 current magnitude', h5name), 'Interpreter', 'none');
export_or_save(fig2, fullfile(out_dir, 'pixel_q99_mid_frequency.png'));
close(fig2);
end

function export_or_save(fig, out_path)
try
    exportgraphics(fig, out_path, 'Resolution', 160);
catch
    saveas(fig, out_path);
end
end

function tf = dataset_exists(h5path, dset_name)
try
    h5info(h5path, dset_name);
    tf = true;
catch
    tf = false;
end
end
