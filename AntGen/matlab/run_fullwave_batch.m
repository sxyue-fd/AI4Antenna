function run_fullwave_batch(input_file, output_file, num_workers)
%RUN_FULLWAVE_BATCH Simulate AntGen topology/feed candidates in parallel.
%
% Input fields are produced by AntGen/run.py. Both Python and MATLAB use the
% same physical (row, col) topology axes; only the feed index base changes
% from Python's zero-based convention to MATLAB's one-based convention.

arguments
    input_file (1,:) char
    output_file (1,:) char
    num_workers (1,1) double {mustBeInteger,mustBeNonnegative} = 0
end

assert(license('test', 'Antenna_Toolbox') == 1, ...
    'AntGen requires Antenna Toolbox for full-wave simulation.');
if num_workers ~= 1
    assert(license('test', 'Distrib_Computing_Toolbox') == 1, ...
        'Parallel simulation requires Parallel Computing Toolbox; use num_workers=1 for serial execution.');
end

payload = load(input_file);
assert(isfield(payload, 'spatial_coordinate_convention'), ...
    ['AntGen MATLAB input predates the physical row/col coordinate fix. ' ...
     'Rerun the infer stage before full-wave simulation.']);
coordinate_convention = strtrim(char(payload.spatial_coordinate_convention));
assert(strcmp(coordinate_convention, 'physical_row_col_v1'), ...
    'Unsupported spatial coordinate convention: %s', coordinate_convention);
required = {'structures', 'feed_rc', 'selected_candidate_indices', ...
    'dataset_indices', 'test_offsets', 'freq_hz', 'pattern_freq_hz', ...
    'pattern_theta_deg', 'geometry'};
for idx = 1:numel(required)
    assert(isfield(payload, required{idx}), ...
        'AntGen MATLAB input is missing field "%s".', required{idx});
end

structures = logical(payload.structures);
feed_rc = double(payload.feed_rc);
assert(ndims(structures) == 4, 'structures must be [m,n,H,W].');
assert(size(feed_rc,1) == size(structures,1) && ...
       size(feed_rc,2) == size(structures,2) && size(feed_rc,3) == 2, ...
       'feed_rc must be [m,n,2].');

params = build_design_parameters(payload);
num_conditions = size(structures, 1);
num_candidates = size(structures, 2);
num_jobs = num_conditions * num_candidates;
num_s11 = numel(params.freq_sweep);
num_pattern_freqs = numel(params.pattern_freqs);
num_theta = numel(params.pattern_theta_deg);

if num_workers ~= 1
    pool = gcp('nocreate');
    if isempty(pool)
        if num_workers > 1
            parpool(num_workers);
        else
            parpool;
        end
    elseif num_workers > 1 && pool.NumWorkers ~= num_workers
        delete(pool);
        parpool(num_workers);
    end
    try
        pctRunOnAll maxNumCompThreads(1)
    catch ME
        warning('AntGen:ThreadLimit', 'Could not limit worker threads: %s', ME.message);
    end
end

s11_cells = cell(num_jobs, 1);
pattern_cells = cell(num_jobs, 1);
success_cells = cell(num_jobs, 1);
error_cells = cell(num_jobs, 1);

fprintf('[AntGen] Simulating %d conditions x %d candidates = %d jobs.\n', ...
    num_conditions, num_candidates, num_jobs);
timer = tic;
if num_workers == 1
    for job = 1:num_jobs
        [condition_index, candidate_index] = ind2sub([num_conditions, num_candidates], job);
        metal = squeeze(structures(condition_index, candidate_index, :, :));
        feed = squeeze(feed_rc(condition_index, candidate_index, :)).';
        [s11_cells{job}, pattern_cells{job}, success_cells{job}, error_cells{job}] = ...
            simulate_one(metal, feed, params);
    end
else
    parfor job = 1:num_jobs
        [condition_index, candidate_index] = ind2sub([num_conditions, num_candidates], job);
        metal = squeeze(structures(condition_index, candidate_index, :, :));
        feed = squeeze(feed_rc(condition_index, candidate_index, :)).';
        [s11_cells{job}, pattern_cells{job}, success_cells{job}, error_cells{job}] = ...
            simulate_one(metal, feed, params);
    end
end
elapsed_seconds = toc(timer);

s11_sim = nan(num_conditions, num_candidates, num_s11, 'single');
pattern_sim = nan(num_conditions, num_candidates, num_pattern_freqs, 4, num_theta, 'single');
success = false(num_conditions, num_candidates);
error_messages = cell(num_conditions, num_candidates);
for job = 1:num_jobs
    [condition_index, candidate_index] = ind2sub([num_conditions, num_candidates], job);
    success(condition_index, candidate_index) = success_cells{job};
    error_messages{condition_index, candidate_index} = error_cells{job};
    if success_cells{job}
        s11_sim(condition_index, candidate_index, :) = s11_cells{job};
        pattern_sim(condition_index, candidate_index, :, :, :) = pattern_cells{job};
    end
end

freq_hz = params.freq_sweep; %#ok<NASGU>
pattern_freq_hz = params.pattern_freqs; %#ok<NASGU>
pattern_theta_deg = params.pattern_theta_deg; %#ok<NASGU>
dataset_indices = payload.dataset_indices; %#ok<NASGU>
test_offsets = payload.test_offsets; %#ok<NASGU>
selected_candidate_indices = payload.selected_candidate_indices; %#ok<NASGU>
spatial_coordinate_convention = coordinate_convention; %#ok<NASGU>
pattern_channel_order = 'XOZ_Gtheta,XOZ_Gphi,YOZ_Gtheta,YOZ_Gphi'; %#ok<NASGU>
pattern_value_type = 'linear_gain'; %#ok<NASGU>
matlab_version = version; %#ok<NASGU>
matlab_release = version('-release'); %#ok<NASGU>
output_parent = fileparts(output_file);
if ~isempty(output_parent) && ~exist(output_parent, 'dir')
    mkdir(output_parent);
end
save(output_file, 's11_sim', 'pattern_sim', 'success', 'error_messages', ...
    'freq_hz', 'pattern_freq_hz', 'pattern_theta_deg', 'dataset_indices', ...
    'test_offsets', 'selected_candidate_indices', 'spatial_coordinate_convention', ...
    'pattern_channel_order', 'pattern_value_type', 'matlab_version', ...
    'matlab_release', 'elapsed_seconds', '-v7');
fprintf('[AntGen] Finished in %.2f s: %d/%d simulations succeeded.\n', ...
    elapsed_seconds, nnz(success), num_jobs);
fprintf('[AntGen] Saved %s\n', output_file);
end


function params = build_design_parameters(payload)
geom = payload.geometry;
params.L = double(geom.patch_L_mm) / 1e3;
params.W = double(geom.patch_W_mm) / 1e3;
params.h = double(geom.sub_thick_mm) / 1e3;
params.substrate = dielectric(char(geom.substrate_name));
params.substrate.Thickness = params.h;
params.ground = antenna.Rectangle( ...
    'Length', double(geom.board_L_mm) / 1e3, ...
    'Width', double(geom.board_W_mm) / 1e3, ...
    'Center', [0, 0]);
params.feed_diameter = double(geom.feed_diam_mm) / 1e3;
params.overlap = double(geom.overlap_m);
params.freq_sweep = double(payload.freq_hz(:)).';
params.pattern_freqs = double(payload.pattern_freq_hz(:)).';
params.pattern_theta_deg = double(payload.pattern_theta_deg(:)).';
params.pattern_phi_xoz = 0;
params.pattern_phi_yoz = 90;
params.pixel_resolution = size(payload.structures, 3);
assert(size(payload.structures, 3) == size(payload.structures, 4), ...
    'Only square pixel topologies are supported.');

N = params.pixel_resolution;
params.patch_origin = [-params.L/2, -params.W/2];
params.pixel_L = params.L / N;
params.pixel_W = params.W / N;
params.pixel_shapes = cell(N, N);
for row = 1:N
    for col = 1:N
        center_x = params.patch_origin(1) + (col - 0.5) * params.pixel_L;
        center_y = params.patch_origin(2) + (row - 0.5) * params.pixel_W;
        params.pixel_shapes{row, col} = antenna.Rectangle( ...
            'Length', params.pixel_L + params.overlap, ...
            'Width', params.pixel_W + params.overlap, ...
            'Center', [center_x, center_y]);
    end
end
end


function [s11_data, pattern_data, ok, error_message] = simulate_one(metal, feed_rc, params)
num_pattern_freqs = numel(params.pattern_freqs);
num_theta = numel(params.pattern_theta_deg);
s11_data = nan(1, numel(params.freq_sweep), 'single');
pattern_data = nan(num_pattern_freqs, 4, num_theta, 'single');
ok = false;
error_message = '';
try
    metal = logical(metal);
    feed_rc = round(double(feed_rc));
    N = params.pixel_resolution;
    assert(isequal(size(metal), [N, N]), 'Topology has an unexpected shape.');
    assert(all(feed_rc >= 1) && feed_rc(1) <= N && feed_rc(2) <= N, ...
        'Feed coordinate is outside the topology.');
    metal(feed_rc(1), feed_rc(2)) = true;
    pixel_indices = find(metal);
    assert(~isempty(pixel_indices), 'Topology contains no metal pixels.');

    patch_shape = union_rectangles(params.pixel_shapes(pixel_indices));
    antenna_model = create_antenna_model(patch_shape, feed_rc, params);
    s_params = sparameters(antenna_model, params.freq_sweep);
    s11_data = single(abs(squeeze(s_params.Parameters(1, 1, :)))).';
    pattern_data = compute_pattern(antenna_model, params);
    ok = true;
catch ME
    error_message = getReport(ME, 'extended', 'hyperlinks', 'off');
end
end


function antenna_model = create_antenna_model(patch_shape, feed_rc, params)
start_x = -params.L/2 + params.pixel_L/2;
start_y = -params.W/2 + params.pixel_W/2;
feed_x = start_x + (feed_rc(2) - 1) * params.pixel_L;
feed_y = start_y + (feed_rc(1) - 1) * params.pixel_W;
antenna_model = pcbStack( ...
    'BoardShape', params.ground, ...
    'BoardThickness', params.h, ...
    'Layers', {patch_shape, params.substrate, params.ground}, ...
    'FeedDiameter', params.feed_diameter, ...
    'FeedLocations', [feed_x, feed_y, 1, 3]);
end


function pattern_data = compute_pattern(antenna_model, params)
num_freqs = numel(params.pattern_freqs);
num_theta = numel(params.pattern_theta_deg);
pattern_data = zeros(num_freqs, 4, num_theta, 'single');
for frequency_index = 1:num_freqs
    frequency = params.pattern_freqs(frequency_index);
    gain = pattern(antenna_model, frequency, params.pattern_phi_xoz, ...
        params.pattern_theta_deg, 'Type', 'gain', 'Polarization', 'V');
    pattern_data(frequency_index, 1, :) = single(10.^(gain(:) / 10));
    gain = pattern(antenna_model, frequency, params.pattern_phi_xoz, ...
        params.pattern_theta_deg, 'Type', 'gain', 'Polarization', 'H');
    pattern_data(frequency_index, 2, :) = single(10.^(gain(:) / 10));
    gain = pattern(antenna_model, frequency, params.pattern_phi_yoz, ...
        params.pattern_theta_deg, 'Type', 'gain', 'Polarization', 'V');
    pattern_data(frequency_index, 3, :) = single(10.^(gain(:) / 10));
    gain = pattern(antenna_model, frequency, params.pattern_phi_yoz, ...
        params.pattern_theta_deg, 'Type', 'gain', 'Polarization', 'H');
    pattern_data(frequency_index, 4, :) = single(10.^(gain(:) / 10));
end
end


function shape = union_rectangles(rectangles)
ids = find(~cellfun('isempty', rectangles));
assert(~isempty(ids), 'No rectangles were supplied.');
shape = rectangles{ids(1)};
for idx = 2:numel(ids)
    shape = shape + rectangles{ids(idx)};
end
end
