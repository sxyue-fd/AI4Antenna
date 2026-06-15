function Ant_data_gen_v1p0()
% Ant_data_gen_v1p0.m
% =========================================================================
% 描述：
%   像素化贴片天线数据集生成脚本（单一仿真版本，进一步精简）。
%
%   1) 只保留一套仿真参数 sim_params
%   2) 不手动指定网格，让软件在求解时自动生成网格
%   3) 进一步合并"评分"和"正式仿真"：优化阶段每个候选只仿真一次
%   4) 优化阶段直接缓存 {cost, X, Y}，入选后直接写入 HDF5，不再重复仿真
%   有辐射方向图
%   四连通
% 输出：
%   HDF5 文件，包含：
%     /X           [B,1,N,N]  0=air, 1=metal, 2=feed
%     /Y           [B,F]
%     /freq_hz     [F,1]
%     /feed_rc     [B,2]
%     /cost        [B,1]
%     /sample_src  [B,1]      0=random, 1=optimized
%     /raw_fill    [B,1]
%     /final_fill  [B,1]
% =========================================================================
clear; clc; close all;

%% 0) User Configuration
num_top_per_feed             = 300;%每个馈电点保留样本数
num_random_designs_target    = 36000;%随机产生的样本数

pixelResolution_N            = 16;%像素分辨率
randomFillFactorRange        = [0.30, 0.50];%形态学操作前的填充率范围
minFinalFillRate             = 0.10;%最小填充率限制
maxTriesPerRandomCase        = 200;

fineN                        = 32;  % 电流插值规则网格
overlap_mm                   = [];  % 定为5e-6保证合并操作不出错

geom.patch_L_mm              = 14;
geom.patch_W_mm              = 14;
geom.sub_thick_mm            = 2.5;
geom.substrate_name          = 'Air';
geom.board_L_mm              = 30;
geom.board_W_mm              = 30;
geom.feed_init_xy            = [geom.patch_L_mm/4, 0];
geom.feed_diam_mm            = min(geom.patch_L_mm/pixelResolution_N, ...
                                   geom.patch_W_mm/pixelResolution_N) / 10;

sim_params.fmin_GHz          = 8;
sim_params.fmax_GHz          = 12;
sim_params.numFreqPoints     = 41;
sim_params.pattern_fmin_GHz   = 8.0;
sim_params.pattern_fmax_GHz   = 12.0;
sim_params.pattern_step_GHz   = 1;

sim_params.pattern_theta_deg  = 1:3:360;   
sim_params.pattern_phi_xoz    = 0;         % XOZ 平面
sim_params.pattern_phi_yoz    = 90;        % YOZ 平面
num_workers_to_use           = 64;
restart_every_feeds          = 32;%每32个馈电点重启一次并行池（后未用）
restart_every_rand_batches   = 40;
random_batch_size            = 512;
%数据集文件位置
output_dir = fullfile(pwd, 'dataset_out');
if ~exist(output_dir, 'dir'), mkdir(output_dir); end
ts = string(datetime("now","Format","yyyyMMdd_HHmmss"));
dataset_filename = fullfile(output_dir, "antenna_dataset_" + ts + ".h5");
%优化阶段参数
cfg = struct();
cfg.init_pool_size        = 500;
cfg.num_iters             = 15;
cfg.children_per_iter     = 200;
cfg.keep_size             = 5000;

cfg.beta                  = 0.18;
cfg.beta_p2               = 0.05;
cfg.p2_biased_prob        = 0.40;%父代2偏置选择概率

cfg.pc                    = 1;%交叉概率

cfg.pm_boundary           = 0.15;%边缘变异概率
cfg.pm_global             = 0.08;%全局变异概率
cfg.pm_feed_local         = 0.08;%馈电点附近变异概率
cfg.block_mut_prob        = 0.30;%块变异概率

cfg.feed_protect_radius   = 2;%馈电点保护半径
cfg.min_area_keep         = 4;%最小保护面积
cfg.min_final_fill        = minFinalFillRate;
cfg.max_fill_check_tries  = 50;
cfg.min_hamming_keep      = 12;%最小差异度

cfg.roi_fmin_GHz          = 8;
cfg.roi_fmax_GHz          = 12;
%cfg.bw                    = 0.12;%饱和带宽（未用）
cfg.w_bw                  = 1.5;%带宽奖励权重
cfg.depth                 = 0.5;%深度奖励权重
cfg.depth_thr_db          = -20;%深度临界值
cfg.thr_db                = -10;%10dB带宽
cfg.w_multiband_count     = 8.0;%额外带宽数量惩罚
cfg.w_multiband_extraBW   = 20.0;%额外带宽惩罚
cfg.f0_GHz                = 10;%标准中心频率
cfg.df_GHz                = 0.5;
cfg.w_f0                  = 4.0;%中心频率偏移惩罚

rng(106);

%% 1) 环境检查
assert(license('test','Distrib_Computing_Toolbox')==1, ...
    '未检测到 Parallel Computing Toolbox 许可证。');
assert(license('test','Antenna_Toolbox')==1, ...
    '未检测到 Antenna Toolbox 许可证。');
assert(license('test','Optimization_Toolbox')==1, ...
    '未检测到 Optimization Toolbox 许可证。');

%geom.overlap_rule_fineN      = fineN;（未用）
overlap_mm = 5e-3;%改overlap

fprintf('正在计算统一仿真设计参数...\n');
designParams = design_antenna_parameters(sim_params, geom, pixelResolution_N, fineN,overlap_mm);
fprintf('参数计算完成。\n\n');

fprintf('正在启动全局并行池...\n');
if num_workers_to_use > 0
    pool = parpool(num_workers_to_use); %#ok<NASGU>
else
    pool = parpool(); %#ok<NASGU>
end
try
    pctRunOnAll maxNumCompThreads(1)
catch
    warning('设定 worker 内线程数失败（可忽略）');
end
cleanupObj = onCleanup(@() finalize_env()); %#ok<NASGU>
fprintf('并行池已启动。\n\n');

%% 2) 初始化 HDF5
N = pixelResolution_N;%像素分辨率
C = 1; H = N; W = N;%通道数，用一个通道，馈电点位置为2
F = sim_params.numFreqPoints;%仿真频点数
Fp = numel(designParams.pattern_freqs);%方向图频点数
P  = 4;%方向图通道（两平面两极化）
T  = numel(designParams.pattern_theta_deg);%角度点数
Cc = designParams.current_num_channels;%电流通道数
Fc = numel(designParams.current_freqs);%电流频点
assert(Fc == Fp, '当前代码要求 current 和 pattern 使用同一组频点。');
if exist(dataset_filename, 'file'), delete(dataset_filename); end
%结构
h5create(dataset_filename, '/X', [Inf C H W], 'Datatype','single', ...
         'ChunkSize',[256 C H W], 'Deflate',5);
%S11幅值，512为批次维度大小
h5create(dataset_filename, '/Y', [Inf F], 'Datatype','single', ...
         'ChunkSize',[512 F], 'Deflate',5);
%方向图：[样本, 频点, 通道, 角度]
h5create(dataset_filename, '/pattern', [Inf Fp P T], 'Datatype','single', ...
         'ChunkSize', [64 Fp P T], 'Deflate', 5);
%电流：[样本, 频点, 通道, x, y]
h5create(dataset_filename, '/current', [Inf Fc Cc fineN fineN], 'Datatype','single', ...
         'ChunkSize', [8 1 Cc fineN fineN], 'Deflate', 5);
%给 /pattern 和 /current 提供"频率坐标轴"
h5create(dataset_filename, '/pattern_freq_hz', [Fp 1], 'Datatype', 'double');
h5write(dataset_filename, '/pattern_freq_hz', designParams.pattern_freqs(:));
h5create(dataset_filename, '/current_freq_hz', [Fc 1], 'Datatype', 'double');
h5write(dataset_filename, '/current_freq_hz', designParams.current_freqs(:));
%提供角度坐标轴
h5create(dataset_filename, '/pattern_theta_deg', [T 1], 'Datatype', 'double');
h5write(dataset_filename, '/pattern_theta_deg', designParams.pattern_theta_deg(:));
%馈电点
h5create(dataset_filename, '/feed_rc', [Inf 2], 'Datatype','uint8', ...
         'ChunkSize',[8192 2], 'Deflate',1);
%优化阶段代价值
h5create(dataset_filename, '/cost', [Inf 1], 'Datatype','single', ...
         'ChunkSize',[8192 1], 'Deflate',1);
%样本来源
h5create(dataset_filename, '/sample_src', [Inf 1], 'Datatype','uint8', ...
         'ChunkSize',[8192 1], 'Deflate',1);
%原始填充率
h5create(dataset_filename, '/raw_fill', [Inf 1], 'Datatype','single', ...
         'ChunkSize',[8192 1], 'Deflate',1);
%形态学处理后填充率
h5create(dataset_filename, '/final_fill', [Inf 1], 'Datatype','single', ...
         'ChunkSize',[8192 1], 'Deflate',1);
%S11 的频率采样点
freq_vector = designParams.freq_sweep(:);
h5create(dataset_filename, '/freq_hz', size(freq_vector), 'Datatype', 'double');
h5write(dataset_filename, '/freq_hz', freq_vector);
%说明 /X 里数值编码的含义
h5writeatt(dataset_filename, '/X', 'value_mapping', ...
    '0:air,1:metal,2:feed');
%说明 /pattern 和 /current 的通道顺序
h5writeatt(dataset_filename, '/pattern', 'channel_order', ...
    '1:XOZ_Etheta,2:XOZ_Ephi,3:YOZ_Etheta,4:YOZ_Ephi');
h5writeatt(dataset_filename, '/current', 'channel_order', ...
    '1:Jx_real,2:Jx_imag,3:Jy_real,4:Jy_imag');
%说明 /current 的空间网格大小
h5writeatt(dataset_filename, '/current', 'grid_size', int32(fineN));
%说明非金属区域的填充值
h5writeatt(dataset_filename, '/current', 'outside_metal_value', single(0));

write_idx = 0;
tic_global = tic;

%% 3) 优化样本阶段：每个馈电点做 morphology-aware evolution
fprintf('=== (阶段1) morphology-aware evolution：单次仿真闭环 ===\n');

time_opt_total = 0;
time_rand_hf = 0;
optimized_valid_total = 0;

for r_feed = 1:N
    for c_feed = 1:N
        feed_rc_fixed = [r_feed, c_feed];

        t_one = tic;
        cfg_local = cfg;
        cx = (N+1)/2; cy = (N+1)/2;
        dist_to_center = sqrt((c_feed-cx)^2 + (r_feed-cy)^2);
        %馈电点在中心，增大带宽奖励，减小频率偏移惩罚，增加馈电点附近变异率
        if dist_to_center <= 0.32 * N
            cfg_local.w_bw = 2.0;
            % cfg_local.w_f0 = 2.5;
            cfg_local.pm_feed_local = 0.2;
        else
            cfg_local.w_bw = cfg.w_bw;
        end
        
        [top_pool] = sample_morphology_aware_designs_for_feed( ...
            designParams, feed_rc_fixed, num_top_per_feed, cfg_local, randomFillFactorRange);
        % [top_pool] = sample_morphology_aware_designs_for_feed( ...
        %     designParams, feed_rc_fixed, num_top_per_feed, cfg, randomFillFactorRange);
        time_opt_total = time_opt_total + toc(t_one);

        K_now = numel(top_pool);
fprintf('Feed (%2d,%2d): selected %d designs.\n', r_feed, c_feed, K_now);
fprintf('耗时： %.2fs\n', time_opt_total);
%对已获得的样本求解辐射方向图和电流
if K_now > 0
    t_two = tic;
    top_vecs = cell(K_now, 1);
    top_feed_rc = repmat(feed_rc_fixed, K_now, 1);

    for ii = 1:K_now
        top_vecs{ii} = top_pool(ii).vec;
    end

    top_patterns = cell(K_now, 1);
    top_currents = cell(K_now, 1);
    top_pc_ok    = false(K_now, 1);

    parfor ii = 1:K_now
        [top_patterns{ii}, top_currents{ii}, top_pc_ok(ii)] = ...
            compute_pattern_and_current_only_for_design(top_vecs{ii}, designParams, top_feed_rc(ii,:));
    end
    time_opt_total = time_opt_total + toc(t_two);
    keep_opt = find(top_pc_ok);
    K_keep = numel(keep_opt);
    fprintf('结束辐射方向图和电流，剩余样本 %d ，总耗时： %.2fs\n',  K_keep ,time_opt_total);
    if K_keep > 0
        X_batch = zeros(K_keep, 1, N, N, 'single');
        Y_batch = zeros(K_keep, F, 'single');
        feed_batch = zeros(K_keep, 2, 'uint8');
        cost_batch = zeros(K_keep, 1, 'single');
        src_batch  = ones(K_keep, 1, 'uint8');
        rawf_batch = zeros(K_keep, 1, 'single');
        finf_batch = zeros(K_keep, 1, 'single');
        pattern_batch = zeros(K_keep, Fp, 4, T, 'single');
        current_batch = zeros(K_keep, Fc, Cc, fineN, fineN, 'single');

        for jj = 1:K_keep
            ii = keep_opt(jj);

            Xc = permute(single(top_pool(ii).sim.X), [3 1 2]);
            X_batch(jj,:,:,:) = Xc;
            Y_batch(jj,:)     = single(top_pool(ii).sim.Y(:)).';
            pattern_batch(jj,:,:,:) = top_patterns{ii};
            current_batch(jj,:,:,:,:) = top_currents{ii};
            feed_batch(jj,:)  = uint8(feed_rc_fixed);
            cost_batch(jj)    = single(top_pool(ii).cost);
            rawf_batch(jj)    = single(top_pool(ii).raw_fill);
            finf_batch(jj)    = single(top_pool(ii).final_fill);
        end

        h5write(dataset_filename, '/X', X_batch, [write_idx+1 1 1 1], [K_keep 1 N N]);
        h5write(dataset_filename, '/Y', Y_batch, [write_idx+1 1],     [K_keep F]);
        h5write(dataset_filename, '/pattern', pattern_batch, ...
            [write_idx+1 1 1 1], [K_keep Fp 4 T]);
        h5write(dataset_filename, '/current', current_batch, ...
            [write_idx+1 1 1 1 1], [K_keep Fc Cc fineN fineN]);
        h5write(dataset_filename, '/feed_rc', feed_batch, [write_idx+1 1], [K_keep 2]);
        h5write(dataset_filename, '/cost', cost_batch, [write_idx+1 1], [K_keep 1]);
        h5write(dataset_filename, '/sample_src', src_batch, [write_idx+1 1], [K_keep 1]);
        h5write(dataset_filename, '/raw_fill', rawf_batch, [write_idx+1 1], [K_keep 1]);
        h5write(dataset_filename, '/final_fill', finf_batch, [write_idx+1 1], [K_keep 1]);

        write_idx = write_idx + K_keep;
        optimized_valid_total = optimized_valid_total + K_keep;
        fprintf('  written %d / %d, total written=%d\n', K_keep, K_now, write_idx);
    else
        fprintf('  written 0 / %d (all pattern/current failed)\n', K_now);
    end
else
    fprintf('  written 0\n');
end
        clear top_pool X_batch Y_batch pattern_batch current_batch top_patterns top_currents top_vecs top_feed_rc feed_batch cost_batch src_batch rawf_batch finf_batch;
        feed_id = (r_feed-1)*N + c_feed;
        if mod(feed_id, restart_every_feeds) == 0 && feed_id < N*N
            delete(gcp('nocreate'));
            if num_workers_to_use > 0
                pool = parpool(num_workers_to_use); %#ok<NASGU>
            else
                pool = parpool(); %#ok<NASGU>
            end
            try pctRunOnAll maxNumCompThreads(1); catch, end
        end
    end
end

fprintf('阶段1 单次仿真优化总耗时: %.2fs\n', time_opt_total);
fprintf('阶段1 valid 总数: %d\n\n', optimized_valid_total);

%% 4) 随机样本阶段：分批生成 -> 仿真 -> H5
fprintf('=== (阶段2) 随机样本：分批生成 -> 仿真 -> H5 ===\n');

num_rand_total = max(0, num_random_designs_target);
random_valid_total = 0;
random_reject_total = 0;

if num_rand_total == 0
    fprintf('随机样本目标为 0，跳过随机阶段。\n');
else
    num_batches = ceil(num_rand_total / random_batch_size);
    fprintf('随机样本总数=%d, batch_size=%d, batches=%d\n', ...
        num_rand_total, random_batch_size, num_batches);

    rng('shuffle');
    global_hash_set = containers.Map('KeyType', 'char', 'ValueType', 'logical');
    %逐批随机生成几何
    for b = 1:num_batches
        t_batch = tic;

        i1 = (b-1)*random_batch_size + 1;
        i2 = min(b*random_batch_size, num_rand_total);
        B  = i2 - i1 + 1;

        rand_designs = false(B, N^2);
        rand_feed_rc = zeros(B, 2, 'uint8');
        rand_raw_fill = zeros(B, 1, 'single');
        rand_final_fill = zeros(B, 1, 'single');
        valid_gen = true(B, 1);
        
        for k = 1:B
            accepted = false;
            tries = 0;
            pMetal = randomFillFactorRange(1);

            while ~accepted && tries < maxTriesPerRandomCase
                tries = tries + 1;

                pMetal = randomFillFactorRange(1) + ...
                    (randomFillFactorRange(2)-randomFillFactorRange(1))*rand();

                feed_r = randi([1,N]);
                feed_c = randi([1,N]);
                feed_rc =[feed_r, feed_c];

                raw = rand(N,N) < pMetal;
                raw(feed_r, feed_c) = true;

                raw_fill = nnz(raw) / numel(raw);
                pheno = morphology_repair(raw, feed_rc, cfg, N);
                final_fill = nnz(pheno) / numel(pheno);
                %做有效性和去重筛选
                if final_fill >= cfg.min_final_fill
                    hash_str = DataHash(uint8(pheno(:)));
                    if ~isKey(global_hash_set, hash_str)
                        global_hash_set(hash_str) = true;
                        accepted = true;
                        rand_designs(k,:) = pheno(:).';
                        rand_feed_rc(k,:) = uint8(feed_rc);
                        rand_raw_fill(k) = single(raw_fill);
                        rand_final_fill(k) = single(final_fill);
                    else
                        random_reject_total = random_reject_total + 1;
                    end
                else
                    random_reject_total = random_reject_total + 1;
                end
            end
            %最大尝试次数后依然失败,在馈点周围造一个小的局部金属块
            if ~accepted
                feed_r = randi([1,N]);
                feed_c = randi([1,N]);
                feed_rc = [feed_r, feed_c];

                pheno = false(N, N);
                r_min = max(1, feed_r-1); r_max = min(N, feed_r+1);
                c_min = max(1, feed_c-1); c_max = min(N, feed_c+1);
                pheno(r_min:r_max, c_min:c_max) = true;

                hash_str = DataHash(uint8(pheno(:)));
                if ~isKey(global_hash_set, hash_str)
                    global_hash_set(hash_str) = true;
                    rand_designs(k,:) = pheno(:).';
                    rand_feed_rc(k,:) = uint8(feed_rc);
                    rand_raw_fill(k) = single(pMetal);
                    rand_final_fill(k) = single(nnz(pheno)/numel(pheno));
                else
                    valid_gen(k) = false;
                end
            end
        end
        
        results = cell(1, B);
        t_hf = tic;
        %并行做电磁仿真S11
        parfor k = 1:B
            if valid_gen(k)
                results{k} = simulate_single_antenna_hf(rand_designs(k,:), ...
                    designParams, double(rand_feed_rc(k,:)));
            else
                results{k} = struct('isValid', false);
            end
        end
        time_rand_hf = time_rand_hf + toc(t_hf);

        valid = ~cellfun(@(res) isempty(res) || ~res.isValid, results);
        nv = sum(valid);%完成高频仿真的样本数

     if nv > 0
        vv = find(valid);
    
        valid_designs = cell(nv, 1);
        valid_feeds = zeros(nv, 2);
        for ii = 1:nv
            valid_designs{ii} = rand_designs(vv(ii),:);
            valid_feeds(ii,:) = double(rand_feed_rc(vv(ii),:));
        end
    
        valid_patterns = cell(nv, 1);
        valid_currents = cell(nv, 1);
        valid_pc_ok    = false(nv, 1);
    %并行补算方向图和电流
        parfor ii = 1:nv
            [valid_patterns{ii}, valid_currents{ii}, valid_pc_ok(ii)] = ...
                compute_pattern_and_current_only_for_design(valid_designs{ii}, designParams, valid_feeds(ii,:));
        end
    
        nv2 = sum(valid_pc_ok);%补算成功样本数
    %写入HDF5
        if nv2 > 0
            X_batch = zeros(nv2, 1, N, N, 'single');
            Y_batch = zeros(nv2, F, 'single');
            pattern_batch = zeros(nv2, Fp, 4, T, 'single');
            current_batch = zeros(nv2, Fc, Cc, fineN, fineN, 'single');
            feed_batch = zeros(nv2, 2, 'uint8');
            cost_batch = single(nan(nv2,1));
            src_batch  = zeros(nv2,1,'uint8');
            rawf_batch = zeros(nv2,1,'single');
            finf_batch = zeros(nv2,1,'single');
    
            jj = 0;
            for ii = 1:nv
                if ~valid_pc_ok(ii), continue; end
                jj = jj + 1;
    
                res = results{vv(ii)};
                Xc = permute(single(res.X), [3 1 2]);
    
                X_batch(jj,:,:,:) = Xc;
                Y_batch(jj,:) = single(res.Y(:)).';
                pattern_batch(jj,:,:,:) = valid_patterns{ii};
                current_batch(jj,:,:,:,:) = valid_currents{ii};
                feed_batch(jj,:) = uint8(rand_feed_rc(vv(ii),:));
                rawf_batch(jj) = rand_raw_fill(vv(ii));
                finf_batch(jj) = rand_final_fill(vv(ii));
            end
    
            h5write(dataset_filename, '/X', X_batch, [write_idx+1 1 1 1], [nv2 1 N N]);
            h5write(dataset_filename, '/Y', Y_batch, [write_idx+1 1], [nv2 F]);
            h5write(dataset_filename, '/pattern', pattern_batch, ...
                [write_idx+1 1 1 1], [nv2 Fp 4 T]);
            h5write(dataset_filename, '/current', current_batch, ...
                [write_idx+1 1 1 1 1], [nv2 Fc Cc fineN fineN]);
            h5write(dataset_filename, '/feed_rc', feed_batch, [write_idx+1 1], [nv2 2]);
            h5write(dataset_filename, '/cost', cost_batch, [write_idx+1 1], [nv2 1]);
            h5write(dataset_filename, '/sample_src', src_batch, [write_idx+1 1], [nv2 1]);
            h5write(dataset_filename, '/raw_fill', rawf_batch, [write_idx+1 1], [nv2 1]);
            h5write(dataset_filename, '/final_fill', finf_batch, [write_idx+1 1], [nv2 1]);
    
            write_idx = write_idx + nv2;
            random_valid_total = random_valid_total + nv2;
        end
    end

        clear results X_batch Y_batch pattern_batch current_batch valid_patterns valid_currents valid_designs valid_feeds feed_batch cost_batch src_batch rawf_batch finf_batch;
        clear rand_designs rand_feed_rc rand_raw_fill rand_final_fill valid_gen;

        fprintf('Batch %d/%d: generated=%d, valid=%d, total_written=%d, elapsed=%.1fs\n', ...
            b, num_batches, B, nv, write_idx, toc(t_batch));

        if restart_every_rand_batches > 0 && mod(b, restart_every_rand_batches) == 0 && b < num_batches
            fprintf('Restarting parpool at random batch=%d ...\n', b);
            delete(gcp('nocreate'));
            if num_workers_to_use > 0
                pool = parpool(num_workers_to_use); %#ok<NASGU>
            else
                pool = parpool(); %#ok<NASGU>
            end
            try pctRunOnAll maxNumCompThreads(1); catch, end
        end
    end
end

fprintf('随机阶段总耗时: %.2f 秒\n', time_rand_hf);
fprintf('随机阶段 valid 总数: %d\n', random_valid_total);
fprintf('随机阶段 reject 总数: %d\n\n', random_reject_total);

%% 5) 写 meta_json
fprintf('=== (阶段3) 写入 meta_json ===\n');

run_info = struct();
%记录本次运行时间和输出文件名。
run_info.timestamp    = char(datetime("now"));
run_info.dataset_file = dataset_filename;
%目标采样规模
run_info.targets = struct( ...
    'num_top_per_feed', num_top_per_feed, ...
    'num_optimized_designs_target', N*N*num_top_per_feed, ...
    'num_random_designs_target', num_random_designs_target);
%随机阶段的关键参数
run_info.random = struct( ...
    'randomFillFactorRange', randomFillFactorRange, ...
    'minFinalFillRate', minFinalFillRate, ...
    'maxTriesPerRandomCase', maxTriesPerRandomCase, ...
    'random_batch_size', random_batch_size);
%并行计算相关配置
run_info.parallel = struct( ...
    'num_workers_to_use', num_workers_to_use, ...
    'restart_every_rand_batches', restart_every_rand_batches);
%仿真和几何参数
run_info.params = struct( ...
    'sim_params', sim_params, ...
    'geom', geom, ...
    'N', pixelResolution_N, ...
    'fineN', fineN, ...
    'overlap_m', designParams.overlap);
%记录方向图和电流数据的语义说明
run_info.pattern = struct( ...
    'freq_hz', designParams.pattern_freqs, ...
    'theta_deg', designParams.pattern_theta_deg, ...
    'channels', {{'XOZ_Etheta','XOZ_Ephi','YOZ_Etheta','YOZ_Ephi'}}, ...
    'stored_value', 'magnitude');

run_info.current = struct( ...
    'freq_hz', designParams.current_freqs, ...
    'fineN', fineN, ...
    'channels', {{'Jx_real','Jx_imag','Jy_real','Jy_imag'}}, ...
    'stored_value', 'interpolated_surface_current_components', ...
    'outside_metal_value', 0, ...
    'interpolation', 'natural', ...
    'extrapolation', 'none');

run_info.morphology_aware_cfg = cfg;
%耗时
perf = struct();
perf.time_opt_total = time_opt_total;
perf.time_rand_hf   = time_rand_hf;
perf.time_total     = toc(tic_global);
run_info.perf = perf;
%记录最终写入的数据量
counts = struct();
counts.num_valid_samples      = write_idx;
counts.num_optimized_valid    = optimized_valid_total;
counts.num_random_valid       = random_valid_total;
counts.num_random_reject      = random_reject_total;
counts.num_feeds              = N*N;
run_info.counts = counts;
%当前 MATLAB 版本信息
try
    run_info.matlab = struct('version', version, 'release', version('-release'));
catch
end

meta_json = jsonencode(make_json_serializable(run_info));
h5writeatt(dataset_filename, '/', 'meta_json', meta_json);
h5writeatt(dataset_filename, '/', 'num_samples', int64(write_idx));

fprintf('数据集已保存到: %s\n', dataset_filename);

%% 6) 总结
fprintf('\n=== 性能总结 ===\n');
fprintf('阶段1 单次仿真优化耗时: %.2f 秒\n', time_opt_total);
fprintf('阶段2 RAND:            %.2f 秒\n', time_rand_hf);
fprintf('最终有效样本:         %d\n', write_idx);
fprintf('总程序耗时:           %.2f 秒\n', toc(tic_global));

end

%% =========================================================================
%%                              核心优化模块
%% =========================================================================
function [top_pool] = sample_morphology_aware_designs_for_feed( ...
    designParams, feed_rc_fixed, K, cfg, randomFillFactorRange)

N = designParams.pixelResolution_N;
pool = repmat(make_empty_individual(N), cfg.init_pool_size, 1);

% ---------- 1) 初始化种群：满足 fill 后直接做单次仿真 ----------
parfor i = 1:cfg.init_pool_size
    accepted = false;
    tries = 0;
    tmp_ind = make_empty_individual(N);

    while ~accepted && tries < cfg.max_fill_check_tries
        tries = tries + 1;

        pMetal = randomFillFactorRange(1) + ...
            (randomFillFactorRange(2) - randomFillFactorRange(1)) * rand();

        raw = rand(N, N) < pMetal;
        raw(feed_rc_fixed(1), feed_rc_fixed(2)) = true;

        pheno = morphology_repair(raw, feed_rc_fixed, cfg, N);%形态学操作
        raw_fill   = nnz(raw) / numel(raw);
        final_fill = nnz(pheno) / numel(pheno);

        if final_fill >= cfg.min_final_fill
            sim = simulate_and_score_single_antenna(pheno(:).', designParams, feed_rc_fixed, cfg);
            if sim.isValid
                accepted = true;
                tmp_ind.raw          = raw;
                tmp_ind.pheno        = pheno;
                tmp_ind.vec          = pheno(:).';
                tmp_ind.raw_fill     = raw_fill;
                tmp_ind.final_fill   = final_fill;
                tmp_ind.cost         = sim.cost;
                tmp_ind.cost_details = sim.cost_details;
                tmp_ind.sim          = sim;
            end
        end
    end

    if ~accepted%最大尝试次数后填充率依然不满足
        pMetal = randomFillFactorRange(1) + ...
            (randomFillFactorRange(2) - randomFillFactorRange(1)) * rand();

        pheno = generateSmoothDesignStable4Conn(N, feed_rc_fixed, pMetal);
        pheno(feed_rc_fixed(1), feed_rc_fixed(2)) = true;
        raw = pheno;
        raw_fill   = nnz(raw) / numel(raw);
        final_fill = nnz(pheno) / numel(pheno);

        tmp_ind.raw        = raw;
        tmp_ind.pheno      = pheno;
        tmp_ind.vec        = pheno(:).';
        tmp_ind.raw_fill   = raw_fill;
        tmp_ind.final_fill = final_fill;

        if final_fill >= cfg.min_final_fill
            sim = simulate_and_score_single_antenna(pheno(:).', designParams, feed_rc_fixed, cfg);
            if sim.isValid
                tmp_ind.cost         = sim.cost;
                tmp_ind.cost_details = sim.cost_details;
                tmp_ind.sim          = sim;
            end
        end
    end

    pool(i) = tmp_ind;
end

pool = pool([pool.final_fill] >= cfg.min_final_fill & isfinite([pool.cost]));
pool = dedup_pool_by_pheno(pool, cfg);

if isempty(pool)
    top_pool = repmat(make_empty_individual(N), 0, 1);
    return;
end

% ---------- 2) 进化迭代：子代满足 fill 后直接做单次仿真 ----------
for it = 1:cfg.num_iters
    costs = [pool.cost].';
    num_pool = numel(pool);

    parent1_idx = biased_select_indices(costs, cfg.children_per_iter, cfg.beta);
    parent2_idx = zeros(cfg.children_per_iter, 1);
    for t = 1:cfg.children_per_iter
        if rand < cfg.p2_biased_prob
            p2_idx = biased_select_indices(costs, 1, cfg.beta_p2);
        else
            p2_idx = randi(num_pool);
        end
        retry = 0;
        while p2_idx == parent1_idx(t) && retry < 5
            p2_idx = randi(num_pool);
            retry = retry + 1;
        end
        parent2_idx(t) = p2_idx;
    end

    children = repmat(make_empty_individual(N), cfg.children_per_iter, 1);

    parfor t = 1:cfg.children_per_iter
        p1 = pool(parent1_idx(t));
        p2 = pool(parent2_idx(t));

        child_raw = p1.raw;%以父1为模板进行更改
        if rand < cfg.pc
            child_raw = morphology_aware_crossover(p1.raw, p2.raw, feed_rc_fixed, cfg);%交叉
        end
        child_raw = morphology_aware_mutation(child_raw, feed_rc_fixed, cfg);%变异

        child_raw(feed_rc_fixed(1), feed_rc_fixed(2)) = true;
        child_pheno = morphology_repair(child_raw, feed_rc_fixed, cfg, N);

        child = make_empty_individual(N);
        child.raw        = child_raw;
        child.pheno      = child_pheno;
        child.vec        = child_pheno(:).';
        child.raw_fill   = nnz(child_raw) / numel(child_raw);
        child.final_fill = nnz(child_pheno) / numel(child_pheno);

        if child.final_fill >= cfg.min_final_fill
            sim = simulate_and_score_single_antenna(child.vec, designParams, feed_rc_fixed, cfg);
            if sim.isValid
                child.cost         = sim.cost;
                child.cost_details = sim.cost_details;
                child.sim          = sim;
                children(t) = child;
            end
        end
    end

    children = children([children.final_fill] >= cfg.min_final_fill & isfinite([children.cost]));
    pool = [pool; children]; %#ok<AGROW>
    pool = dedup_pool_by_pheno(pool, cfg);
    pool = keep_archive_diverse_by_pheno(pool, cfg.keep_size, cfg.min_hamming_keep);

    if isempty(pool)
        top_pool = repmat(make_empty_individual(N), 0, 1);
        return;
    end

    if mod(it, 5) == 0 || it == cfg.num_iters
        costs_now = [pool.cost];
        fprintf('  iter %2d: best cost = %.4f, archive = %d\n', ...
            it, min(costs_now), numel(pool));
    end
end

% ---------- 3) 输出最优 Top-K ----------
[~, ord] = sort([pool.cost], 'ascend');
ord = ord(1:min(K, numel(ord)));
top_pool = pool(ord);

fprintf('\n=== Top 16 Optimized Designs Cost Details ===\n');
fprintf('%-5s | %-8s || %-8s | %-8s | %-8s | %-8s | %-10s | %-8s\n', ...
        'Rank', 'Total', 'Depth', 'Bands', 'ExtraBW', 'Ripple', 'BW_Reward', 'f0_Pen');
fprintf('------------------------------------------------------------------------------------\n');
num_to_print = min(16, numel(ord));
for i = 1:num_to_print
    c_tot = top_pool(i).cost;
    c_det = top_pool(i).cost_details;
    fprintf(' #%-3d | %8.3f || %8.3f | %8.3f | %8.3f | %8.3f | %10.3f | %8.3f\n', ...
        i, c_tot, c_det.depth, c_det.bands, c_det.extraBW, c_det.ripple, c_det.bw_reward, c_det.f0);
end
fprintf('====================================================================================\n\n');
end

function ind = make_empty_individual(N)
ind = struct( ...
    'raw',        false(N,N), ...
    'pheno',      false(N,N), ...
    'vec',        false(1,N*N), ...
    'raw_fill',   0, ...
    'final_fill', 0, ...
    'cost',       inf, ...
    'cost_details', struct(), ...
    'sim',        struct('isValid', false));
end

function pheno = morphology_repair(raw, feed_rc, cfg, pixelN)
bw = logical(raw);
[X,Y] = meshgrid(1:pixelN, 1:pixelN);
cx = (pixelN+1)/2;
cy = (pixelN+1)/2;

% 只有馈点不在中心附近时，才做中心金属增强
dist_to_center = sqrt((feed_rc(2)-cx)^2 + (feed_rc(1)-cy)^2);
if dist_to_center > 0.32 * pixelN
    R2 = ((X-cx)/(0.42*pixelN)).^2 + ((Y-cy)/(0.42*pixelN)).^2;
    maskCenter = R2 <= 1.0;
    centerBoost = 0.02 + 0.04 * rand();
    tmp = rand(pixelN, pixelN) < centerBoost;
    bw(maskCenter) = bw(maskCenter) | tmp(maskCenter);
end
r = feed_rc(1); c = feed_rc(2);
bw(r,c) = true;
bw = imclose(bw, strel('square', 2));
bw = imopen(bw,  strel('square', 2));
bw = bwareaopen(bw, cfg.min_area_keep, 4);
bw = bwmorph(bw, 'spur', 1);
bw(r,c) = true;
bw = keepFeedConnectedComponent4(bw, feed_rc);
bw(r,c) = true;
pheno = logical(bw);
end

function child = morphology_aware_crossover(p1, p2, feed_rc, cfg)
[N, ~] = size(p1);
child = p1;
mode = randi(3);
switch mode%三种交叉方式
    case 1%生成方块区域
        h = randi([3, max(3, round(N/2))]);
        w = randi([3, max(3, round(N/2))]);
        r0 = randi([1, N-h+1]);
        c0 = randi([1, N-w+1]);
        mask = false(N,N);
        mask(r0:r0+h-1, c0:c0+w-1) = true;
    case 2%椭圆形区域
        [X,Y] = meshgrid(1:N, 1:N);
        cx = randi([1,N]);
        cy = randi([1,N]);
        a = randi([2, max(2, round(N/3))]);
        b = randi([2, max(2, round(N/3))]);
        mask = (((X-cx)/a).^2 + ((Y-cy)/b).^2) <= 1.0;
    otherwise%条带（横竖各占0.5）
        if rand < 0.5
            cut = randi([2, N-1]);
            mask = false(N,N);
            mask(:, cut:end) = true;
        else
            cut = randi([2, N-1]);
            mask = false(N,N);
            mask(cut:end, :) = true;
        end
end
pr = get_feed_protect_radius(feed_rc, N);%馈电点保护半径
protect = feed_protection_mask(N, feed_rc, pr);%馈电点附近保护（不交叉）
%protect = feed_protection_mask(N, feed_rc, cfg.feed_protect_radius);
mask(protect) = false;
child(mask) = p2(mask);%将生成区域变为p2
child(feed_rc(1), feed_rc(2)) = true;
end
function radius = get_feed_protect_radius(feed_rc, N)
cx = (N+1)/2;
cy = (N+1)/2;
dist_to_center = sqrt((feed_rc(2)-cx)^2 + (feed_rc(1)-cy)^2);

if dist_to_center <= 0.20 * N
    radius = 1;   % 中心馈电，给更多局部调整自由度
else
    radius = 2;   % 边缘馈电，保持原值
end
end
function child = morphology_aware_mutation(raw, feed_rc, cfg)
[N, ~] = size(raw);
child = raw;

boundary = bwperim(child, 4);%四连通边界
neigh = imdilate(boundary, strel('square', 3));%边界膨胀
flip_boundary = rand(N,N) < cfg.pm_boundary;
child(neigh & flip_boundary) = ~child(neigh & flip_boundary);%边界变异

flip_global = rand(N,N) < cfg.pm_global;
child(flip_global) = ~child(flip_global);%全局变异

if rand < cfg.block_mut_prob%块状变异
    bh = randi([2,4]);
    bw = randi([2,4]);
    r0 = randi([1, N-bh+1]);
    c0 = randi([1, N-bw+1]);
    if rand < 0.5
        child(r0:r0+bh-1, c0:c0+bw-1) = true;%全为金属
    else
        child(r0:r0+bh-1, c0:c0+bw-1) = false;%挖空
    end
end
pr = get_feed_protect_radius(feed_rc, N);
local_mask = feed_protection_mask(N, feed_rc, pr);
%local_mask = feed_protection_mask(N, feed_rc, cfg.feed_protect_radius);
flip_local = rand(N,N) < cfg.pm_feed_local;
child(local_mask & flip_local) = ~child(local_mask & flip_local);
child(feed_rc(1), feed_rc(2)) = true;
% 中心馈电时，增加馈点附近减铜操作，抬高输入阻抗
cx0 = (N+1)/2;
cy0 = (N+1)/2;
dist_to_center = sqrt((feed_rc(2)-cx0)^2 + (feed_rc(1)-cy0)^2);

if dist_to_center <= 0.32 * N && rand < 0.25
    rr = feed_rc(1); cc = feed_rc(2);
    r1 = max(1, rr-1); r2 = min(N, rr+1);
    c1 = max(1, cc-1); c2 = min(N, cc+1);

    holeMask = rand(r2-r1+1, c2-c1+1) < 0.20;
    child(r1:r2, c1:c2) = child(r1:r2, c1:c2) & ~holeMask;
    child(rr,cc) = true;
end
end

function mask = feed_protection_mask(N, feed_rc, radius)
[X, Y] = meshgrid(1:N, 1:N);
rr = feed_rc(1); cc = feed_rc(2);
mask = (X-cc).^2 + (Y-rr).^2 <= radius^2;
end

function pool_u = dedup_pool_by_pheno(pool, cfg)
if isempty(pool)
    pool_u = pool;
    return;
end

keys = cell(numel(pool),1);
for i = 1:numel(pool)
    keys{i} = DataHash(uint8(pool(i).pheno(:)));
end

[ukeys, ~, ic] = unique(keys, 'stable');
pool_u = repmat(pool(1), numel(ukeys), 1);
for k = 1:numel(ukeys)
    idx = find(ic == k);
    [~, j] = min([pool(idx).cost]);
    pool_u(k) = pool(idx(j));
end
keep = [pool_u.final_fill] >= cfg.min_final_fill & isfinite([pool_u.cost]);
pool_u = pool_u(keep);
end

function pool_k = keep_archive_diverse_by_pheno(pool, keep_size, dmin)
if isempty(pool)
    pool_k = pool;
    return;
end

[~, ord] = sort([pool.cost], 'ascend');
pool_s = pool(ord);
selected = false(size(pool_s));
chosen_idx = [];

for i = 1:numel(pool_s)
    if numel(chosen_idx) >= keep_size
        break;
    end

    if isempty(chosen_idx)
        selected(i) = true;
        chosen_idx = i;
    else
        v = pool_s(i).vec;
        minDist = inf;
        for j = chosen_idx
            d = sum(xor(v, pool_s(j).vec));
            if d < minDist
                minDist = d;
            end
        end
        if minDist >= dmin
            selected(i) = true;
            chosen_idx(end+1) = i; %#ok<AGROW>
        end
    end
end

pool_k = pool_s(selected);
if numel(pool_k) < min(keep_size, numel(pool_s))
    remain = find(~selected);
    need = min(keep_size, numel(pool_s)) - numel(pool_k);
    extra = remain(1:min(need, numel(remain)));
    pool_k = [pool_k; pool_s(extra)]; %#ok<AGROW>
end
end

function idx = biased_select_indices(costs, M, alpha)
c = costs(:);
c = c - min(c);
p = exp(-alpha * c);
p = p / (sum(p) + eps);
idx = randsample(numel(c), M, true, p);
end

%% =========================================================================
%%                                Cost + 仿真合并
%% =========================================================================
function result = simulate_and_score_single_antenna(designVector, designParams, feed_rc, cfg)
result = simulate_single_antenna_hf(designVector, designParams, feed_rc);
if ~result.isValid
    result.cost = inf;
    result.cost_details = struct('depth', 0, 'bands', 0, 'extraBW', 0, ...
        'ripple', 0, 'bw_reward', 0, 'f0', 0, 'no_band', 1e3);
    return;
end
[result.cost, result.cost_details] = calc_cost_from_s11(result.Y, designParams.freq_sweep, cfg);
end

function [cost, details] = calc_cost_from_s11(s11_mag, f, cfg)
details = struct('depth', 0, 'bands', 0, 'extraBW', 0, 'ripple', 0, 'bw_reward', 0, 'f0', 0, 'no_band', 0);
s11_db = 20*log10(abs(s11_mag)+eps);

roi = (f >= cfg.roi_fmin_GHz*1e9) & (f <= cfg.roi_fmax_GHz*1e9);
s11_db_roi = s11_db(roi);
f_roi = f(roi);

if isempty(f_roi)
    cost = 1e3;
    details.no_band = 1e3;
    return;
end

[bw_hz, seg, nbands, extra_bw_hz] = longest_contiguous_bw(s11_db_roi, f_roi, cfg.thr_db);

f0 = cfg.f0_GHz * 1e9;
df = cfg.df_GHz * 1e9;
if seg.valid
    f_center = 0.5 * (f_roi(seg.i1) + f_roi(seg.i2));
    pen_f0 = ((f_center - f0) / df)^2;%偏移惩罚
else
    pen_f0 = 0;
end

roi_bw = f_roi(end) - f_roi(1) + eps;
bw_norm = bw_hz / roi_bw;
pen_bands   = max(0, nbands - 1);
pen_extraBW = extra_bw_hz / roi_bw;

depth = min(s11_db_roi);
depth_term = max(depth, cfg.depth_thr_db);%深度项

pen_ripple = 0;
if nbands > 1%惩罚多通带中高于-10db的点，压低该点，扩展带宽
    mask = (s11_db_roi < cfg.thr_db);
    mask = mask | ([false; mask(1:end-1)] & [mask(2:end); false]);
    d = diff([false; mask(:); false]);
    starts = find(d == 1);
    ends   = find(d == -1) - 1;
    if numel(starts) > 1
        highest_ripple = max(s11_db_roi(ends(1):starts(end)));
        if highest_ripple > cfg.thr_db
            pen_ripple = (highest_ripple - cfg.thr_db) * 6.0;
        end
    end
end

bw_reward = exp(bw_norm * 4) - 1;

details.depth     = depth_term * cfg.depth;
details.bands     = (cfg.w_multiband_count * 0.2) * pen_bands;
details.extraBW   = (cfg.w_multiband_extraBW * 0.2) * pen_extraBW;
details.ripple    = pen_ripple;
details.bw_reward = -(20 * cfg.w_bw) * bw_reward;
details.f0        = cfg.w_f0 * pen_f0;

cost = details.depth + details.bands + details.extraBW + details.ripple + details.bw_reward + details.f0;
if nbands == 0%无通带的惩罚
    cost = cost + 20;
    details.no_band = 20;
end
if isnan(cost) || isinf(cost)
    cost = 1e3;
end
end

function [bw_hz, seg, nbands, extra_bw_hz] = longest_contiguous_bw(s11_db, f_hz, thr_db)
mask = (s11_db < thr_db);
mask = mask | ([false; mask(1:end-1)] & [mask(2:end); false]);

seg = struct('valid',false,'i1',1,'i2',0);
bw_hz = 0;
nbands = 0;
extra_bw_hz = 0;

if ~any(mask)
    return;
end

d = diff([false; mask(:); false]);
starts = find(d == 1);
ends   = find(d == -1) - 1;

nbands = numel(starts);
bw_list = f_hz(ends) - f_hz(starts);
[bw_hz, k] = max(bw_list);
seg.valid = true;
seg.i1 = starts(k);
seg.i2 = ends(k);
extra_bw_hz = sum(bw_list) - bw_hz;
end

%% =========================================================================
%%                          统一仿真 / 天线建模
%% =========================================================================
function result = simulate_single_antenna_hf(designVector, designParams, feed_rc, needPattern)
if nargin < 4%输入参数小于四，默认不计算辐射方向图
    needPattern = false;
end

feed_rc = double(feed_rc);
N = designParams.pixelResolution_N;
dv = designVector(:)';

if numel(dv) ~= N^2
    error('设计向量长度不匹配：得到 %d，但期望 %d。', numel(dv), N^2);
end

designMatrix = reshape(dv, N, N);
designMatrix(feed_rc(1), feed_rc(2)) = 1;

pixel_indices = find(designMatrix);
if isempty(pixel_indices)
    result.isValid = false;
    return;
end

rectCells = designParams.pixelShapes(pixel_indices);
patchShape = union_rectangles_batch(rectCells, 64, false);
ant = create_antenna_model(patchShape, feed_rc, designParams);

try
    s = sparameters(ant, designParams.freq_sweep);
    s11_mag = abs(squeeze(s.Parameters(1,1,:)));
    result.Y = s11_mag;%S11幅值

    X_data = zeros(N, N, 1, 'single');
    X_single = single(designMatrix);   % 背景0，金属1
    X_single(feed_rc(1), feed_rc(2)) = 2;   % 馈电点标为2
    X_data(:,:,1) = X_single;
    result.X = X_data;
    result.isValid = true;

    if needPattern
        result.pattern = compute_pattern_from_antenna(ant, designParams);
    end
catch ME
    fprintf('仿真失败: %s | %s | at %s:%d\n', ...
        ME.message, ME.identifier, ME.stack(1).file, ME.stack(1).line);
    result.isValid = false;
end
end

function [pattern_data, current_data,ok] = compute_pattern_and_current_only_for_design(designVector, designParams, feed_rc)
%加入防崩溃机制，方向图/电流补算失败"从致命错误改成"丢弃该样本
ok = false;
Fp = numel(designParams.pattern_freqs);
T  = numel(designParams.pattern_theta_deg);
Fc = numel(designParams.current_freqs);
Cc = designParams.current_num_channels;
fineN = designParams.fineN;

pattern_data = zeros(Fp, 4, T, 'single');
current_data = zeros(Fc, Cc, fineN, fineN, 'single');
try
    feed_rc = double(feed_rc);
    N = designParams.pixelResolution_N;
    dv = designVector(:)';
    
    if numel(dv) ~= N^2
        error('设计向量长度不匹配：得到 %d，但期望 %d。', numel(dv), N^2);
    end
    
    designMatrix = reshape(dv, N, N);
    designMatrix(feed_rc(1), feed_rc(2)) = 1;
    
    pixel_indices = find(designMatrix);
    if isempty(pixel_indices)
        error('方向图/电流补算失败：设计中没有金属像素。');
    end
    
    rectCells = designParams.pixelShapes(pixel_indices);
    patchShape = union_rectangles_batch(rectCells, 64, false);
    ant = create_antenna_model(patchShape, feed_rc, designParams);
    pattern_data = compute_pattern_from_antenna(ant, designParams);
    current_data = compute_current_from_antenna(ant, designMatrix, designParams);
    ok = true;

catch ME
    fprintf('[pattern/current failed] %s | %s\n', ME.message, ME.identifier);
    ok = false;
end
end

function pattern_data = compute_pattern_from_antenna(ant, designParams)
Fp = numel(designParams.pattern_freqs);
T  = numel(designParams.pattern_theta_deg);
theta_deg = designParams.pattern_theta_deg;

pattern_data = zeros(Fp, 4, T, 'single');
for kf = 1:Fp
    f_pat = designParams.pattern_freqs(kf);

    % XOZ, Gaintheta
    pat_xoz_gth = pattern(ant, f_pat, designParams.pattern_phi_xoz, theta_deg, ...
        'Type', 'gain', 'Polarization', 'V');
    pattern_data(kf, 1, :) = single(10.^(pat_xoz_gth(:) / 10));

    % XOZ, Gainphi
    pat_xoz_gph = pattern(ant, f_pat, designParams.pattern_phi_xoz, theta_deg, ...
        'Type', 'gain', 'Polarization', 'H');
    pattern_data(kf, 2, :) = single(10.^(pat_xoz_gph(:) / 10));

    % YOZ, Gaintheta
    pat_yoz_gth = pattern(ant, f_pat, designParams.pattern_phi_yoz, theta_deg, ...
        'Type', 'gain', 'Polarization', 'V');
    pattern_data(kf, 3, :) = single(10.^(pat_yoz_gth(:) / 10));

    % YOZ, Gainphi
    pat_yoz_gph = pattern(ant, f_pat, designParams.pattern_phi_yoz, theta_deg, ...
        'Type', 'gain', 'Polarization', 'H');
    pattern_data(kf, 4, :) = single(10.^(pat_yoz_gph(:) / 10));
end
end

function params = design_antenna_parameters(sim_params, geom, N, fineN,overlap_mm)
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
params.overlap = overlap_mm/1e3;%%%改overlap
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
f_stop  = sim_params.fmax_GHz * 1e9;
params.freq_sweep = linspace(f_start, f_stop, sim_params.numFreqPoints);
params.pattern_freqs = (sim_params.pattern_fmin_GHz : ...
                        sim_params.pattern_step_GHz : ...
                        sim_params.pattern_fmax_GHz) * 1e9;
params.current_freqs = params.pattern_freqs;

params.pattern_theta_deg = sim_params.pattern_theta_deg(:).';   % 行向量
params.pattern_phi_xoz   = sim_params.pattern_phi_xoz;
params.pattern_phi_yoz   = sim_params.pattern_phi_yoz;
params.pattern_num_channels = 4;
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
N = params.pixelResolution_N;
pixel_L = params.L / N;
pixel_W = params.W / N;
startX = -params.L/2 + pixel_L/2;
startY = -params.W/2 + pixel_W/2;

cx = startX + (feed_rc(2)-1)*pixel_L;
cy = startY + (feed_rc(1)-1)*pixel_W;

ant = pcbStack( ...
    'BoardShape', params.ground, ...
    'BoardThickness', params.h, ...
    'Layers', {patchShape, params.substrateMaterial, params.ground}, ...
    'FeedDiameter', params.feedDiameter, ...
    'FeedLocations', [cx, cy, 1, 3] );

% 不手动调用 mesh，让软件在求解时自动生成网格
end

function current_data = compute_current_from_antenna(ant, designMatrix, designParams)
Fc = numel(designParams.current_freqs);
fineN = designParams.fineN;
current_data = zeros(Fc, 4, fineN, fineN, 'single');
metalMask64 = build_current_metal_mask(designMatrix, designParams);

for kf = 1:Fc
    f_now = designParams.current_freqs(kf);
    [J_surface, tri_centroids] = current(ant, f_now);
    [Jx64, Jy64] = interpolate_current_to_grid(J_surface, tri_centroids, metalMask64, designParams);

    current_data(kf,1,:,:) = single(real(Jx64));
    current_data(kf,2,:,:) = single(imag(Jx64));
    current_data(kf,3,:,:) = single(real(Jy64));
    current_data(kf,4,:,:) = single(imag(Jy64));
end
end

function metalMask64 = build_current_metal_mask(designMatrix, designParams)
Xc = designParams.current_Xc;
Yc = designParams.current_Yc;
metalMask64 = false(size(Xc));
halfL = (designParams.pixel_L + designParams.overlap) / 2;
halfW = (designParams.pixel_W + designParams.overlap) / 2;
N = designParams.pixelResolution_N;

for rr = 1:N
    for cc = 1:N
        if ~designMatrix(rr,cc)
            continue;
        end
        cx = designParams.pixel_centers_x(cc);
        cy = designParams.pixel_centers_y(rr);
        inRect = (Xc >= (cx - halfL)) & (Xc <= (cx + halfL)) & ...
                 (Yc >= (cy - halfW)) & (Yc <= (cy + halfW));
        metalMask64 = metalMask64 | inRect;
    end
end
end

function [Jx64, Jy64] = interpolate_current_to_grid(J_surface, tri_centroids, metalMask64, designParams)
if size(J_surface,1) ~= 3
    J_surface = J_surface.';
end
if size(tri_centroids,1) ~= 3
    tri_centroids = tri_centroids.';
end

Jx = J_surface(1,:).';
Jy = J_surface(2,:).';
x_scatter = tri_centroids(1,:).';
y_scatter = tri_centroids(2,:).';
z_scatter = tri_centroids(3,:).';

zPatch = designParams.h;
tolZ = 1e-8;
isPatchTri = abs(z_scatter - zPatch) < tolZ;

inPatchXY = (x_scatter >= designParams.patch_origin(1) - designParams.overlap/2) & ...
            (x_scatter <= designParams.patch_origin(1) + designParams.L + designParams.overlap/2) & ...
            (y_scatter >= designParams.patch_origin(2) - designParams.overlap/2) & ...
            (y_scatter <= designParams.patch_origin(2) + designParams.W + designParams.overlap/2);

keep = isPatchTri & inPatchXY;
x_scatter = x_scatter(keep);
y_scatter = y_scatter(keep);
Jx = Jx(keep);
Jy = Jy(keep);

if isempty(x_scatter)
    Jx64 = zeros(designParams.fineN, designParams.fineN);
    Jy64 = zeros(designParams.fineN, designParams.fineN);
    return;
end

XY = [x_scatter, y_scatter];
[~, ia] = unique(XY, 'rows', 'stable');
x_u = x_scatter(ia);
y_u = y_scatter(ia);
Jx_u = Jx(ia);
Jy_u = Jy(ia);

Jx64 = interpolate_complex_scattered(x_u, y_u, Jx_u, designParams.current_Xc, designParams.current_Yc);
Jy64 = interpolate_complex_scattered(x_u, y_u, Jy_u, designParams.current_Xc, designParams.current_Yc);

Jx64(~metalMask64) = 0;
Jy64(~metalMask64) = 0;
Jx64(~isfinite(Jx64)) = 0;
Jy64(~isfinite(Jy64)) = 0;
end

function Vq = interpolate_complex_scattered(x, y, v, Xq, Yq)
if numel(x) == 1
    Vq = complex(zeros(size(Xq)), zeros(size(Xq)));
    [~, idx] = min((Xq(:)-x(1)).^2 + (Yq(:)-y(1)).^2);
    Vq(idx) = v(1);
    return;
end

method = 'natural';
% if numel(x) < 3 || rank([x(:)-mean(x), y(:)-mean(y)]) < 2
%     method = 'nearest';
% end

% try
    Fre = scatteredInterpolant(x, y, real(v), method, 'none');
    Fim = scatteredInterpolant(x, y, imag(v), method, 'none');
% catch
%     Fre = scatteredInterpolant(x, y, real(v), 'nearest', 'none');
%     Fim = scatteredInterpolant(x, y, imag(v), 'nearest', 'none');
% end
Vq = Fre(Xq, Yq) + 1i * Fim(Xq, Yq);
Vq(~isfinite(Vq)) = 0;
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

%% =========================================================================
%%                          当前随机生成器兼容函数
%% =========================================================================
function design = generateSmoothDesignStable4Conn(pixelN, feedPixelIdx, pMetal)
raw = rand(pixelN, pixelN) < pMetal;
[X,Y] = meshgrid(1:pixelN, 1:pixelN);
cx = (pixelN+1)/2;
cy = (pixelN+1)/2;
R2 = ((X-cx)/(0.42*pixelN)).^2 + ((Y-cy)/(0.42*pixelN)).^2;
maskCenter = R2 <= 1.0;

if pMetal < 0.45
    centerBoost = 0.02 + 0.04 * rand();
else
    centerBoost = 0.0;
end
%填充率比较小时，在中心区域做金属增强
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
    else
        bw2(r, c) = true;
    end
    bw2(r, c) = true;
end

%% =========================================================================
%%                                工具函数
%% =========================================================================
function finalize_env()
try
    delete(gcp('nocreate'));
end
fprintf('并行池已关闭，环境已清理。\n');
end

function S2 = make_json_serializable(S)
if isa(S, 'function_handle')
    S2 = func2str(S);
elseif isstruct(S)
    fn = fieldnames(S);
    for k = 1:numel(fn)
        S.(fn{k}) = make_json_serializable(S.(fn{k}));
    end
    S2 = S;
elseif iscell(S)
    for k = 1:numel(S)
        S{k} = make_json_serializable(S{k});
    end
    S2 = S;
elseif isa(S, 'optim.options.BE')
    S2 = make_json_serializable(struct(S));
else
    S2 = S;
end
end

function h = DataHash(A)
if ~isa(A,'uint8')
    A = uint8(A);
end
md = java.security.MessageDigest.getInstance('MD5');
md.update(A(:));
h = char(org.apache.commons.codec.binary.Hex.encodeHex(md.digest())).';
end
