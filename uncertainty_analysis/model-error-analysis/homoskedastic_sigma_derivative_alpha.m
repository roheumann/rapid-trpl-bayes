%% List of folders
folders = {...
    'simulation_results-shallow-plateau_alphas-mu-1',...
    'simulation_results-shallow_photodoping_alphas-mu-1',...
    'simulation_results-no-photodopgin_alphas-mu-1',...
    'simulation_results-trap-high-Nt_alphas-mu-1',...
    'simulation_results-trap-low-Nt_alphas-mu-1'};

%% Physical constants and parameters
kB    = 8.617333262e-5;   % Boltzmann constant [eV/K]
T     = 300;              % Temperature [K]
kT    = kB*T;             % eV
Eg    = 1.625;            % Bandgap [eV]
Nc    = 2.2e18;           % Effective DOS conduction band [cm^-3]
Nv    = 2.2e18;           % Effective DOS valence band [cm^-3]
npulse= 1.6e17;           % Pulse carrier density [cm^-3]
ni    = sqrt(Nc*Nv*exp(-Eg/kT));  % Intrinsic carrier density

% QFLS bounds
qfls_min = 0.9;                          % eV (lower cutoff)
qfls_max = 2*kT*log(npulse/ni);          % eV (upper cutoff)

% alpha uncertainty in log10: factor of 10 -> log10(10) = 1
dlog10_alpha_unc = log10(1e1);

% Grid
N = 256;

fprintf('qfls_max (from 2*kT*ln(npulse/ni)) = %.4f eV\n', qfls_max);
fprintf('ni = %.3e cm^-3\n', ni);

%% Output folder for exports & figures
outdir = 'derivative_analysis_output';
if ~exist(outdir, 'dir')
    mkdir(outdir);
end

%% Common QFLS grid (shared across all folders)
qfls_common = linspace(qfls_min, qfls_max, N).';

%% Storage for per-folder derivative curves on common grid
deriv_folder = nan(N, length(folders));

for f = 1:length(folders)
    foldername = folders{f};
    files = dir(fullfile(foldername, '*.csv'));

    % Exclude the 0D file from the alpha analysis
    idx_0D = startsWith({files.name}, '0D');
    files_alpha = files(~idx_0D);

    if isempty(files_alpha)
        warning('No alpha files in folder %s', foldername);
        continue
    end

    %% Parse alpha from each filename
    nF = length(files_alpha);
    alphas = nan(nF,1);
    for k = 1:nF
        tok = regexp(files_alpha(k).name, ...
            'alpha-([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)', 'tokens', 'once');
        if ~isempty(tok)
            alphas(k) = str2double(tok{1});
        end
    end

    valid = ~isnan(alphas);
    files_alpha = files_alpha(valid);
    alphas      = alphas(valid);

    [alphas_sorted, sort_idx] = sort(alphas);
    files_sorted = files_alpha(sort_idx);
    nF = length(alphas_sorted);

    if nF < 2
        warning('Need at least 2 alpha files in %s, got %d', foldername, nF);
        continue
    end

    %% Pre-load all (qfls, tau) for each alpha, restricted to [qfls_min, qfls_max]
    qfls_cell = cell(nF,1);
    tau_cell  = cell(nF,1);
    for k = 1:nF
        T_k = readtable(fullfile(foldername, files_sorted(k).name));
        qv  = T_k.('QFLS_eV');
        tv  = T_k.('tau_diff_s');
        ok  = ~isnan(qv) & ~isnan(tv) & tv>0 & qv>=qfls_min & qv<=qfls_max;
        qv  = qv(ok);  tv = tv(ok);
        [qv, ia] = unique(qv);
        tv = tv(ia);
        qfls_cell{k} = qv;
        tau_cell{k}  = tv;
    end

  %% Compute centered derivative d log10(tau) / d log10(alpha)
    % For each file k, use the derivative between its neighbors:
    %   - interior (2..nF-1): centered   (k-1, k+1)
    %   - first   (k=1)     : forward    (k,   k+1)
    %   - last    (k=nF)    : backward   (k-1, k  )
    % The derivative is assigned to file k.
    deriv_pairs = nan(N, nF);
    for k = 1:nF
        if k == 1
            k_lo = 1;   k_hi = 2;         % forward
        elseif k == nF
            k_lo = nF-1; k_hi = nF;        % backward
        else
            k_lo = k-1; k_hi = k+1;        % centered
        end

        qv1 = qfls_cell{k_lo}; tv1 = tau_cell{k_lo};
        qv2 = qfls_cell{k_hi}; tv2 = tau_cell{k_hi};
        if numel(qv1)<2 || numel(qv2)<2, continue; end

        log10_tau1 = interp1(qv1, log10(tv1), qfls_common, 'linear', NaN);
        log10_tau2 = interp1(qv2, log10(tv2), qfls_common, 'linear', NaN);

        dlog10_tau   = log10_tau2 - log10_tau1;
        dlog10_alpha = log10(alphas_sorted(k_hi)) - log10(alphas_sorted(k_lo));
        deriv_pairs(:,k) = dlog10_tau / dlog10_alpha;
    end
    %% Per-folder mean derivative
    deriv_folder(:,f) = mean(deriv_pairs, 2, 'omitnan');

    %% Per-file sigma (assigned to lower-alpha file of each pair)
    sigma_pairs = abs(deriv_pairs) * dlog10_alpha_unc;

    %% Plot per-file derivatives
    fig = figure('Visible','off','Position',[100 100 900 550]);
    hold on;
    cmap = lines(nF-1);
    for k = 1:nF-1
        label_k = sprintf('%s (alpha=%.3g)', files_sorted(k).name, alphas_sorted(k));
        plot(qfls_common, deriv_pairs(:,k), '-', 'Color', cmap(k,:), ...
             'LineWidth', 1.0, 'DisplayName', label_k);
    end
    plot(qfls_common, deriv_folder(:,f), 'k-', 'LineWidth', 2.2, ...
         'DisplayName','Folder mean');
    hold off;
    xlabel('QFLS [eV]');
    ylabel('d log_{10}(\tau_{diff}) / d log_{10}(\alpha)');
    title(sprintf('Per-file derivatives: %s', foldername), 'Interpreter','none');
    legend('Location','best','Interpreter','none');
    grid on;
    saveas(fig, fullfile(outdir, sprintf('derivatives_%s.png', foldername)));
    close(fig);

    %% Plot per-file derivatives
    fig = figure('Visible','off','Position',[100 100 900 550]);
    hold on;
    cmap = lines(nF);
    for k = 1:nF
        label_k = sprintf('%s (alpha=%.3g)', files_sorted(k).name, alphas_sorted(k));
        plot(qfls_common, deriv_pairs(:,k), '-', 'Color', cmap(k,:), ...
             'LineWidth', 1.0, 'DisplayName', label_k);
    end
    plot(qfls_common, deriv_folder(:,f), 'k-', 'LineWidth', 2.2, ...
         'DisplayName','Folder mean');
    hold off;
    xlabel('QFLS [eV]');
    ylabel('d log_{10}(\tau_{diff}) / d log_{10}(\alpha)');
    title(sprintf('Per-file derivatives: %s', foldername), 'Interpreter','none');
    legend('Location','best','Interpreter','none');
    grid on;
    saveas(fig, fullfile(outdir, sprintf('derivatives_%s.png', foldername)));
    close(fig);

    %% Plot per-file sigma
    fig_s = figure('Visible','off','Position',[100 100 900 550]);
    hold on;
    for k = 1:nF
        label_k = sprintf('%s (alpha=%.3g)', files_sorted(k).name, alphas_sorted(k));
        plot(qfls_common, sigma_pairs(:,k), '-', 'Color', cmap(k,:), ...
             'LineWidth', 1.0, 'DisplayName', label_k);
    end
    plot(qfls_common, abs(deriv_folder(:,f))*dlog10_alpha_unc, 'k-', ...
         'LineWidth', 2.2, 'DisplayName','Folder mean \sigma');
    hold off;
    xlabel('QFLS [eV]');
    ylabel('\sigma(QFLS)');
    title(sprintf('Per-file \\sigma: %s', foldername), 'Interpreter','none');
    legend('Location','best','Interpreter','none');
    grid on;
    saveas(fig_s, fullfile(outdir, sprintf('sigma_%s.png', foldername)));
    close(fig_s);

    %% Export per-file derivatives and sigmas for this folder
    col_names = arrayfun(@(k) matlab.lang.makeValidName( ...
        sprintf('%s_alpha_%.3g', files_sorted(k).name, alphas_sorted(k))), ...
        1:nF, 'UniformOutput', false);

    Tout_d = array2table([qfls_common, deriv_pairs], ...
        'VariableNames', [{'QFLS_eV'}, col_names]);
    writetable(Tout_d, fullfile(outdir, sprintf('deriv_per_file_%s.csv', foldername)));

    Tout_s = array2table([qfls_common, sigma_pairs], ...
        'VariableNames', [{'QFLS_eV'}, col_names]);
    writetable(Tout_s, fullfile(outdir, sprintf('sigma_per_file_%s.csv', foldername)));

    fprintf('Folder %s: processed %d alphas (centered deriv)\n', ...
            foldername, nF);
end

%% Mean derivative across folders at each QFLS point
mean_deriv = mean(deriv_folder, 2, 'omitnan');

%% Per-point uncertainty sigma(qfls)
sigma_qfls   = abs(mean_deriv) * dlog10_alpha_unc;
sigma_folder = abs(deriv_folder) * dlog10_alpha_unc;   % per-folder sigma curves

%% Homoskedastic sigma for the whole curve (three variants)
valid = ~isnan(sigma_qfls);
Nv_pts = sum(valid);

sigma_rms  = sqrt(mean(sigma_qfls(valid).^2));
sigma_mean = mean(sigma_qfls(valid));
sigma_samp = sqrt(sum(sigma_qfls(valid).^2) / (Nv_pts - 1));

fprintf('\n--- Homoskedastic sigma over QFLS curve ---\n');
fprintf('  RMS      : sqrt(mean(sigma^2))         = %.4e\n', sigma_rms);
fprintf('  Mean     : mean(sigma)                 = %.4e\n', sigma_mean);
fprintf('  Sample   : sqrt(sum(sigma^2)/(N-1))    = %.4e\n', sigma_samp);

%% ---- EXPORTS ----

% 1) Per-folder mean derivative
T_deriv_folders = array2table([qfls_common, deriv_folder], ...
    'VariableNames', [{'QFLS_eV'}, folders]);
writetable(T_deriv_folders, fullfile(outdir, 'mean_derivative_per_folder.csv'));

% 2) Overall mean derivative
T_deriv_overall = array2table([qfls_common, mean_deriv], ...
    'VariableNames', {'QFLS_eV','mean_deriv_all_folders'});
writetable(T_deriv_overall, fullfile(outdir, 'mean_derivative_all_folders.csv'));

% 3) Per-folder sigma
T_sigma_folders = array2table([qfls_common, sigma_folder], ...
    'VariableNames', [{'QFLS_eV'}, folders]);
writetable(T_sigma_folders, fullfile(outdir, 'sigma_per_folder.csv'));

% 4) Overall sigma
T_sigma_overall = array2table([qfls_common, sigma_qfls], ...
    'VariableNames', {'QFLS_eV','sigma_all_folders'});
writetable(T_sigma_overall, fullfile(outdir, 'sigma_all_folders.csv'));

% 5) Summary of homoskedastic sigmas
T_homo = table({'RMS';'Mean';'Sample'}, [sigma_rms; sigma_mean; sigma_samp], ...
    'VariableNames', {'Method','Sigma_homoskedastic'});
writetable(T_homo, fullfile(outdir, 'sigma_homoskedastic_summary.csv'));

%% ---- SUMMARY PLOTS ----

% Mean derivative per folder + overall mean
fig1 = figure('Position',[100 100 900 550]);
hold on;
cmap = lines(length(folders));
for f = 1:length(folders)
    plot(qfls_common, deriv_folder(:,f), '-', 'Color', cmap(f,:), ...
         'LineWidth', 1.2, 'DisplayName', folders{f});
end
plot(qfls_common, mean_deriv, 'k-', 'LineWidth', 2.5, ...
     'DisplayName','Mean across all folders');
hold off;
xlabel('QFLS [eV]');
ylabel('d log_{10}(\tau_{diff}) / d log_{10}(\alpha)');
title('Mean derivative per folder + overall mean');
legend('Location','best','Interpreter','none');
grid on;
saveas(fig1, fullfile(outdir, 'summary_mean_derivative.png'));

% Sigma per folder + overall
fig2 = figure('Position',[100 100 900 550]);
hold on;
for f = 1:length(folders)
    plot(qfls_common, sigma_folder(:,f), '-', 'Color', cmap(f,:), ...
         'LineWidth', 1.2, 'DisplayName', folders{f});
end
plot(qfls_common, sigma_qfls, 'k-', 'LineWidth', 2.5, ...
     'DisplayName','Mean across all folders');
hold off;
xlabel('QFLS [eV]');
ylabel('\sigma(QFLS)');
title(sprintf('\\sigma(QFLS) = |deriv| \\cdot %g', dlog10_alpha_unc));
legend('Location','best','Interpreter','none');
grid on;
saveas(fig2, fullfile(outdir, 'summary_sigma.png'));

%% ---- FINAL SUMMARY FILE ----

% Choose which homoskedastic formula to use as "the" sigma
chosen_method  = 'RMS';   % options: 'RMS', 'Mean', 'Sample'
switch chosen_method
    case 'RMS',    sigma_chosen = sigma_rms;
    case 'Mean',   sigma_chosen = sigma_mean;
    case 'Sample', sigma_chosen = sigma_samp;
end

% Human-readable summary
fid = fopen(fullfile(outdir, 'analysis_summary.txt'), 'w');
fprintf(fid, '=== Derivative / Sigma Analysis Summary ===\n\n');
fprintf(fid, 'Date: %s\n\n', datestr(now));

fprintf(fid, '--- Physical parameters ---\n');
fprintf(fid, '  T       = %g K\n',  T);
fprintf(fid, '  kT      = %.6f eV\n', kT);
fprintf(fid, '  Eg      = %g eV\n', Eg);
fprintf(fid, '  Nc = Nv = %.3e cm^-3\n', Nc);
fprintf(fid, '  npulse  = %.3e cm^-3\n', npulse);
fprintf(fid, '  ni      = %.3e cm^-3\n', ni);
fprintf(fid, '  QFLS range: [%.4f, %.4f] eV (N=%d points)\n\n', ...
        qfls_min, qfls_max, N);

fprintf(fid, '--- Alpha uncertainty ---\n');
fprintf(fid, '  dlog10_alpha_unc = log10(10^%g) = %g\n', ...
        dlog10_alpha_unc, dlog10_alpha_unc);
fprintf(fid, '  (i.e. alpha assumed uncertain by a factor of 10^%g)\n\n', ...
        dlog10_alpha_unc);

fprintf(fid, '--- Homoskedastic sigma (three formulas) ---\n');
fprintf(fid, '  RMS     : sqrt(mean(sigma^2))        = %.6e\n', sigma_rms);
fprintf(fid, '  Mean    : mean(sigma)                = %.6e\n', sigma_mean);
fprintf(fid, '  Sample  : sqrt(sum(sigma^2)/(N-1))   = %.6e\n\n', sigma_samp);

fprintf(fid, '--- Chosen value ---\n');
fprintf(fid, '  Method        : %s\n', chosen_method);
fprintf(fid, '  sigma_chosen  = %.6e\n\n', sigma_chosen);

fprintf(fid, '--- Folders included ---\n');
for f = 1:length(folders)
    fprintf(fid, '  %s\n', folders{f});
end
fclose(fid);

% Machine-readable CSV summary
T_final = table( ...
    {'RMS';'Mean';'Sample';'chosen'}, ...
    [sigma_rms; sigma_mean; sigma_samp; sigma_chosen], ...
    {'sqrt(mean(sigma^2))'; 'mean(sigma)'; 'sqrt(sum(sigma^2)/(N-1))'; chosen_method}, ...
    [dlog10_alpha_unc; dlog10_alpha_unc; dlog10_alpha_unc; dlog10_alpha_unc], ...
    'VariableNames', {'Method','Sigma','Formula','dlog10_alpha_unc'});
writetable(T_final, fullfile(outdir, 'analysis_summary.csv'));

fprintf('\nAll outputs written to folder: %s\n', outdir);
fprintf('Summary files:\n  %s\n  %s\n', ...
    fullfile(outdir, 'analysis_summary.txt'), ...
    fullfile(outdir, 'analysis_summary.csv'));