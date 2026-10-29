# Corner plots in Origin from one HDF5 file

## Workflow

1. Write the HDF5 file of the figure, e.g. with `smc.ipynb` (`smc/smc.h5`), `scripts/modes_corner.py`,
   `scripts/paper_slices.py` or `scripts/nn_vs_ode_corner.py`. (The eigenvector-cylinder corner of the original workflow writes
   `results/<run>/cylinder_<net>/cylinder.h5` plus matplotlib previews in `figures/8_*_preview.{png,pdf}`.)
2. **Copy** `cylinder.h5` to Windows (WinSCP/scp, or open it from the network share).
3. **Origin 2021+ (Windows)**, once: *Connectivity → Python Packages* → install `h5py`.
4. Set `FIGURE` at the top of `build_corner.py` to `physical`, `eigen` or `loadings`. Then in the Script Window:

   ```
   run -pyf "C:\path\to\corner-plot\origin\build_corner.py"
   ```

   Start from a new project (*File → New Project*), because the script saves the whole open project. A file
   dialog asks for the `.h5` file. The script writes the following next to it, the project first:
   - `corner_<figure>.opju`: workbooks with every panel, plus the graph;
   - `corner_<figure>.pdf/.png`.

   It also saves the template `corner-<layout>-<figure>-v2.otpu` into Origin's User Files folder. In a German
   Origin this folder is called *Anwenderdateien*, e.g. `C:\Users\ro.heumann\Documents\OriginLab\Anwenderdateien`.

## Combined corner: NUTS, emcee and SMC in one figure

`scripts/combined_corner.py results/<run>_<mode> --smc smc` (for
trpl, sspl and joint) writes `results/<run>_<mode>/corner_combined/corner_combined.h5`:

- **contours:** the 1σ / 2σ HPD regions and the 1D marginals of NUTS, emcee and SMC. They use the usual colours (1σ sky
  blue, 2σ light grey) and differ only in dash pattern: NUTS solid, emcee dashed, SMC dotted;
- **colour map:** the posterior of the fastest sampler whose diagnostics pass (R-hat ≤ 1.01, ESS ≥ 400). This is
  normally SMC, which also gives log Z. Its samples are drawn as dots, each in the colour of the posterior-mass band
  at its position, on black (the colour of "outside 99 %"). With `BACKGROUND = "bands"` the band map is drawn instead.

In Origin: keep `FIGURE = "physical"` and pick `corner_combined.h5`. The script detects the `methods/` group by itself.
The previews `figures/corner_combined_samples.png` and `figures/corner_combined_bands.png` show both variants.
`samplers.csv` (time, evaluations, ESS, R-hat, log Z, which one gives the colour map) and `marginals_by_sampler.csv`
(68 / 95 % HPD intervals per sampler) are written next to the file.

## Sliced grid of the paper + simulations at the best fit and the true parameters

`scripts/paper_slices.py results/<run>_<mode>` writes `results/<run>_<mode>/slices_paper/slices_paper.h5`. Pick it
in `build_corner.py` (`FIGURE = "physical"`). The script then builds three things:

1. **Slice corner.** The paper's sliced grid: 100 × 100 points per parameter pair, all other parameters at the best
   fit, σ_eff = √(σ² + MSE_min). There are no diagonal panels. The colour is the paper's "relative p(Θ|data)": the
   rank of the likelihood ratio to the best fit among all slice points, pooled over the panels (best 1 % … 90 %),
   with the usual 13 Inferno bands. The 1σ / 2σ lines are the probability mass inside each slice. The empty upper
   right holds
   - τ_diff vs QFLS: data, then NN (solid) and ODE (dashed) at the best fit (red) and at the true parameters
     (cyan), each with its ε_rms;
   - below it, PL(t) (mode trpl) or the fitted ss-PL data (modes sspl and joint).

   **Iso-τ₀ lines** (green) are drawn in every n₁,k – τ_p,k panel. They follow
   τ₀ = τ_p a / (1 + k_rad τ_p n₁), with a = √(n₁ N_t)/n_i for N_t ≥ 10 n₁ (photodoping; dashed) and
   a = n₁/n_i for N_t ≤ 0.1 n₁ (dash-dot). Each line is drawn only inside its own regime, so there is a gap of
   two decades in n₁ around n₁ = N_t. N_t and k_rad are the best-fit values, which are the fixed values of the
   slice. τ₀ comes from the shallow regime of the data, τ_diff = τ₀ exp(−QFLS/2kT), fitted inside the QFLS
   window of the trap's `start_point` setting (or the automatic shallow regime when there is no window). It is
   computed on the QFLS calibration of the plotted data. `--tau0 <s> [<s> …]` sets τ₀ by hand (one value, or one
   per trap); `--no-tau0` turns the lines off. The τ₀ values and the best fit's own τ₀ are stored in
   `summary.json` (`tau0`). The lines are also in `csv/tau0_lines_<n1>_vs_<taup>.csv` (decades and physical units).
2. **Graph from your template `t_pl_qfls_tau_diff_paper`.** It has four layers. (a) gets PL/PL₀ vs t in µs (log-log),
   (b) PL/PL₀ vs t in s (linear t), (c) τ_diff vs t and (d) τ_diff vs QFLS. The template keeps its own axis types,
   titles and annotations; only the curves are added and the axes rescaled. The model curves are clipped to the
   time / QFLS range of the data.
3. **Graph from your template `concentration-rates`.** Layer 1: n, p, n_t vs QFLS. Layer 2: electron and hole
   trapping and detrapping rates vs QFLS. Best fit solid, true parameters dashed.

The templates are looked up as `<name>.otpu` / `.otp` in Origin's User Files folder (*Anwenderdateien* in a
German Origin) and in the folders listed in `TEMPLATE_DIRS`, subfolders included. Which curve set goes into which
layer is set in `TEMPLATES` at the top of the script (curve sets `pl_t`, `tau_t`, `tau_qfls`, `conc`, `rates`, `sspl`; append
`_us` to `pl_t` / `tau_t` for time in µs). The script prints the path of each template it found, its number of
layers and the axis titles of every layer, so a wrong mapping is easy to spot. If a template is not found, a plain graph is made instead, and the folders it searched are
printed.

The workbook `simulations_best_fit_truth` uses the column names of the paper export (`fitting/trpl_fitting.py`):
`time_tr_s`, `pl_tr`, `qfls_tr_eV`, `tau_tr_s`, `n_tr_1/cm3`, `p_tr_1/cm3`, `nt_tr_1_1/cm3`,
`e_trapping_tr_trap1_1/cm3/s`, `e_detrapping_…`, `h_trapping_…`, `h_detrapping_…`. Time is in seconds; the paper
column was called `time_tr_ns` but also held seconds. So you can apply your templates by hand to these sheets as
well. The same tables are CSV files in `slices_paper/csv/`. Previews are in `slices_paper/figures/`:
`slices_corner`, `t_pl_qfls_tau_diff` and `concentration_rates`.

The synthetic data have no time axis. Their t and PL come from τ_diff(QFLS) via dt = −τ_diff/(2kT) dQFLS,
anchored on the ODE solution of the true parameters. The NN curves are converted the same way, each anchored on the
ODE solution of its own parameter set.

## Getting the look of the old templates (once)

Run `inspect_template.py` in Origin. A folder dialog asks for the folder that holds `corner-Plot-1Trap.otpu` /
`corner-Plot-2Traps.otpu`. The script opens both and writes `old_template_style.json` (page size, layer geometry,
fonts, ticks) into that folder. Copy the numbers into `STYLE` in `build_corner.py`, or send the JSON back.

## What is in the HDF5 file

The schema is described at the top of `inference/corner_export.py`:

| Group | Content |
|---|---|
| `/inference` | cylinder samples, log L, weights, eigen-system, chords |
| `/corner/physical`, `/corner/eigen` | `params` (labels as Origin rich text, ranges, best fit / truth / start), `layout` table (row, col, panel kind, axis sides, extra E_t − E_g axis), `panels/<x>__<y>/{hpd, density, x, y, sigma1, sigma2}`, `panels/<k>/{x, pdf, hpd_band, intervals68, intervals95}`, `overlays/eigen_*`, `diag/*` (eigen corner: Δln L along v_k, NN/ODE/Laplace) |
| `/loadings` | eigenvectors in decades (rows = parameters, columns = v_k), σ_k |

**Colours.** Every 2D panel uses the same 14 bands, from ≤ 1 % (pale yellow) to outside 99 % (black), in
Inferno, so one colour bar is valid for the whole figure. How the colours are made in Origin:

- **Band per point.** The script decides each point's band in Python, the same way as the matplotlib preview. It
  writes the centre of that band in % into the matrix (0.5, 1.5, 3.5, 7.5, 15, 25, …, 94.5, 99.5).
- **Levels and colours.** Each heatmap gets colour levels at the band edges 0, 1, 2, 5, 10, 20, …, 90, 99, 100 %.
  It also gets one fixed colour per level, which comes from `band_colors_json` in the file. The script sets these
  level by level (`layer.cmap.zN` / `colorN`). No palette file is involved: `Inferno.pal` does not exist in every
  Origin, and the old version then silently fell back to Origin's default colours.
- **Colour bar.** The colour bar is a separate layer to the right of the grid. It holds the same heatmap with the
  14 bands, the densest band on top. The level values (1 %, 2 %, …, 99 %) are text labels at the band edges, and
  the title is rotated to the right. Under the bar come the legends: the iso-τ₀ lines and, for the combined corner,
  the samplers. To move the bar, set `STYLE["cbar"] = [left, top, width, height]` in % of the page. The default
  right margin is 16 %.
- **Check.** The first panel prints `colour map: Origin reports N levels, z = [...]`. If Origin's levels differ from
  0, 1, 2, 5, …, 99, the list of failed steps at the end contains a `[colour levels]` line. Send that line back.

| Element | Colour / style |
|---|---|
| 1σ (68.27 %) HPD contour and 1D band | sky blue `#56B4E9` |
| 2σ (95.45 %) HPD contour and 1D band | light grey `#D9D9D9` |
| best fit | red, dotted |
| truth | cyan |
| start point | green |
| eigen-axes | white |
| ticks on the 2D (off-diagonal) panels | white; a linked frame layer per panel keeps the axis lines black |
| combined corner: samplers | same colours, dash pattern: NUTS solid, emcee dashed, SMC dotted |
| combined corner: samples of the fill sampler | dots in the band colour (`band_colors_json`), on black |

**Coordinates.** log10 parameters are stored and plotted in decades (`10**x` gives cm⁻³ or s). The axes are
linear in decades, and each axis title says so. If you want tick labels shown as 10ⁿ, set this up once per axis
in the Origin axis dialog (Tick Labels → Display → Custom), then save it into the v2 template.

## Status

The first run in Origin (Robin, 2026-10-19, v3 slices) had three problems:

- the colours of the 2D panels were wrong;
- there was no colour bar;
- the project was not saved.

The script was rewritten for all three, following the LabTalk documentation (`layer.cmap` level by level, a
colour-bar layer, project saved before the exports, and the template saved with `template_saveas pg:=… template:=…`
into the User Files folder). On Linux the rewrite was tested only with a mock of `originpro`. Each Origin-specific
call is a small helper wrapped in a `step(...)` guard. If one fails, the script prints `[name] error` at the end and
builds the rest. Send that list back to get it fixed.

Second run (2026-10-19): only the first panel was coloured, all other panels and the colour bar were solid red. The
page stayed at Origin's default 4:3 size, and fonts shrank with each small panel. The paper template got its curves
in the wrong layers. The rewrite after that run:

- sets the colour levels one by one (`layer.cmap.zN`), with no shared dataset;
- creates panels with plain `layer -n` and the E_t − E_g axis with `layer -n X` plus `layer.x.link = 0`;
- sets `layer.fixed = 1` / `layer.factor` so fonts keep their size;
- computes the page size from `page.resx` / `page.resy`;
- runs every LabTalk statement on its own. At the end the script prints **"LabTalk statements that failed"** with
  counts. Until then these failures were silent, so this list is the most useful thing to send back.

Third run (2026-10-19, with the failure list):

- **Colour map.** `layer.cmap` worked only in layer 1 of a page. In every layer added with `layer -n`, each colour
  map property was rejected. So every panel (and the colour bar) is now built as layer 1 of its own graph page. The
  pages are then joined with `merge_graph` (sources deleted, 18 × 18 cm, fonts not rescaled), and every layer is put
  back at its own position, found by the layer name `M<page>L<layer>`.
- **Tick labels.** `layer.x.showLabels` was accepted but had no visible effect, so the per-axis switches
  `layer.x / x2 / y / y2 .showlabel` are set as well.
- **Fonts.** Default sizes are now real points: 7 pt ticks, 8 pt titles.
- **τ₀ legend.** It failed because of the negative position in `label -a 0 -0.8`; it is now created at a positive
  position and then moved.
- **Template graphs.** After plotting, each template layer prints its number of plots, axis ranges, types and links.
  A template that is missing (`concentration-rates` was not in Anwenderdateien) gives a plain two-panel graph with
  log axes.

Fourth run (2026-10-19): colours and colour bar correct. Changes after that run:

- **Layout in cm.** Panels are squares of `STYLE["panel_cm"]` (2.4 cm, gap 0.15 cm). After the merge, `pfit2l` fits
  the page to the figure, because `page.width` keeps Origin's aspect ratio and `merge_graph` ignored `width`/`height`.
- **Log axes.** Every 2D panel has a tick layer with true log10 axes for the `log10_` parameters. It carries white
  major and minor ticks (8 per decade, so the log spacing is visible) on all four sides, and a black frame layer sits
  on top. The 10ⁿ labels and the axis titles are text in the panel layer.
- **Fonts.** 9 pt ticks, 11 pt titles, 9 pt colour bar and legends.
- **Interpolation.** With `STYLE["interpolate"] = True` (the default), the 2D maps are Origin colour-fill contours
  (plot type 226) of the continuous value. Origin interpolates between the grid points, and the 14 levels and colours
  stay the same. `False` gives heatmap blocks.
- **Legend text.** `=` in legend text is written as `\(61)`.

Fifth run (2026-10-19). Square panels, fonts, `pfit2l` and the 10ⁿ text labels all worked. Three things did not:

- **Colour map.** It failed from the 15th page on, the colour bar included. The suspect is the 32 cm temporary pages.
  The fix: panels are built on default-size pages at (0.5, 0.5) cm and get their final cm position only after the
  merge. A failed colour map is retried after a redraw. If it still fails, that panel is rebuilt as a 3× resampled
  heatmap, and the list of failed steps names it.
- **Log tick labels.** The tick layers still showed numeric log labels. Their labels are now hidden as the very last
  step, after `label.pt`, and made transparent as well.
- **Curve panels.** They now use 9 pt labels, `numFormat = 2` (the 10ⁿ format of the paper template) and at most
  about 4 decades per axis. The E_t axis uses a coarser step, and `\-` in its title is a minus.

Sixth run (2026-10-19). The colours were right on all panels, but the off-diagonal panels showed three sets of tick
labels and black ticks. The cause: `merge_graph` (default `groupgraph:=1`) linked the tick and frame layers to the
page's layer 1, which reset their formatting. Changes after that run:

- **Merge.** It now runs with `groupgraph:=0`, and the panel, tick and frame formatting is applied again after the
  merge. The result is white major and minor ticks with only the 10ⁿ text labels.
- **Fit panel.** The wide fit panel keeps one panel width of space on its left.
- **ODE curves.** They are turquoise (`#00CED1`).
- **Curve panels.** They have thinner axes and shorter ticks.

Seventh run (2026-10-19): only the 10ⁿ labels remained, but the ticks were not visible. Changes after that run:

- **Ticks.** They are drawn as white line segments in each panel itself, on all four sides. Major ticks sit at every
  decade, with 8 minor ticks at log₁₀(2…9) in between, as in the old Figure 5.
- **Tick labels.** Each log axis shows at least three 10ⁿ labels.
- **Simulation panels.** For layouts with 6 or more columns, τ_diff(QFLS) is two rows tall (columns 3 to the end).
  PL(t) is 2 × 2 below it, with room above for the x title of τ_diff(QFLS).

## Per-trap corner (`build_corner_traps.py`)

This plot shows only the relations of the parameters within each trap, as in Figure 9 of the paper. It has no
marginal posteriors and no simulation panels.

- **Trap 1** fills the lower-left triangle, with labels at the bottom and on the left.
- **Trap 2** is mirrored into the upper-right triangle, with labels on top and on the right.
- **E_t − E_g** is an extra axis on the n₁ column of each trap.

Run it like `build_corner.py`, on the same `slices_paper.h5` (or `cylinder.h5`). It loads `build_corner.py` from
`BUILD_CORNER` at its top, so keep both scripts in one folder or adjust that path. It writes
`corner_traps.opju/.pdf/.png` next to the `.h5` file.

## tr-PL, ss-PL and joint in one corner (`build_corner_modes.py`)

1. **On rekenbeest:** `python scripts/modes_corner.py results/<base> --gpu 1`, for example
   `results/2trap_85-15_experimental_v3`. It reads the SMC draws in `results/<base>_{trpl,sspl,joint}/smc/smc.h5` and
   writes `results/<base>_joint/corner_modes/corner_modes.h5`, plus a matplotlib preview.
2. **In Origin:** `run -pyf "...\origin\build_corner_modes.py"` and pick that `.h5` file. It writes
   `corner_modes.opju/.pdf/.png` next to it. `build_corner.py` must be in the same folder (`BUILD_CORNER`).

What the figure shows:

- **Lower triangle.** Only tr-PL (`FILL_MODE` in `modes_corner.py`) is filled: 1σ dark and 2σ light blue, drawn as
  an interpolated contour fill of its continuous HPD mass (`<mode>_mass` in the file).
  - Joint (red) and ss-PL (grey) are contour lines on top: 1σ solid, 2σ dashed.
  - The 2D histograms are smoothed with `SMOOTH = 3.5` bins.
  - Robin's choice (2026-10-20): with all three modes filled, tr-PL disappeared under joint, and the overlaps were hard to
    read.
- **Diagonal.** The 1D marginals of each mode.
- **Upper right.** The posterior predictive check of τ_diff(QFLS), from the NN curves of 2000 draws per mode
  (parameter uncertainty), drawn in the same style:
  - the tr-PL band filled, and the joint and ss-PL band edges as solid (68 %) and dashed (95 %) lines;
  - the joint median as a solid bright red line (`MEDIAN_COLOR`);
  - the measured data as open circles drawn as line pieces (symbol plots did not render), thinned to one per
    `data_spacing_cm`.
  - The legend sits in the empty lower left (`LEGEND_POS`).
- **ss-PL panel.** The same style.
- **Colour maps.** Origin left every 4th colour map at its default (solid red) in the first run, so the levels and
  colours are now written a second time on every page before the merge.
- **Workbook `ppc_<mode>`.** The band numbers.
- **ss-PL axes.** They come from the file (`x_log10`, `y_log10`): QFLS vs log illumination (85:15), or log PLQY
  vs linear QFLS (`plqy_vs_qfls`, 1-trap synthetic).
- **Few parameters (1 trap, 4 parameters).** The upper triangle has no room for the PPC panels.
  - τ_diff and ss-PL become squares of 2 cells (`PPC_CELLS`) in a row ABOVE the corner: τ_diff on the left, ss-PL on
    the right.
  - x labels are below them and horizontal; the ss-PL y axis is on the right.
  - The corner starts `PPC_BELOW_CM` lower, and the legend sits at the top of the free upper triangle (Robin,
    2026-10-20).
- **Options at the top of `build_corner_modes.py`:**
  - `SHOW_TRUTH`: the true value (synthetic data) as dashed orange lines in the 2D, 1D and slice panels.
  - `SHOW_BEST_FIT`: the joint best fit as a red star in the 2D and slice panels. The star is drawn as nested line
    outlines.
  - `SHOW_SLICE` (1 trap): the tr-PL slice panel (default τ_p–N_t, `--slice-panel` of `modes_corner.py`) at the right end
    of row 0, at the height of the top marginal.
    - It has its own Inferno colour bar, labelled at every 2nd level.
    - The two analytic iso-τ₀ curves at the best-fit n₁ are labelled in the panel: photodoping (N_t ≥ 10 n₁, dashed)
      and no photodoping (N_t ≤ 0.1 n₁, dash-dot, a vertical line).
    - The legend then moves below it.
  - The truth comes from the config's `true_params`, the best fit from the joint run's `6_best_fit.json`, and the slice
    from `<base>_trpl/slices_paper/slices_paper.h5`. `modes_corner.py` writes all three into the h5.
- **Contour fill refused.** If Origin refuses the colour-fill contour on a page, that page is made again as a 3×
  resampled heatmap. The second colour pass skips layers without a colour map.

## NN SMC vs ODE SMC in one corner (`build_corner_nn_vs_ode.py`)

Same look as the modes corner, without the τ_diff(QFLS) and ss-PL panels.

1. **On rekenbeest:** `python scripts/nn_vs_ode_corner.py <run>`, for example
   `results/1trap_synthetic/1trap_synthetic_trpl`. It reads `<run>/ode_vs_nn/{nn,ode}_smc_weighted_samples.csv`
   (`--nn-source main` takes the NN posterior from `<run>/smc/smc.h5` instead) and writes
   `<run>/ode_vs_nn/corner_nn_vs_ode/corner_nn_vs_ode.h5` plus a matplotlib preview.
2. **In Origin:** `run -pyf "...\origin\build_corner_nn_vs_ode.py"` and pick that `.h5` file. It writes
   `corner_nn_vs_ode.opju/.pdf/.png` next to it. `build_corner.py` and `build_corner_modes.py` must be in the same folder
   (`BUILD_CORNER`).

- NN SMC is filled blue (1σ dark, 2σ light). ODE SMC is drawn as wine-red lines (#7B1E3A): 1σ solid, 2σ dashed.
- The true value is shown as dashed black lines, and the best fit as an open red star drawn with lines. Turn them off
  with `SHOW_TRUTH` and `SHOW_BEST`.
- The diagonal shows both 1D marginals. The legend is in the free upper triangle (`LEGEND_SHIFT_CM`).
