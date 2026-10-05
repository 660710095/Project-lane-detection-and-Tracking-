# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Binary lane-line segmentation (U-Net) trained on a CULane-derived dataset of Thai dashcam frames.
Not a package — a Jupyter-driven training workspace. Code comments and print output are in Thai; keep
that convention when editing existing files.

## Environment & running

The kernel has torch 2.11.0+cu128 with CUDA available, but **`segmentation_models_pytorch`, `kornia`,
and `cv2` are not installed**. Running the install cell is a required first step, not boilerplate:

```
pip install --user opencv-python matplotlib numpy segmentation-models-pytorch tqdm kornia
```

Use `--user`. Installs into `/opt/conda` do **not** survive a container restart — this was hit in
practice: `cv2`, `kornia`, and `smp` vanished mid-session while `numpy`/`scipy`/`torch`/`matplotlib`
(base image) stayed. `/home/jovyan/.local` is on the mounted volume, so `--user` persists.

`results/` is split by **what kind of measurement** produced the file. Every writer creates its own
subfolder, so new runs land in the right place without manual tidying:

| folder | holds | written by |
| --- | --- | --- |
| `results/training/` | `metrics_*.csv` per-epoch, `training_history_*.png` | both training notebooks |
| `results/test/` | `test_metrics_*.csv` held-out pixel scores | resnet34 notebook, test-eval cell |
| `results/slope/` | `slope_metrics_*`, `slope_baselines_*`, `slope_blindtest_*`, `slope_metric_by_category.csv` | resnet34 notebook + `slope_metrics_lab.ipynb` |
| `results/comparison/` | `metric_perturbation_study.csv`, `metric_comparison_*`, `metric_disagreement_gallery.png` | `evaluate_metrics_comparison.ipynb` |
| `results/frames/` | `result_thai_road_*.png` overlays (produced off-machine, **drivable surface, not lane lines**) | — |
| `results/thai/` | Thai-transfer CSVs, `REPORT_THAI.md`, `REPORT_NIGHT.md`, retention tables | `finetune_thai.py`, `eval_retention.py`, `analyze_night.py` |
| `results/ip/` | scores for masks from the image-processing side | `eval_masks.py` |
| `results/` (root) | `SUMMARY_*.txt` daily work logs | written by hand |

Two more directories at the root are **generated, not source**: `data_subsets/thai_{250,500}/`
holds clip-stratified subsets of the Thai training split (built by `make_thai_subsets.py`; `images/`
and `masks/` are absolute symlinks into `thai_road_lane/`, and val/test are copied unchanged so only
the training set shrinks). `run_datasize.py` trains on them by rebinding `finetune_thai.THAI_ROOT`,
which is why it needs no edit to `finetune_thai.py`.

`eval_masks.py` is the **handoff interface**: it scores a folder of predicted PNG masks against a
split's ground truth and prints the same columns the DL pipeline reports, plus pixel
precision/recall. It needs no model and no GPU, which is the point — `finetune_thai.evaluate()`
takes a torch model as its first argument, so the image-processing side had no way to score its own
output. Verified to reproduce `evaluate()` **exactly (0.00 on every column)** when fed the model's
own masks written out as PNGs, *provided* those PNGs came from the same batched forward pass;
single-image inference shifts pixels at the 0.5 threshold and the numbers drift. It resizes inputs
to `IMG_SIZE` with `INTER_NEAREST` — lossless from an integer multiple (896×1600 → 0.00 difference),
but a non-integer size like 1080×1920 costs 0.005–0.010 IoU, so it warns when it sees one.
`HANDOFF_IP.md` is the one-file brief for the other side.

`models/README.md` is a **generated** table of every `.pth` — architecture, encoder and file format
read out of each state_dict rather than guessed from the filename, plus whatever measured F1 can be
matched to it from `results/`. Regenerate with `python3 make_models_readme.py` after any new run;
never edit it by hand. Consult it before loading any checkpoint, since filenames here carry no
architecture token.

One cross-folder read to know about: `slope_metrics_lab.ipynb` reads
`results/comparison/metric_perturbation_study.csv` to compare against its own per-category table.
It degrades gracefully when absent, so a broken path there fails **silently** — check that cell's
output says it found the file, not that it skipped the comparison.

Two training notebooks, both driven by a CONFIG cell (`ENCODER_NAME`, `ENCODER_WEIGHTS`, `EPOCHS`,
`BATCH_SIZE`, `LEARNING_RATE`, `DATASET_ROOT`); edit it, then re-run the `train(...)` call cell. No
other cell needs re-running between experiments. Both write to relative `models/` and `results/`,
so the kernel CWD must be the project root.

- `train_unet_model.ipynb` — the original. Plain fp32, default `DataLoader`, no normalization.
- `train_unet_resnet34_imagenet.ipynb` — resnet34 + ImageNet variant. Adds ImageNet mean/std
  normalization, `num_workers`/`pin_memory`, AMP, `channels_last`, a pre-flight smoke-test cell, a
  final-epoch model save, and a test-set evaluation cell. Measured ~0.37 s/step at bs32
  (~10 GB VRAM, ~12 min per train pass, ~6 h for 30 epochs).
  `LIMIT_TRAIN` / `LIMIT_VAL` in its CONFIG cell subset the data for quick trials (stride sampling,
  which preserves the category split exactly because the CSVs are ordered by category). Set both to
  `None` for a full run. **Limiting appends `_n{LIMIT_TRAIN}` to `exp_name`** so a trial run never
  collides with the full run's checkpoint — without that, the resume guard would silently skip the
  real training later. Reduce `LIMIT_VAL` alongside `LIMIT_TRAIN`: leaving val at 12,877 makes it
  ~70% of each epoch and cancels most of the speedup.

Container limits: **4 CPUs and 16 GiB RAM** regardless of what `nproc` (64) reports — keep
`num_workers` at 4. Data loading sustains ~342 img/s there, while the bs32 training step runs at
~87 img/s, so training is GPU-bound and more workers will not help.

There is no test suite. Two self-checks exist, both requiring the deps above:
- `python dataset.py <train_csv> <img_dir> <mask_dir>` — CSV/image load + category counts. Pass real
  paths; its argv defaults point at a stale `data/unified_dataset_laneline_v2` location.
- `python model.py` — forward-pass shape check on the from-scratch UNet.

## Dataset

Live root: `/home/jovyan/shared/donut/CarDashCam/CULane` — `images/`, `masks/`, and
`train_list.csv` / `val_list.csv` / `test_list.csv` (~61k / ~13k / ~13k rows).

Splits are **CSV-defined on purpose**: each row is `filename,category,clip_id`, pre-stratified by
category (normal/curve/night) and grouped by clip to prevent leakage between splits. Do not replace
this with a random split (`train.py` still has an unused `random_split` import).

`LaneDatasetFromCSV` derives the mask path by swapping `.jpg` → `.png`, resizes with Kornia, and
binarizes at >127. `IMG_SIZE = (448, 800)` in `dataset.py` is an external contract — it must match the
resolution the masks were drawn at, and stay divisible by 32 for smp encoders. Not a free knob.
Augmentation (`augment=True`, train only) applies hflip to image and mask together, plus brightness
jitter.

## `exp_name` governs outputs and resume

`exp_name = f"{encoder_name}_{encoder_weights}_bs{batch_size}_ep{epochs}"` names all four artifacts:
`models/final_result_epoch_{exp_name}.pth` (last epoch, with optimizer state and full metric history — this is the file `train()` looks for when it resumes), `models/best_model_{exp_name}.pth`
(lowest val_loss), `results/training/metrics_{exp_name}.csv`, `results/training/training_history_{exp_name}.png`.

`train_unet_resnet34_imagenet.ipynb` adds a fifth, `models/unet_lane_model_{exp_name}.pth` — the
final-epoch weights as a bare `state_dict` (the format the older `unet_lane_model*.pth` files use).
Its test-eval cell defaults to that file, and it also re-creates it from the checkpoint if training
completed but the file went missing, since the early-return below would otherwise skip it forever.

It also writes a **sixth**, `models/best_slope_model_{exp_name}.pth` — the epoch with the lowest
median lane-angle error on val, selected independently of `best_model_{exp_name}.pth` (which still
uses `val_loss`, unchanged). Both are kept deliberately: overwriting one criterion with the other
would destroy the baseline needed to answer whether slope-based selection picks a better model.
The two genuinely disagree — a 2-epoch smoke run already chose different epochs.

**Per-epoch slope tracking** (`TRACK_SLOPE` / `SLOPE_VAL_N` in CONFIG, default 2000 images) adds a
`val_slope_median` column to `results/training/metrics_*.csv` and a fifth panel to the history plot. It reuses
`outputs_f` already in the val loop — no second forward pass — selecting a **fixed** image set every
epoch via batch-index arithmetic (`val_loader` is `shuffle=False` with no `drop_last`, so batch `b`
holds indices `[b*bs, (b+1)*bs)`). Fixed matters: resampling each epoch would mix sampling noise into
the epoch-to-epoch comparison that drives checkpoint selection. Cost measured at 6.1 ms/image →
+6 min over a 30-epoch run at 2000 images (+40 min if you use all 12,877).

The median is **pooled over every matched lane**, matching `evaluate_slope`, not a median of
per-image medians — see the two-medians note further down.

**Verified equivalence:** the in-loop value reproduces an independent computation exactly
(`38.7886534864`) but only when AMP, `channels_last`, and batching are all replicated. Those knobs
shifted the number by up to 2.35° on a degenerate 2-epoch model — yet only **0.0044°** on a properly
trained one, so this is not a practical concern; it only matters when reproducing a specific logged
value from an under-trained checkpoint.

**Pre-existing quirk, now shared by both criteria:** the latest checkpoint is saved *before* the
best-model comparison, so `best_val_loss` / `best_slope` stored there lag by one epoch. After a
resume they read back stale (`inf` on a run that had none), which makes the first post-resume epoch
always re-save the best model. This predates the slope work and was matched rather than changed,
since altering it would change `best_model_*` behavior.

On startup `train()` loads the latest checkpoint and resumes — but **returns early** if
`ckpt['epoch'] >= epochs`. So re-running an identical config is a silent no-op; delete
`models/final_result_epoch_{exp_name}.pth` to retrain from scratch, or raise `EPOCHS` to extend a finished run.
Loss is `BCEWithLogitsLoss + DiceLoss`; Adam + `ReduceLROnPlateau` on val_loss. Metrics logged per
epoch: IoU (labeled "accuracy"), recall, F1.

## Legacy / gaps

- `train.py` is a **dead** earlier version: it resolves `dataset_dir` to `/home/jovyan/data/unified_dataset_laneline_v2`,
  which does not exist, so it exits at its CSV-missing guard. `dataset.ipynb` has similarly stale `../` paths.
- `model.py` is an unused hand-written UNet. Neither training path imports it — both use `smp.Unet`, so
  editing it will not change training.
- The PNGs in `results/frames/` were produced elsewhere, and they are a **different task** —
  drivable road *surface*, one solid green band, not lane lines. Not a precedent for lane overlays.
- `plot_thai_examples.py` is the one standalone renderer: Thai frames as
  original | GT | three models, written to `results/thai/examples_thai_3methods.png`. It loads through
  `LaneDatasetFromCSV` (never `cv2.imread` directly) so its numbers match `finetune_thai.evaluate()` —
  verified equal to 1e-9 on a frame. There is still no inference script for arbitrary video, and CULane
  test-set evaluation still lives inside `train_unet_resnet34_imagenet.ipynb`.

## Slope metric (`slope_metrics.py`)

A second way to score the model: instead of pixel overlap, compare the **angle** of each lane line.
Useful because a prediction shifted a few pixels sideways tanks IoU while the steering direction is
still right.

**`slope_metrics.py` is the single source of truth** — a pure numpy/cv2/scipy module with no torch or
smp dependency, imported by both `train_unet_resnet34_imagenet.ipynb` and
`evaluate_metrics_comparison.ipynb` (notebooks cannot import each other). Edit the formulas there,
never in a notebook cell. `python slope_metrics.py` runs its own self-check.

It holds `extract_lanes` / `angle_error` / `match_lanes` / `slope_metrics_for_pair` /
`slope_metrics_for_lanes`, plus `mask_iou` (per-image numpy IoU), `AnglePositionPrior` (the
no-learning baseline), and mask perturbations `shift_mask` / `dilate_mask` / `erode_mask` /
`rotate_lanes` / `drop_one_lane` used to probe what each metric can see.

The notebook then supplies `evaluate_slope` (whole test split, ~2.5 min), `plot_slope_blindtest`
(overlay grid), and `evaluate_slope_baselines` (null test + baseline scale, ~1 min).

Method: connected components → collapse each lane to a **row-wise-centroid centerline** → least
squares `x = a·y + b` → angle → Hungarian match GT↔pred on x near the image bottom.

Two traps encoded in the code, both found by measurement — do not "simplify" them away:
- **Do not use PCA** on the pixel cloud. Lane masks are thick bands; thickness and curvature skew
  the principal axis, which returned `5°` for a clearly vertical lane. The centerline step fixes it.
- **`cv2.line` rejects the array** from `image_t.permute(1,2,0).numpy()` — torch's permute yields a
  non-C-contiguous view and numpy preserves that memory order, so wrap it in `np.ascontiguousarray`.

Also note `numpy` is **not** imported by the notebook's import cell; the slope cell imports it itself.

Reported numbers separate **detection** (`lane_recall` / `lane_precision`) from **slope correctness**
(`median_err`, `acc@5°`), and `*_wf` columns repeat everything for the ~90% of GT lanes that fit a
straight line well (`rms ≤ 10px`) — the remaining 10% are curves or merged lanes where a single
slope is genuinely undefined.

## Is the ruler straight? (`slope_metrics_lab.ipynb`)

The notebook form of `python slope_metrics.py`: same self-checks, run cell by cell, results as
pandas tables instead of print lines, and **split by road type** (normal / curve / night).
It **imports** from `slope_metrics.py` and holds no copy of the formulas — it even displays the
code via `inspect.getsource()`, so what you read is the live module. No torch/smp, no GPU, ~2 min.

Distinct from `evaluate_metrics_comparison.ipynb`, and the two are worth keeping straight: this one
asks *"is the ruler straight?"* (perturb the GT by known amounts, no model involved), that one asks
*"how good is the model?"*. Balanced sampling here — 400 images **per category**, not a stride over
the whole split, which would leave curve with ~70 frames.

Result worth keeping: **the metric is not biased by road type.** Across all three categories the
angle-error column is flat (max−min ≤ 0.03° on every rotation row). That is what licenses the
curve finding in the comparison notebook — if the metric itself ran hot on curves, "IoU −1.2% but
angle +41%" would mean nothing.

**Trap — the small IoU differences between categories have two drivers, and neither is the obvious
one.** Two plausible-sounding explanations were tested and refuted by the data: *"fewer lanes per
image → higher IoU"* (night has fewer lanes than normal yet lower IoU) and *"lanes spaced further
apart → higher IoU"* (curve has the *narrowest* spacing and the *highest* IoU). What actually
drives it:

- **Shift → lane thickness** `w = area / nrows`. A band of width `w` shifted by `d` gives
  IoU = `(w−d)/(w+d)` exactly. Predicting from per-lane `w` matches the measurement to within
  0.015 at both 5px and 10px, and reproduces the ordering (curve 24.8px > normal 22.1 > night 21.5).
- **Rotate → lane length**. `rotate_lanes` turns each lane about its own centroid, so a longer lane
  swings its tips further. The ordering inverts exactly (curve 149 rows → IoU 0.302, normal 223 →
  0.241, night 233 → 0.239).

The notebook re-derives both orderings in code and prints ✓/✗ rather than asserting them, and it
re-tests the two refuted explanations on every run, so a changed `N_PER_CAT` cannot quietly
resurrect a wrong story.

## Comparing the two metrics (`evaluate_metrics_comparison.ipynb`)

Answers "IoU or slope — which is better?" by refusing the question as posed (0.535 vs 0.95° share no
units) and asking instead **which kind of error each metric can see**. It perturbs the GT by known
amounts and reports both metrics side by side. Writes everything under `results/comparison/` —
`metric_perturbation_study.csv`, `metric_comparison_by_category.csv`, and two PNGs.

Findings worth keeping (all measured, and two of them contradict the guesses that preceded them):

- **IoU is not blind to rotation** — it drops hard for rotation *and* translation alike. Its real
  limitation is that it **cannot tell them apart**: shifting lanes 20px gives IoU 0.088 at a 0.00°
  angle error, rotating them 10° gives IoU 0.123 at 9.93°. Same IoU, opposite meaning.
- **On curves, IoU hides the problem.** IoU says curve ≈ normal (0.544 vs 0.550, a 1.2% gap) while
  the slope metric says curve is clearly worse — **1.4×** by the pooled median (0.89° → 1.25°), 2.0×
  by the per-image one (1.11° → 2.20°). The ratio depends on which median you read (see below); the
  direction does not. Measuring only IoU would miss that curves are the model's weak spot.
- Neither metric sees a missing lane: deleting one lane leaves the angle error at 0.00° and only
  `lane_recall` moves (1.000 → 0.682). Always read recall alongside.
- Per-image Spearman between IoU and angle error is only about −0.5 to −0.6, so they genuinely rank
  individual frames differently.
- The clearest cases sit at IoU ≈ 0.00 with a 1–2° angle error: zero pixel overlap, direction still
  right. Read `median_err`, not `mean_err` — inspecting the gallery shows the ~29° upper-tail cases
  are usually a thin, near-vertical GT lane matched to the wrong prediction, i.e. a limitation of the
  Hungarian matching step rather than the model aiming badly.

**Which metric wins, per road type** (deltas from `normal`, replicated on both models):
curve moves IoU only −1.2% / −4.3% but the angle **+41% / +59%** — they disagree, and slope is the
one that sees it. Night moves IoU −13.2% / −14.1% against an angle change of only +22% / +7.4%, with
recall down too — they agree, because the night failure is *detection*, not aiming, which IoU and
recall already report. So slope earns its keep on curves; on night it is the less sensitive of the
two. Report both.

**Two different medians, both labelled "angle error" — do not diff them blindly.**
`evaluate_slope` (training notebook, `results/slope/slope_metrics_*.csv`) pools every matched lane pair
across all frames and takes one median → **0.95°**. `paired_eval` here medians *within* each frame
first, then medians those → `slope_median_per_image`, **1.27°**. Median-of-medians is not the pooled
median; both are correct, they answer different questions. `metric_comparison_by_category.csv`
therefore carries **both** columns, and the summary and model-ranking use `slope_median_pooled` so it
lines up with the rest of the project.

Perturbation helpers live in `slope_metrics.py`. The rotation one turns **each lane about its own
centroid**, not the whole image, so direction changes while position does not — that separation is
what makes the table interpretable. Two guards run inside the notebook: identity must give IoU 1.000
and 0.00°, and a d-degree rotation must read back as ~d degrees.

Matplotlib here has no Thai font (DejaVu/Liberation only), so text drawn **on figures** is English
while markdown and `print` output stay Thai.

## Pre-existing checkpoints and the normalization trap

`models/unet_lane_model.pth`, `models/unet_lane_model_bs32_ep30.pth`, and
`models/best_unet_lane_model_bs32_ep20.pth` are all **resnet34 smp U-Nets** (verified: BasicBlock,
layer counts 3/4/6/3) despite carrying no encoder token in their filenames.

They were trained on un-normalized `[0,1]` input, so **evaluating them with ImageNet normalization
silently wrecks the numbers** — measured on `unet_lane_model_bs32_ep30.pth`: F1 **0.676** without
normalization vs **0.414** with it. The test-eval cell in the resnet34 notebook therefore pairs each
model with its correct `normalize` flag in an `EVAL_TARGETS` table; select one via `EVAL_CHOICE`
rather than setting the path and flag separately.

## The six `results/evaluate/` runs — weights, architecture, normalization

Weights for these arrived later as `Best_model.zip` and are now flat in `models/`. Three facts were
established by measurement, not by trusting filenames:

- **Architecture is detected from the state_dict**, not the name: `decoder.aspp.*` means
  `smp.DeepLabV3Plus`, `decoder.blocks.*` means `smp.Unet`. `load_seg_model(path, encoder, device)`
  in the resnet34 notebook (cells 20 and 26) does this and builds the right class, so both
  architectures load through the same code path. It raises rather than falling back to `strict=False`.
- **All six need `normalize=False`.** Checked by scoring each both ways on 200 val images: raw input
  gives F1 ~0.70, ImageNet normalization gives 0.004–0.40. The gap is unmissable, so this is a
  detectable property — never guess it.
- **F1 measured from these weights matches the per-epoch CSVs** to within +0.009…+0.024 (slightly
  high because the check pools TP/FP/FN while training averaged per batch), which confirms each
  `.pth` is paired with the right run.

**Trap: `best_model_*.pth` is the `val_loss` argmin, not the F1 argmax.** Verified — the epoch stored
in each checkpoint equals `argmin val_loss` exactly, and that differs from `argmax val_f1` in 3 of 4
runs (e.g. resnet50 imagenet: weights are ep9, best F1 is ep16). Any table putting CSV-derived F1
beside weight-derived slope numbers is mixing two epochs; `results/evaluate/REPORT.md` says so
explicitly.

Slope results for all six live in `results/slope/slope_metrics_{run}.csv`. Two findings worth
keeping, both cases where F1 and slope disagree:

- `brightonly` is last by F1 (0.6772, clearly below the pack) yet mid-pack on angle (1.03°) with the
  second-best acc@5° (84.2%) — it places the band worse but still aims correctly.
- For DeepLabv3+ resnet50, `best_model` (ep4) is worse than `final_model` (ep30) on every slope
  column (1.17° vs 1.04°, recall 0.771 vs 0.872). Selecting a checkpoint on `val_loss` alone can pick
  an undertrained epoch.

## Thai road transfer (`thai_road_lane/`, `finetune_thai.py`)

The project's actual target is Thai roads, but every model is trained on CULane (Chinese roads).
`thai_road_lane/` holds 1,037 labelled frames (749/97/191), already group-stratified with **no clip
leakage across splits** (verified). Masks are binary 448×800, matching `IMG_SIZE` — no conversion.

`train()` in the notebooks cannot fine-tune: it only resumes from its own checkpoint and has no way
to accept starting weights from another file. `finetune_thai.py` exists for that. It exposes `run()`
so the CLI and `finetune_thai.ipynb` drive identical code, and takes `--mode finetune|scratch`.
All Thai runs use `normalize=False`, matching the CULane source weights.

**Measured domain gap** (`unet_lane_model_bs32_ep30`, same model on both test sets):
IoU 0.535 → 0.324 (−39%), F1 0.676 → 0.490, angle 0.95° → 2.60° (+174%) — but
**`lane_recall` is unchanged at 0.899 → 0.900**. The model still *finds* Thai lanes; it places and
aims them worse. That asymmetry is the finding, and IoU alone cannot express it.

**Three-way comparison** on Thai test (`results/thai/REPORT_THAI.md`), all using the slope-selected
checkpoint: zero-shot 0.354 IoU / 1.90° / 0.953 recall; fine-tune 0.458 / 2.03° / 0.827;
Thai-only-from-scratch 0.466 / 2.57° / 0.821. **The two metric families pick different winners** —
IoU/F1 favour Thai-only, angle/recall favour zero-shot. Training on Thai data improves placement and
degrades direction.

**Trap — the checkpoint criterion matters more than the training method.** `finetune_thai.py` now
saves three checkpoints per run (`best_f1`, `best_slope`, `best_recall`) because selecting on val F1
alone picks the worst-aiming epoch in *both* experiments. From one identical fine-tune run, night
angle error is **26.47°** at the F1-chosen epoch and **4.28°** at the slope-chosen one — 6× better
for free. The spread between criteria dwarfs the spread between the three training methods.

**Trap — fine-tuning forgets night.** Thai train is only 11% `night` (86 frames), so fine-tuning on
it erases the night robustness inherited from CULane: night angle goes 2.37° (zero-shot) → 4.28°
(fine-tune) → 15.90° (scratch). The val history shows the damage starting at epoch 3 while val F1 is
still climbing — another case where F1 hides the regression.

**Training is not deterministic here.** Re-running fine-tune with an identical config moved the
F1-selected epoch from 5 to 7 and shifted the reported numbers. `REPORT_THAI.md` carries the latest
run; `results/SUMMARY_2026-09-17.txt` carries the earlier one, and they disagree slightly for this reason.

Reading the per-category tables is mandatory: Thai test is **82% `normal`** (157 of 191), with only
17 frames each for `curve` and `night`, so the ALL row is dominated by daytime straight roads and
the two interesting categories are small enough to swing.

**Trap — Thai `night` is one scene, not 17 samples.** Counting `clip_id`: night is **2 clips** in
train (86 frames), **1** in val (20), **1** in test (17, `sathon_road_night01`); `curve` test is 2
clips. So every Thai night number in `REPORT_THAI.md` — including the 2.37° / 4.28° / 15.90° spread —
measures one stretch of Sathon road at night. Only `normal` (6 test clips) has enough scenes to
carry a conclusion on its own. Never add Thai val night as a fourth `CRITERIA` entry: that selects
a checkpoint on one scene.

## Fixing the night regression (`eval_retention.py`, `analyze_night.py`)

Because Thai night is one clip, retention is measured on **CULane test night — all 2,553 frames
across 16 clips**, broken out per clip. `eval_retention.py` gets that breakdown for free by passing
`evaluate()` a `filename → clip_id` map instead of `filename → category`; it takes ~70 s, so there
is no reason to subsample. Results append to `results/thai/retention_night_full.csv`
(one row per checkpoint × scope). `analyze_night.py` turns those plus `{tag}_by_criterion.csv` into
`night_summary.csv` and two PNGs.

`finetune_thai.py` gained four **opt-in** flags; omitting them reproduces the old behaviour exactly
(verified: same optimizer param list and order, same first 7 history columns, sampler branch never
entered). `--balance-cats` and `--rehearse-frac` share one `WeightedRandomSampler` whose
`num_samples` is always `len(thai_train)`, so steps/epoch match the control and `epoch` stays
comparable across arms. `history_*.csv` gained `val_night_*`, `val_curve_median_err`,
`val_normal_median_err` and `thai_imgs_seen` (the x-axis to use when rehearsal halves Thai images
per epoch).

| flag | effect |
| --- | --- |
| `--freeze-encoder` | decoder + head only |
| `--balance-cats` | every road category drawn equally (night 11% → ~33%) |
| `--rehearse-frac F` | mix CULane into fraction F of each batch |
| `--seed S` | for deliberately repeating the control under a different seed |

**Trap — `requires_grad = False` does not freeze the encoder.** Measured by hashing
`encoder.state_dict()` before and after one epoch: with `requires_grad=False` alone the hash
**changes**, because BatchNorm updates `running_mean`/`running_var` outside the gradient path.
`model.encoder.eval()` must be called inside the epoch loop (`model.train()` resets it every epoch);
only then is the hash bit-identical. Frozen params must also be filtered out when constructing Adam.

**Results** (`results/thai/REPORT_NIGHT.md`, all on the slope-selected checkpoint, every Thai IoU
from the same `finetune_thai.evaluate()` path — the zero-shot row was re-measured through it and
matches its stored CSV exactly; two control seeds differ by only 0.004° / 0.003 recall, so the gaps
below are real):

| arm | ep | CULane night angle | recall | Thai IoU | verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| zero-shot (ceiling) | — | **1.07°** | 0.839 | 0.354 | — |
| control | 1 | 1.35° | 0.842 | 0.429 | floor |
| freeze encoder | 3 | **1.10°** | 0.811 | 0.444 | partial (recall below guard) |
| balanced categories | 17 | 1.58° | 0.765 | 0.448 | **failed** (3/16 clips better) |
| **rehearse CULane 50%** | 25 | 1.14° | **0.851** | **0.478** | **works** (15/16 clips) |

The finding is not "rehearsal reduces the angle a bit". It is that **the control only avoids
forgetting by not learning** — its slope criterion stops at epoch 1, Thai IoU 0.429. Rehearsal
trains to epoch 25 and reaches Thai IoU 0.478 (matching the old `finetune_ssl` run's 0.479) while
CULane night stays at 1.14° / 0.851 instead of that run's 1.51° / 0.738.

Balanced sampling failing is informative, not a null: Thai train holds only 2 night clips, so
oversampling night to 33% shows the same two scenes repeatedly. Thai F1 rises to 0.619 while CULane
night gets worse on every column — exactly the confound the CULane probe exists to catch.

**The "night" framing turned out to be wrong, and this is the more useful finding.** Measuring
CULane `normal` (8,338) and `curve` (2,407) as well shows fine-tuning costs IoU almost uniformly
across road types — −0.034 normal, −0.042 curve, −0.035 night — and on angle **`curve` degrades more
than `night`** (+0.35° vs +0.27°, with normal +0.14°). It is domain-wide drift, not a night-specific
failure. Night only *looked* special because the alarming 26.47° came from Thai test night, which is
**one clip**. Rehearsal recovers 58–75% of the angle gap in every category and raises IoU in all
three, so the fix generalizes; freeze encoder wins on normal-road angle (0.95°) but loses recall
everywhere (0.841 / 0.841 / 0.811). Run `eval_retention.py --category normal|curve` to reproduce.

The slope column above is not a trade-off against Thai angle, though it reads like one: control's
slope criterion collapses to epoch 1, so the two rows are different points on the training curve.
Compared **at the same criterion**, rehearsal wins outright on `f1` (Thai IoU 0.470 vs 0.463 *and*
2.01° vs 2.61°) and on `recall` (1.57° vs 1.83°). Ship `models/thai_ft_rh_s1_best_f1.pth`.

Worth noting for the checkpoint-criterion thread: the slope criterion picks ep1 under control but
ep25 under rehearsal. The criterion behaves differently because the val-night signal it selects on
stops collapsing — selection rule and training method are not independent.

**Replicated:** rehearsal was re-run under a second seed and holds on every column — CULane night
angle 1.14° vs 1.11°, recall
0.851 vs 0.832, IoU 0.456 vs
0.462, and 15/16 clips improved in both. The
spread (0.03° / 0.019 recall) is far smaller than the 0.21° distance to the control. What did *not*
replicate is the epoch: the slope criterion picked **ep25** under one seed and
**ep12** under the other, while landing in the same place — more evidence that the
selection rule and the training method are entangled.

**Not worth trying: `--freeze-encoder` combined with `--rehearse-frac`.** The primary metric is
already saturated — rehearsal sits 0.068° from the zero-shot angle ceiling and *above* it on recall,
so the combination's maximum possible gain is 0.068° while risking freeze's recall loss (−0.040) and
Thai IoU loss (−0.033). The remaining gap is elsewhere: Thai test IoU 0.478 vs the 0.535 the same
model reaches on CULane, which is a question about Thai *data volume*, not about forgetting.

## The Thai labels follow a different convention from CULane (`check_annotation_style.py`, `check_convention_effect.py`)

Checked before asking anyone to label more frames. Measured on ground-truth masks only for the
geometry, then with the zero-shot CULane model for the effect. Full report:
`results/thai/REPORT_ANNOTATION.md`.

Three differences, all measured on 200 train images per category per dataset:

| | CULane | Thai |
| --- | ---: | ---: |
| lanes per image (`normal`) | 3.00 (3 lanes 54%, 4 lanes 34%) | 2.00 (**2 lanes 82%**) |
| lane top end, fraction of height | 0.498 | 0.681 |
| lane bottom end | 0.998 | 0.998 |
| GT lanes with `rms > 10px`, `curve` | 7.7% | **32.5%** |

Thai labels **only the ego lane's two boundaries**, starts them much lower in the frame, and — unlike
CULane — its `curve` annotations are genuinely curved. (This confirms the note further up that
CULane's "curve" tag describes the scene rather than the annotated segment; the Thai set is the
opposite.)

**The effect splits by category, and only one of the three is a model failure.** Zero-shot on test,
pixel recall vs pixel precision separates "cannot see the lane" from "draws lanes the label omits":

| | pixel recall | pixel precision | IoU | GT lanes | predicted lanes |
| --- | ---: | ---: | ---: | ---: | ---: |
| CULane normal | 0.671 | 0.723 | 0.534 | 3.03 | 3.62 |
| **Thai normal** | **0.727** | **0.438** | 0.376 | 1.77 | **3.05** |
| Thai curve | **0.221** | 0.514 | 0.183 | 1.82 | 2.12 |
| Thai night | 0.497 | 0.510 | 0.336 | 1.82 | 2.94 |

On `normal` — **82% of Thai test** — pixel recall is *higher* than on CULane and lane recall is
0.989. The model sees the Thai lanes. It loses IoU to precision,
drawing 1.72× the
labelled lane count, and `convention_extra_lanes.png` shows those extra lanes sitting on real road
markings. `curve` is the opposite — pixel recall collapses to
0.221 with no over-prediction, so it is a real model failure.

**This qualifies the domain-gap claim above.** "IoU 0.535 → 0.324 while lane recall holds" is
correct, but attributing it to *placement* is incomplete: on `normal` a large part is the label
convention, not the model. The `REPORT_DATASIZE.md` conclusion (label more curves) is unaffected,
because `curve` is a genuine failure.

**Before any new labelling, fix the convention first** — how many lanes, how far up the frame, and
whether curves are traced as curves. Those are exactly the three things measured to differ. Then
decide: relabel to CULane convention (gains on `normal` for free, costs a pass over 1,037 frames), or
keep ego-lane-only (matches the end use, but Thai IoU is then not comparable to other lane-detection
work, and fine-tuning has to teach the model to stop drawing neighbours).

## How much more Thai data is worth labelling (`run_datasize.py`, `make_datasize_report.py`)

Trained on 245 / 505 / 749 Thai images (clip-stratified subsets from `make_thai_subsets.py`), same
config throughout — fine-tune from the SSL CULane weights, `--rehearse-frac 0.5 --seed 1`, 30 epochs,
identical val/test. Full report in `results/thai/REPORT_DATASIZE.md`.

**Read two checkpoint criteria, not one.** The slope criterion picked epochs 17 / 8 / 25 across the
three points, and that alone makes Thai IoU look like it climbs 0.462 →
0.478. Under the f1 criterion — same selector at every point — the curve is
flat: 0.467 / 0.467 / 0.470.
Tripling the data buys +0.003 IoU there, and both criteria agree that **angle error gets worse and
lane recall drops** as Thai data grows.

**The ALL row lies, because Thai test is 82% `normal`.** Per category (f1 criterion, 245 → 749):

| category | train images | IoU | Δ IoU | Δ recall |
| --- | --- | --- | ---: | ---: |
| normal | 150 → 455 | 0.553 → 0.544 | -0.009 | -0.043 |
| curve | 69 → 208 | 0.147 → 0.196 | **+0.049** | **+0.161** |
| night | 26 → 86 | 0.202 → 0.239 | **+0.037** | +0.065 |

(That table is the **f1** checkpoint at every point — the 749 row is ep15, not the ep25 model the
slope rows above use. Two different models; don't diff the tables.)

`normal` is **saturated** — three times the daytime straight-road images made IoU slightly *worse*.
So the answer to "should we label more?" is **"label curves, not more daytime straight road."**

Rest that claim on `curve` alone: it goes 3 → 7 → 12 clips, so its gain measures added images.
`night` moves too, but it is **1 clip at both 245 and 505 and 2 clips at 749**, so its 245→749 delta
is partly "a second scene appeared", a different mechanism — the two categories' numbers are not
comparable. Night is still worth collecting, because 2 clips in the full set is too few to measure
anything, not because this curve proved it pays.

Data volume is **not** a lever on forgetting: CULane night retention is 1.23° / 1.22° / 1.14° across
the three points, all a similar distance from the 1.07° ceiling. Rehearsal controls that, and it was
on at the same strength everywhere.

## Slope in the loss, not just the metric (`slope_loss.py`, `make_slopeloss_report.py`)

The advisor asked whether the loss was changed to slope or whether training stayed normal and only
the measurement differed. It was the latter, throughout — every run used `BCEWithLogitsLoss +
DiceLoss`. `slope_loss.py` closes that gap. Full report: `results/thai/REPORT_SLOPELOSS.md`.

`slope_metrics.py` has three non-differentiable steps (`cv2.connectedComponentsWithStats`,
`np.linalg.lstsq`, `scipy linear_sum_assignment`) and must stay torch-free, so the differentiable
version lives in `slope_loss.py` and follows DETR's trick: **match under `no_grad`, compute the loss
with grad**. Matching reuses `slope_metrics_for_pair`; the angle comes from a soft row-wise centroid
of the raw probability gated by that lane's component mask, then `torch.linalg.lstsq` and
`atan2(1,a)`. `lane_components()` calls `extract_lanes` per component rather than copying the
formula, so `soft_angle` on a binarised mask reproduces `extract_lanes(...)["ang"]` to **0.0000°**
(39 lanes, checked by `python3 slope_loss.py`). Enable with `--slope-loss-weight L`; cost is
**1.84× per step** (805 → 1482 ms at bs16).

Results, f1-selected checkpoint, same data and schedule, only λ differs:

| λ | Thai IoU | Thai angle | Thai recall | normal / curve / night angle |
| ---: | ---: | ---: | ---: | --- |
| 0 (control) | 0.470 | 2.01° | 0.856 | 1.72° / 5.94° / 4.87° |
| 0.1 | 0.470 | 1.99° | 0.862 | 1.68° / 6.27° / 4.62° |
| 0.3 | 0.468 | **1.87°** | 0.865 | 1.74° / 5.99° / 4.92° |

**Trap — the ALL row moved without any category moving.** The pre-registered bar was "Thai angle
improves ≥0.10° with IoU dropping ≤0.02", and λ=0.3 cleared it (+0.14°, −0.002). But per category it
is **not better than control anywhere** (1.74 vs 1.72, 5.99 vs 5.94, 4.92 vs 4.87). The ALL row is a
pooled median over *matched* lane pairs, so when `lane_recall` rises (normal
0.896 → 0.903, night
0.710 → 0.742) the
*composition* of pairs changes and the pooled median shifts on its own. A per-category guard
(≥2 of 3 categories must improve) was added to `make_slopeloss_report.py` **after** seeing this —
it makes the criterion stricter, not looser, and the report says so. Verdict: λ=0.1 **failed**,
λ=0.3 **not established**.

What the experiment does establish: the term is real and gets optimized (`train_slope_loss` falls
over 30 epochs, harder at higher λ; ≥1,533 matched pairs per epoch, never 0), and it **does not cost
IoU** (−0.002). The pre-run prediction — "angle improves a little, IoU drops a little" — was wrong on
both halves. Untested: λ ≥ 1.0, and more than one seed, which matters because the per-category
differences (0.02–0.05°) sit at the 0.03° seed spread measured earlier.

Mechanism limit to repeat whenever this is cited: gradient flows only through lanes that were
**already matched**, so this teaches aiming, not detection. It cannot help Thai `curve`, which fails
at pixel recall 0.221.

## Experiment log

`results/training/metrics_*.csv` per-epoch, `results/test/test_metrics_*.csv` held-out test scores.

| run | val F1 | note |
| --- | --- | --- |
| `bs32_ep20` | 0.693 | resnet34, produced off-machine |
| `resnet50_ssl_bs16_ep30` | 0.680 | train F1 0.94 — heavy overfit gap |
| `deeplabv3plus_bs32_ep15` | 0.693 | no code in repo produces this |
| `timm-efficientnet-b0_imagenet_bs32_ep30` | — | header-only; died before epoch 1 |

Test-set (13,298 frames), `unet_lane_model_bs32_ep30.pth`:

- pixel metrics — IoU 0.535 / P 0.687 / R 0.655 / F1 0.676
- slope metrics — median angle error **0.95°**, acc@5° 84.1% (matched) / 75.6% (strict),
  lane recall 0.899, lane precision 0.788. By category: `normal` 0.89° > `night` 1.08° >
  `curve` 1.25°. Curves are hardest — but **not** because a straight line is a bad fit there:
  measured on GT labels, the share of lanes that fit a straight line poorly (`rms > 10px`) is
  essentially flat across categories (curve 8.4%, normal 9.7%, night 10.0% on the full test split;
  median rms 1.25–1.35px everywhere). CULane's "curve" tag describes the scene, not the annotated segment near the car.
  The extra error on curves is the model, not the metric.

Worth knowing when interpreting these: the two views disagree in a useful way. A middling IoU of
0.535 sits alongside a median direction error under one degree — the model finds where lanes point
much better than the pixel score suggests, it just places the band slightly off.

`resnet34_imagenet_bs32_ep30_n2000` — trained here on a 2,000-image subset (3.3% of train), 14 min,
best epoch 24:

- test pixel — IoU 0.449 / P 0.607 / R 0.583 / F1 0.597
- test slope — median 1.40°, acc@5° 80.4% (matched), lane recall 0.871
- train F1 reached 0.886 while val F1 flattened near 0.60 from epoch 13 — heavy overfit, as expected

**Do not read the gap against the full-data run as a pure data-size effect.** That baseline was
trained off-machine *without* ImageNet normalization (`normalized=False` in its test CSV) while this
run used it, so data volume, preprocessing, and any unknown differences in the older setup are all
confounded. A clean data-scaling comparison needs a full-data run from this same notebook.
