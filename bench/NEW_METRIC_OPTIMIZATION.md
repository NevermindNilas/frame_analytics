# New metric optimization, 2026-09-23

## Scope and measurement

All 15 metric modules added for 0.7.0 were inspected and profiled in four
nonoverlapping groups. The exact dirty-tree starting modules were saved before
edits. Timed work was serialized so CPU and GPU benchmarks did not contend with
another group. Rejected changes were removed, and SHA-256 checks confirmed that
the other 12 source files match their starting bytes.

Measurements were warm per-call latency on Windows 11, Python 3.14.6,
PyTorch 2.13.0+cu132, and an RTX 3090. Each group's harness fixes its input
shape, data type, device, warmup, varying seeded fixtures, and paired sample
order. CUDA calls are synchronized. The figures below apply to these workloads;
they are not claims for every image size or device. CPU and CUDA timing numbers
from different groups should not be compared directly because their workloads
and harness boundaries differ.

## Kept changes

| Metric and path | Warm baseline → final | Held-out result | Guardrails and verdict |
| --- | --- | --- | --- |
| ADM-like `wavelet="db2_like"`, CPU, 2×1×256×256 | 4.215 → 3.092 ms, paired latency ratio 0.731 (95% bootstrap CI 0.721–0.739) | CPU 1×3×128×160: 3.896 → 2.705 ms, ratio 0.699 (0.681–0.710) | Kept. Reference and distorted images share each Db2 batch; default Haar remains original. CPU/CUDA differential and input-gradient comparisons passed. |
| ADM-like Db2, CUDA, same primary shape | 1.315 → 1.089 ms, ratio 0.863 (0.792–1.038) | 1.215 → 0.972 ms, ratio 0.808 (0.772–0.905) | CUDA primary CI includes no gain. Incremental peak allocation rose 2.37 → 4.22 MB (+1.85 MB). No numeric VRAM limit had been frozen for this group. |
| FSIM, CUDA, 1×3×256×256 | 6.025 → 4.259 ms, 29.3% paired improvement (95% CI 22.6–31.3%) | 2×3×384×320: 8.107 → 7.448 ms, 8.1% (4.3–11.3%) | Kept CUDA-only phase/amplitude scale grouping. A compiled-path check was 5.525 → 3.804 ms. CPU uses original code. Peak CUDA allocation rose 29.89 → 30.94 MB primary and 111.03 → 114.96 MB held-out. Eighty-eight differential cases included gradients; maximum reported score/gradient difference was 1.9e-9. |
| FLIP, CUDA, 1×3×64×64 | 2.184 → 1.844 ms, 15.6% paired improvement (95% CI 4.4–26.1%) | 1×3×128×128: 2.457 → 2.257 ms; paired improvement 10.3%, CI −20.4–21.2% | Kept CUDA-only paired feature filtering. The held-out result is noisy and its CI includes a regression. CPU uses original code. Peak extra CUDA allocation rose about 0.06 MiB primary and 0.23 MiB held-out (~8.3%). Differential checks, including maps, options, and input gradients, were exact in the tested cases. |

## Other metric verdicts

| Metric module | Measured candidate and verdict |
| --- | --- |
| `ms_gmsd` | Batched pooling gave exploratory primary gains (1.849 → 1.742 ms CPU; 0.730 → 0.645 ms CUDA), but peak CUDA allocation rose 27% and held-out CUDA timing was unstable. Reverted. |
| `vifp` | Batched channel epilogue slowed the primary workload (12.275 → 13.109 ms CPU; 4.290 → 7.291 ms CUDA). Reverted. |
| `iwssim` | Batched pyramid was initially promising, but fresh confirmation gave only a 1.9% CPU gain, a 3.3% CUDA regression, and an 8% held-out CUDA regression. Reverted. |
| `nlpd` | CPU channels-last layout did not improve the primary workload (7.270 → 7.327 ms CPU; 2.064 → 2.275 ms CUDA). Reverted. |
| `dss` | Batched local-statistic convolution had less than a practical primary gain and regressed CUDA held-out work. Reverted. |
| `haarpsi` | A single 2-D Haar filter bank slowed CPU primary work by about 10%. Reverted. |
| `psnrhvs` | Batched quadrant reductions for PSNR-HVS-M showed no stable gain. Reverted; PSNR-HVS proper remained unchanged. |
| `srsim` | Complex normalization in place of angle/polar had inconclusive or regressive timing. Reverted. |
| `vsi` | Combined extrema reductions had inconclusive or regressive timing. Reverted. |
| `mdsi` | Reusing directional gradient responses had inconclusive or regressive timing. Reverted. |
| `ciede2000` | Stacked RGB conversion helped eager execution but regressed the default compiled CPU path by about 26%. Reverted. |
| `scielab` | Matrix precomposition and batched Lab conversion gave inconclusive or regressive default compiled results. Reverted. |

An FSIM median-to-`kthvalue` candidate was also reverted: tied values produced
equal scores but different input gradients. No safe measurable improvement was
found for the 12 unchanged modules under these workloads and guardrails.

## Validation and artifacts

- `python bench/check_transform_candidates.py`: passed on CPU and CUDA,
  including ADM Db2 input gradients with fresh weight caches. The checker's
  first run reused inference-mode cached weights for autograd and failed in
  both versions; the harness now clears those caches before the gradient case.
- `python -m pytest -q`: 385 passed, 7 skipped, 2 failed. The same two
  VapourSynth tests failed before optimization because they still expect the
  newly enabled feature IDs 1 and 4 to be unavailable. No new failure appeared.
- `git diff --check` and `python -m compileall -q frame_analytics`: passed.
- Baseline and paired raw results: multiscale under
  `%TEMP%/fa_multiscale_logs_20260923`, transform under
  `%TEMP%/frame_analytics_transform_logs_20260923`, feature/saliency under
  `build/optimization/features`, color under
  `bench/results/color_opt_20260923`.
- Reproducible harnesses: `bench/bench_new_multiscale.py`,
  `bench/bench_transform_candidates.py`, `bench/check_transform_candidates.py`,
  and `bench/bench_color_candidates.py`. The feature harness is
  `build/optimization/features/measure.py` (ignored build artifact).

The first multiscale timing run imported an older installed package because
its direct script invocation did not prioritize the repository. It was
discarded; both arms were rerun with `python -m bench.bench_new_multiscale`.
The first transform single-call CUDA timings were discarded for excessive
scheduling noise; both arms were rerun with a fixed 20-call paired protocol.

The optimization preserved an ADM lifecycle issue: populating its filter cache
inside `torch.inference_mode()` could break a later autograd call in the same
process. A subsequent correctness pass fixed cache construction across the new
metrics and added inference-then-training regression tests. That pass also
corrected FSIM's directional filters and noise statistics. The measurements
above describe the optimization before those correctness changes; they do not
benchmark the corrected implementations.
