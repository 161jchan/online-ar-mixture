# Online mixture of autoregressive experts

Jenny's data, AR experts and FPE handoff for ECE 685D Project 1. The directions PDF is authoritative; the group guide assigns Tasnuba the gate and Kurt RLS/comparisons. Their methods are not implemented here. The report uses the supplied ICLR 2024 LaTeX template and its nine-page main-text limit.

## Reproduce

Run from the repository root after installing requirements.txt:

~~~powershell
python -m unittest discover -s tests -v
python -m unittest discover -s tests -p test_multichannel.py -v
python -m unittest discover -s tests -k LevinsonTests -v
python -m data.real --download
python -m experiments.experts_fpe synthetic --variants
python -m experiments.experts_fpe real
python -m experiments.experts_fpe aggregate
python -m experiments.experts_fpe plots
~~~

Downloads occur only when a cache is absent. Cache hits verify SHA-256 and use no network. Existing run summaries skip completed experiments. To independently recompute into a separate destination, append --output outputs/reproduction to the synthetic, real, aggregate and plots commands. A reproduction is an audit of frozen settings, not a new opportunity to tune held-out data.

The plots command reads saved forecasts and redraws every new PDF without refitting or examining real targets to choose settings. It creates expert_fpe.pdf for all 23 main streams; stationary coefficient_paths.pdf, coefficients_by_order.pdf, covariance_paths.pdf and covariance_by_order.pdf for n=1,3 and all three seeds; and comparisons.pdf. Coefficient figures compare raw-unit estimates with generating coefficients. Covariance entries and trace(Sigma)/n summaries are labeled. Temporal SD describes fluctuations, not confidence intervals.

Original data-overview figures: python -m experiments.generate_synthetic (recreates the original seeded datasets). Original scalar diagnostics: python -m experiments.stationary (now reads saved observations). These reproduce all PDFs under outputs/synthetic_data and outputs/stationary, respectively.

## Teammate contract

~~~python
from experts import ExpertConfig, OnlineARExperts
bank = OnlineARExperts(ExpertConfig()).initialize(values[:40])
forecasts_z = bank.predict()               # (L,n), before target arrives
before = bank.snapshot()                  # copied pre-target state
target_z = bank.standardize(values[40])    # score saved forecasts
bank.update(values[40])                   # refit for the next target
~~~

Inputs are raw observations (M,n); initialization consumes exactly 5L rows. Warm-up per-channel mean and population SD are frozen; constant channels use scale 1. No generating coefficients enter the forecasting loop. Legacy scalar coefficients/covariances retain their old shapes; explicitly named coefficient arrays use (order,lag,output,input).

forecasts.npz contains pre-target expert forecasts (time,L,n), error covariances (time,L,n,n), targets, FPE forecasts/orders/log-scores/fallback flags, fresh/stale diagnostics, observed/scoring masks, target indices and normalization. Raw and standardized fields are named explicitly. Coefficient paths are saved every 10 targets (n<=3), or 25 (ETT), with their own target indices; covariances/forecasts are saved every target. Load with numpy.load(...), without pickle. Loss is squared Euclidean error summed across channels; divide by n for per-channel MSE. Gate/comparison owners should use standardized forecasts/targets consistently.

For x=mu+D z, A_raw[k]=D A_z[k] D^-1, Sigma_raw=D Sigma_z D, and the raw intercept is mu-sum(A_raw[k])mu. Thus multichannel standardized coefficients cannot be compared directly with raw generating matrices.

## Recursion and verification

Use R[k]=E[z_t z_(t-k)^T] and newest-first lag vectors. In the dense check, block T[i,j] is R[j-i] for j>=i and R[i-j]^T otherwise; solve A T=[R[1] ... R[p]] independently with numpy.linalg.solve.

At order p, the forward/backward recursion uses preceding coefficients A,B and error covariances F,G:

- Delta=R[p]-sum_j A[j] R[p-j], for j=1..p-1.
- Kf=Delta G^-1; Kb=Delta^T F^-1.
- Anew[j]=A[j]-Kf B[p-j]; Bnew[j]=B[j]-Kb A[p-j]; append Kf,Kb.
- Fnew=F-Kf Delta^T; Gnew=G-Kb Delta.

All updates use unchanged preceding-order arrays. This is one O(L^2 n^3) order recursion, not repeated block fitting. Regularize only a working copy: R[0]+=epsilon*trace(R[0])/n I, epsilon=1e-6. Both updated covariances must be positive definite above 1e-12; the largest singular value of Cholesky-whitened Delta must be below 1-1e-10. Whitening gives the assignment's normalized singular values; raw Kf norm may exceed one. A failed order and every higher order retain their previous estimates, with stale counters. Never-successful orders start at zero coefficients and identity covariance.

Tests obtain deterministic population covariances from an independent companion Lyapunov equation, check scalar reduction, nonsymmetric lag covariances, cross-channel lag orientation, seven channels, valid raw reflections above one, failure retention, copied snapshots and causality. Every fresh order at every main-run pre-target state is also checked against the independent dense solve.

Inspected derivation: [Scherrer and Funovits, RLDM technical notes, Durbin-Levinson-Whittle section](https://berndfunovits.com/RLDM/articles/b_RLDM_technicaldetails.html). Also inspected: [Barnett/Seth MVGC solver documentation](https://users.sussex.ac.uk/~lionelb/MVGC/html/autocov_to_var.html). Original Whittle (1963) and Akaike (1969) full texts were inaccessible in this session; Wiggins-Robinson (1965) was not accessed. FPE follows the assignment's explicit formula. **Human multichannel derivation/code review remains PENDING.**

## FPE and experimental protocol

FPE(p)=det(Sigma_p)*((Neff+p*n+1)/(Neff-p*n-1))^n, Neff=1/(1-lambda). Evaluate in log space. Exclude stale, asymmetric, nonfinite, non-PD covariances and nonpositive denominators (including roundoff-zero). Ties choose the smallest order. If all are excluded, retain the previous selected order (initially 1), flag fallback, and forecast with its retained coefficients.

All 21 existing 10,000-row synthetic datasets are reused, seeds 685,686,687; main L=8, lambda=.99 was fixed in advance. L=4,8,12 and lambda=.98,.99,.995 are compared on stationary and T=500 data for n=1,3. L and lambda vary separately; no gate ablations. Each L uses its required 5L warm-up. Comparisons use common targets 60..9999 and raw loss, with sample SD across three seeds; aggregate removes the short earlier prefix by deterministic replay. Main metrics retain the assignment's full post-warm-up interval.

For each real series, select lambda from the same three-value grid using only first-20% prequential standardized FPE loss; write frozen_protocol.json before the full pass. L=8 stays fixed. The remaining 80% is evaluated once with ongoing causal updates. Held-out metrics are separate from descriptive full-stream metrics. Experts/FPE are deterministic, so real-stream repetitions under different gate seeds would be identical.

ETTh1: [ETDataset, Zhou et al., Informer (AAAI 2021)](https://github.com/zhouhaoyi/ETDataset), all seven numeric columns in source order; date is excluded from model inputs. Hourly timestamps are timezone-naive source labels. Sunspots: [WDC-SILSO, Royal Observatory of Belgium, Brussels](https://www.sidc.be/SILSO/datafiles), monthly V2.0 column 4, [DOI](https://doi.org/10.24414/qnza-ac80), [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Monthly timestamps label the first of the month. These are frozen historical vintages, not an as-published historical-vintage backtest.

Both loaders reject duplicate/out-of-order timestamps, insert missing grid rows, and forward-fill each channel; leading missing cells use zero. Incomplete targets are excluded from losses. SILSO -1 is missing. Its documented future-interpolated February 1824 is also masked and past-filled. Cached source bytes remain unmodified; manifests record URLs, retrieval times, version/credit and hashes.

## Observed results

All 43 correctness checks passed. At L=8 and lambda=.99, stationary FPE raw loss was 1.04270 +/- 0.01826 for n=1 and 3.34593 +/- 0.13200 for n=3 (three-seed mean +/- sample SD; vector loss sums channels). Population tests recover exact coefficients and the AR(4) covariance plateau; finite-stream estimates fluctuate under fixed forgetting, and additional orders can lower fitted covariance while worsening prediction.

Held-out standardized FPE loss was 9.83540 on ETTh1 (lambda=.98, n=7) and .37224 on sunspots (lambda=.995, n=1). ETTh1 used the explicitly flagged FPE fallback on 8,215 of 13,936 held-out targets; sunspots used it on none of 2,667 observed held-out targets. These ETTh1 results include retained fits and must be interpreted with the guard/stale diagnostics. Human multichannel review remains pending.

## Files

- data/synthetic.py: Original seeded generators; observations and truth are separate.
- data/real.py: Causal parsing, cached downloads and source verification.
- experts/scalar.py: Scalar Levinson-Durbin and independent direct solve.
- experts/multichannel.py: Forward/backward Whittle recursion and block direct solve.
- experts/online.py: Causal normalized scalar/vector expert bank and copied handoff.
- experts/fpe.py: Guarded log-FPE order selection with explicit fallback.
- experiments/experts_fpe.py: Saved-data runs, tuning freeze, verification and PDF figures.
- experiments/stationary.py: Original scalar diagnostics, reading saved data.
- experiments/generate_synthetic.py: Original synthetic datasets/data-overview figures.
- experiments/gate_example.py: Forecast/score/update hooks for Tasnuba.
- tests/test_pipeline.py and tests/test_synthetic.py: Original scalar/data checks.
- tests/test_multichannel.py: Independent population, matrix and online checks.
- tests/test_fpe_real.py: FPE boundaries, missing cells, timestamp/cache checks.
- tests/test_experiments.py: Saved handoff causality and FPE alignment.
- outputs/experts_fpe: Main forecasts, variant summaries, PDFs, aggregate and verification.
