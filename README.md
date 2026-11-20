# Ten of 40 neurons firing together

[Read the illustrated article on GitHub Pages](https://fredluv.github.io/retina-synchrony-reproduction/).

A partial reproduction of Figure 5B in Tkačik, Marre, Amodei et al. (2014). The [executed notebook](reproduction.ipynb) contains the calculations and three charts. It uses the authors’ [released recording](https://research-explorer.ista.ac.at/record/5562) of 160 salamander retinal cells and the [original paper](https://doi.org/10.1371/journal.pcbi.1003408).

Across 10,000 overlapping groups of 40 cells in this retina, bursts with **exactly ten active cells** occurred a median **1,463×** as often as average-rate independence predicted. **64 groups** reached 10,000×. Whether the authors measured 10,000× for the particular 40 cells shown in their figure is unknown. [All group results](results/group-sensitivity.csv.gz) · [Held-out comparison](results/endpoint.csv)

## Run the analysis

The numerical analysis was run with Python 3.11 on macOS ARM64. A C++17 compiler is needed to refit or check the replica-exchange sampler on macOS or Linux. Reading the notebook outputs and recalculating the independent forecasts use the supplied data and results.

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python code/reproduce.py --mode all --jobs 2
python code/figure5b.py
python code/summarize.py
python code/group_sensitivity.py
python code/figure_discrepancy.py
python code/plot_group_sensitivity.py
```

To open the notebook interactively, install a Jupyter front end in the same environment, then run `python -m jupyterlab reproduction.ipynb` from this directory. The notebook recomputes both independent forecasts for the first group with convolution, checks them against a separate coefficient-recursion implementation, and redraws the figures. Its `REFIT` switch reruns all five pairwise fits. For just the primary model, run `python code/reproduce.py --mode pairwise --groups 1 --jobs 1`. To re-extract the published central curves from the authors’ source bundle, run `python code/digitize_reference.py` before `python code/figure5b.py`. The two-PDF check downloads the journal and author source files on request. Its `--art-pdf` and `--journal-pdf` options accept local copies.

## What was fixed in advance

A local [protocol file](protocol.json) lists 30 groups of 40 cells, the primary group, the five fit groups, the held-out repeats, and the endpoint. The protocol SHA-256 is `e6c442f56f95e1913cb05b204ab70ed5ba549d0b8da6ca5e0dea186eeef2b87b`. The file records the prospective choices for this analysis. It has no external registration timestamp.

All 953 bins of each repeat stay together. The zero-based held-out repeats are 2, 5, 8, …, 296. This leaves 188,694 training windows and 94,347 test windows. Each window lasts 20 ms. A cell is active if it emitted at least one spike. A burst in this analysis always means exactly ten active cells in one window. Overlapping windows are avoided by using the archive’s original bins. Consecutive bins need not be statistically independent.

## Model and uncertainty

- **Stationary independence.** A Poisson-binomial calculation uses the 40 individual training rates. No identical-rate assumption is made. A single binomial calculation applies when all 40 rates are equal.
- **Pairwise maximum entropy.** A model with 40 fields and 780 symmetric pair couplings is fitted to the pooled training single-cell and pair moments. A shared-parameter pseudolikelihood fit initializes a regularized likelihood refinement. A scalar line search changes every coupling by the same amount to match the *sum of the 780 training pair moments*. This is a linear combination of the original sufficient statistics. It introduces no count-specific model term. Seven Monte Carlo reweighting stages refine the same parameters. The objective contains weak Gaussian penalties. The original paper used its own fitting code and regularization.
- **Numerical validation.** A first short fit with the pseudolikelihood initialization and four refinements failed its training moment checks because it put excessive weight on dense states. Its predictions were discarded. The final fits use the training-only calibration above. Every fit is checked against its defining training moments. Each final simulation uses eight temperatures. Four independent runs alternate sparse and dense starting states and retain 250,000 cold sweeps per run. The reported MC standard error is the larger of the between-run and 5,000-sweep block estimates. It conditions on the fitted parameters. The [eight-cell validation](results/sampler-check.json) compares the same sampler against exact enumeration of all 256 states. Final JSONs give each chain’s event rate, swap acceptance, moment residuals and refinement diagnostics. Weights and moments for each fit are supplied as NumPy archives.
- **Conditional independence given movie time.** For each of the 953 movie windows, estimate 40 single-cell rates across the 198 training repeats. Compute its independent ten-cell probability, then average the 953 probabilities. This control has 38,120 rate estimates per group, can use the precise time in a repeated movie, and assumes no residual coupling given that time. Its predictive accuracy addresses a specific count in this repeated stimulus.
- **Test uncertainty.** Percentile 95% intervals for observed frequencies resample whole held-out repeats 2,000 times. The 30 groups overlap and all share the same recording. They are sensitivity analyses across cell selections. In the notebook, multiplicative error is defined as the ratio of the larger of predicted and observed rates to the smaller. Absolute log error is its natural logarithm.

## Reproduction audit

**Whether the authors measured 10,000× for the 40 cells in Figure 5B is unknown.** Their text reports approximately that ratio. The panel plots about 1.29 observed ten-cell bursts and a pairwise prediction of 2.15 per 1,000 windows. The gray independent curve crosses the lower plotting limit at 9.82 cells. Extending its last segment the remaining 0.18 cell on the log scale would give 0.000695 per 1,000 windows and a ratio near 1,860×. For a 10,000× ratio, the gray value would have to be 0.000129 per 1,000. A straight segment to that value would cross the plotting limit at 9.45 cells. The published PDF and the author artwork yield the same coordinates to the displayed precision. Both reproduce the same drawing. The hidden point and the identities of the 40 cells remain unknown. [Numerical readings and assumptions](results/figure-discrepancy.json) · [Published PDF, p. 7](https://journals.plos.org/ploscompbiol/article/file?id=10.1371/journal.pcbi.1003408&type=printable#page=7) · [Authors’ artwork](https://arxiv.org/abs/1306.3061)

A post-hoc sensitivity check used 10,000 uniform random groups of 40 cells from the 160 recorded. A second [protocol](group-sensitivity-plan.json) fixed that sample design before scoring these groups. Using the full recording and each cell’s own firing rate, the median observed-to-independent ratio was 1,463× (middle half 943–2,342×). **64 of 10,000 groups (0.64%)** reached 10,000×. On our 99 held-out repeats the median was 1,430× and 55 groups reached 10,000×. These results show that selecting different cells is sufficient to produce both scales in this recording. The groups overlap, and the sample proportions describe this retina. Our pairwise solver is absent from this ratio. The paper uses “approximately”; the 0.64% uses the explicit threshold of at least 10,000. [All group IDs, counts and ratios](results/group-sensitivity.csv.gz) · [Summary](results/group-sensitivity.json) · [Population selection chart](figures/group-sensitivity.png)

This is a partial reanalysis of the 2014 paper’s Figure 5B endpoint. Its redrawn values (approximately 2.2 and 1.3 per 1,000) come from the vector figure. The text of the paper provides the roughly 10,000× comparison. The fresh fits use separately selected groups from the same recording, and a different numerical fitter. In all 30 groups the stationary independence model underestimates ten-cell bursts. The median ratio is 1,373× (range 686–5,622×). The five pairwise models overpredict by 1.33–5.41×. The published K-pairwise and larger-network analyses fall outside this reanalysis. The movie-timed comparison is an additional control. Repeating our computation exactly verifies our own pipeline.

The paper’s model-validation split used 277 training and 20 test repeats. A **post-hoc** check holds the original 30 groups fixed and recomputes the independent model using 100 splits of that size. The median of the 30-group ratios ranges from 1,124× to 1,748× (median over splits 1,417×). These are descriptive split-to-split values. Their constituent groups and trial partitions overlap. The authors’ chosen 40 cells and exact validation repeat indices are unavailable in the materials we checked. Run `python code/check_repetition_split.py` for the [split check](results/split-check.json). [Paper and methods](https://doi.org/10.1371/journal.pcbi.1003408)

An additional archive-wide check finds 1,622,436 active cell-windows out of 45,286,560 (3.58%). The paper describes the average as about 3.1%. The cause of this difference is unresolved. The ISTA MATLAB and text downloads contain identical binary values in all 160 rows after aligning dimensions. The MATLAB file also matches the independent Figshare download byte for byte, including Figshare’s own recorded MD5, `ecce389eba7c20acc878e5d680de15ad`. ISTA’s catalogue displays different MD5 strings for its downloads. [ISTA dataset](https://research-explorer.ista.ac.at/record/5562) · [Second public deposit](https://doi.org/10.6084/m9.figshare.5009840.v2)

## Data and attribution

The included `data/IST-2017-61-v1+1_bint_fishmovie32_100.mat` is a byte-for-byte copy of the 2017 public deposit from Marre, Tkačik, Amodei, Schneidman, Bialek and Berry. It is distributed under CC0 1.0. Its SHA-256 is `50fa85a05950b76033809c8468f70c91fcd0c76d04587a685c1b76085f7f4dae`. The script validates it before analysis. If the bundled file is absent it downloads the original and validates the same hash. [Data and metadata](https://research-explorer.ista.ac.at/record/5562) · [Direct MAT file](https://research-explorer.ista.ac.at/download/5562/5622/IST-2017-61-v1%2B1_bint_fishmovie32_100.mat) · [CC0](https://creativecommons.org/publicdomain/zero/1.0/)

The first chart adapts central curves from Tkačik et al., *PLOS Computational Biology* 10, e1003408 (2014), Figure 5B, under CC BY 4.0. Color, layout and units changed. The displayed range is 0–14 active cells. The original error bars are omitted. At ten cells, the independent curve lies below the lower limit of the published panel. The figure’s vector artwork supplies the pairwise and observed values. The approximate 10,000× comparison comes from the paper’s text. The original 40 cell IDs are unspecified in the publication and the public deposit. [Journal article](https://doi.org/10.1371/journal.pcbi.1003408) · [Figure 5](https://doi.org/10.1371/journal.pcbi.1003408.g005) · [Original vector source](https://arxiv.org/src/1306.3061v1) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

The research behind the 2014 paper followed Amodei’s 2011 Princeton dissertation. [Actual dissertation](https://www.proquest.com/openview/1c37e3a50c16b9187eec9792140f3a17/1)

The charts use unmodified Adobe Source Sans 3 font files. The SIL Open Font License is included under `code/fonts/OFL.txt`. [Font project](https://github.com/adobe-fonts/source-sans)
