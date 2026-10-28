# hybrid-drl-route

Code for the paper *Deep Reinforcement Learning for Maritime Route Optimization on AIS–ERA5 Cost
Maps with Navigability Constraints* (B. Danilov, J. Dong, A. Said Ben Salah, P. O. Arowosegbe,
M. A. Mahmud, Ocean University of China).

Archived version and result files: Zenodo (the DOI is added here after the first release).

AIS reports of cargo ships and tankers in the Houston Ship Channel (NOAA Marine Cadastre,
January–June 2024) are matched with ERA5 wind and wave reanalysis. The code then

1. trains an LSTM that predicts wind (u, v) and significant wave height along vessel tracks
   (temporal split: January–April / May / June) and compares it with persistence and the training
   mean; MC Dropout uncertainty and a zero-shot test on Chesapeake Bay and Puget Sound;
2. builds a 60 × 80 cost map of fuel, risk and time over the channel and a navigability mask
   (land is not navigable);
3. trains PPO agents (Stable-Baselines3) in a Gymnasium environment on this map and compares them
   with Dijkstra and A* in four scenarios (Normal, Storm, Dense Traffic, Sparse Traffic;
   10 seeds × 30 evaluation episodes), with an ablation of the cost map and of the weather in the
   observation, and a SAC-Discrete baseline under the same protocol (2×10⁵ instead of 10⁶ steps per
   seed).

## Data

All data are open. ERA5 needs a free account; the key stays on your computer and is never stored
in this repository.

| Data | Source | Used for | Account |
|---|---|---|---|
| AIS, daily files 2024 | NOAA Marine Cadastre, coast.noaa.gov/htdata/CMSP/AISDataHandler | vessel tracks, SOG, cost map, navigable channel | no |
| ERA5 hourly single levels (00, 06, 12, 18 UTC) | Copernicus CDS, doi:10.24381/cds.adbb2d47 | wind u10, v10; wave height swh | CDS key in `~/.cdsapirc` |
| GLOBE 1-km land mask | NOAA NGDC, doi:10.7289/V52R3PMS, via the `global-land-mask` package | navigable mask | no |

Houston: all days of January–June 2024, filtered to vessel types 70–89 (cargo, tanker) and
28.5–30.0° N, 95.6–94.0° W (`filter_ais.py`). The cost map grid covers 29.0–30.2° N,
95.5–94.0° W. Additional regions (Chesapeake Bay, New York Harbor, Mississippi Delta, Puget Sound):
the first day of each month, January–June 2024 (`download_ais_regions.py`).

The raw Houston AIS files are about 60 GB. `merged_data.zip` on Zenodo holds the merged AIS+ERA5
files of all regions and the ERA5 files, so all models can be rerun from step 2 below without any
download.

## Installation

Python 3.10 or newer (the paper used Python 3.13, PyTorch 2.6.0, Stable-Baselines3 2.8.0,
Gymnasium 1.2.3).

    pip install -r requirements.txt

## Running

`run_all.bat` (Windows, Anaconda Prompt) runs everything in order and shows the output of every
step in the console and in `logs\`; the commands also work in any other shell. All data and results
are written into subfolders of the repository folder. Trained models are reused when present, so a
run can be stopped and started again.

    # 1. data
    python download_ais.py --start 2024-01-01 --end 2024-06-30 --out ais_raw
    python filter_ais.py --in-dir ais_raw --out-dir ais_filtered --start 2024-01-01 --end 2024-06-30 --bbox 28.5,-95.6,30.0,-94.0 --vessel-types 70,...,89
    python download_ais_regions.py        # 4 additional regions
    python download_era5_regions.py       # ERA5 for Houston and the 4 regions
    python merge_ais_era5.py              # Houston: ais_filtered + era5 -> ais_merged
    python merge_regions.py               # regions -> ais_merged_<region>
    # 2. LSTM environment predictor
    python lstm_predictor_temporal_split.py   # -> results/lstm_model_temporal.pt, metrics
    python plot_predictions.py
    python mc_dropout_generalization.py
    # 3. cost map (navigable_mask.npy is included; make_navigable_mask.py rebuilds it)
    python cost_map.py
    # 4. PPO, 4 scenarios x 10 seeds
    python train_parallel.py experiments  # training in 4 parallel processes
    python run_experiments.py             # evaluation, statistics
    # 5. ablation, 4 variants x 10 seeds
    python train_parallel.py ablation
    python ablation_study.py
    # 6. SAC-Discrete, 4 scenarios x 10 seeds
    python sac_maritime_baseline.py all
    # 7. figures and tables of the paper, archives for Zenodo
    python make_paper_figures.py
    python pack_results.py --data

Computing time on a desktop PC (i9 CPU, RTX 3060): LSTM about 1.5 h; PPO about 4 h for the 40
seeds (1 million steps each, 4 parallel processes); ablation about 2 h; SAC about 4.5 h (200,000
steps per seed, 4 scenarios in parallel).

## Result files

`paper_results.zip` (Zenodo) holds the results of the runs behind the paper: the LSTM model,
scalers and metrics; the PPO models (best checkpoints) and training logs of all seeds, scenarios
and ablation variants; the raw per-episode evaluation results; the SAC results; and the figures
and tables of the paper. Unzip it into this folder to redraw the figures
(`make_paper_figures.py`) or rerun the evaluations without training. `pack_results.py` builds the
same archives from a recomputation with this code.

## Where the numbers of the paper come from

| Paper | Script | File |
|---|---|---|
| Fig. 1, Table 3: LSTM accuracy, comparison with persistence | `lstm_predictor_temporal_split.py` | `results/metrics_lstm_temporal.csv`, `results/metrics_baselines_temporal.csv`, `results/figures/fig_loss_curve_temporal.png` |
| Figs. 2–3: LSTM predictions, error by horizon | `plot_predictions.py` | `results/figures/fig_predictions_v2.png`, `fig_error_by_horizon.png` |
| Figs. 10–12: MC Dropout, cross-region test | `mc_dropout_generalization.py` | `results/figures/ieee/fig_mc_dropout`, `fig_uncertainty`, `fig_generalization`; table printed to the console |
| Figs. 4–5: cost map, navigable mask | `cost_map.py`, `make_navigable_mask.py`, `make_paper_figures.py` | `results/cost_map_stats.csv`, `results/figures/paper/fig_cost_map`, `fig_cost_components` |
| Fig. 6: PPO training curves | `train_parallel.py`, `make_paper_figures.py` | `results/figures/paper/fig_training_curves` |
| Figs. 7–8, Tables 4–5: PPO vs Dijkstra, 4 scenarios | `run_experiments.py`, `make_paper_figures.py` | `results/experiments/summary_pathcost.csv`, `results/figures/paper/table_scenarios.csv`, `fig_scenarios`, `fig_routes` |
| Fig. 9, Table 6: ablation | `ablation_study.py`, `make_paper_figures.py` | `results/ablation/ablation_pathcost.csv`, `results/figures/paper/table_ablation*.csv`, `fig_ablation` |
| Fig. 13, Table 5: SAC-Discrete vs PPO | `sac_maritime_baseline.py` | `sac_results/sac_vs_ppo.csv`, `fig_sac_vs_ppo`; per-seed results in `sac_results/<scenario>/seed*.json` |

## Layout

    download_ais.py                   daily AIS files of NOAA Marine Cadastre (Houston)
    filter_ais.py                     bbox and vessel type filter, CSV -> parquet
    download_ais_regions.py           AIS of the 4 additional regions
    download_era5_regions.py          ERA5 wind and waves for Houston and the 4 regions
    merge_ais_era5.py                 ERA5 values at every Houston AIS report
    merge_regions.py                  the same for the 4 regions
    lstm_predictor_temporal_split.py  LSTM environment predictor, temporal split, baselines
    plot_predictions.py               LSTM prediction figures
    mc_dropout_generalization.py      MC Dropout uncertainty and cross-region test of the LSTM
    cost_map.py                       cost map statistics and component figures (without the mask)
    make_navigable_mask.py            navigable_mask.npy: GLOBE land mask + AIS channel
    run_experiments.py                cost map, environment, PPO, Dijkstra/A*, evaluation, statistics
    train_parallel.py                 parallel training of the PPO seeds (scenarios, ablation)
    ablation_study.py                 PPO with no map, traffic-only map, full map, full map + weather
    sac_maritime_baseline.py          SAC-Discrete in the same environment and protocol
    make_paper_figures.py             figures and tables of the paper
    pack_results.py                   result archives for Zenodo
    tee_run.py                        shows a command's output in the console and writes it to a log

## Method notes

- **Environment.** The agent sails from the Port of Houston (29.73° N, 95.27° W) to Bolivar Roads
  (29.31° N, 94.79° W) on the 60 × 80 grid (cell ≈ 1.2 nm). Action: keep course, turn ±30°, slow
  down or speed up by 1 kn (4–18 kn); step 0.1 h. Observation (18 values): position, speed, course,
  wind speed, wave height and mean AIS speed in the current cell, distance and bearing to the goal,
  and the 3 × 3 patch of the cost map (−1 for land). Reward: −10 × path-integrated cost of the step
  + progress towards the goal (nm) + 100 on arrival (within 1.5 nm); a move into land is blocked
  (the vessel stays, −5); leaving the grid ends the episode (−50); at most 300 steps.
- **Navigability.** A cell is navigable if less than half of it is land in the 1-km GLOBE land
  mask or if it contains AIS positions or touches such a cell (the Houston Ship Channel is narrower
  than the land mask); only the water body connected to the start and the goal is kept.
  Dijkstra and A* do not enter land; land cells cost 1.0 in the metric. The straight line from the
  start to the goal crosses land and is reported only as a reference.
- **Cost map.** Fuel (wind speed²), risk (logistic in wave height) and time (12 kn / mean AIS SOG)
  terms, weights 0.4/0.4/0.2, from ERA5 values at the AIS positions of the first 80 days, averaged
  on the grid and smoothed (σ = 1.5 cells); fixed during training and evaluation. The scenarios
  scale the wave height (Storm, ×2.5) or the SOG (Dense ×2, Sparse ×0.3). Cells without AIS
  positions have no ERA5 values (zero wind and waves, maximum time cost). The LSTM is not called by
  the RL scripts.
- **Metric.** Path-integrated cost: cell cost × distance along the route (cost·nm), sampled every
  0.05 nm, the same for PPO, SAC and the grid routes. An unfinished PPO/SAC route is completed by a
  straight segment to the goal.
- **Protocol.** 10 seeds × 1,000,000 steps per scenario; the checkpoint with the best deterministic
  return on 5 separate start positions (evaluated every 20,000 steps) is kept; final evaluation on
  30 episodes per seed. The unit of statistical analysis is the seed (mean of its 30 episodes):
  mean with 95% t-interval, median, interquartile mean, Wilcoxon signed-rank and Mann–Whitney tests.
- **SAC-Discrete.** Same environment, observation, checkpoint selection and evaluation as PPO, but
  200,000 steps per seed (it converges within this budget). Twin critics with layer normalization,
  fixed entropy coefficient α = 0.05, replay buffer 200,000, batch 256, 10,000 random steps at the
  start, one gradient update every 4 steps. The seeds of SAC and PPO are independent samples, so the
  paper compares them with the Mann–Whitney test; the Wilcoxon column of `sac_vs_ppo.csv` pairs the
  seeds by their number and is not used.
- **LSTM.** Sequences are cut per vessel from consecutive AIS reports (12 input and 6 target
  reports; the median reporting interval is about 3 min). Every AIS report gets the ERA5 values of
  the nearest ERA5 time step (6-hourly files) and grid node, so the targets change only every 6 h
  along a track; persistence is therefore a strong baseline and is better than the LSTM in MAE.
- `mc_dropout_generalization.py` does not fix the random seed of the dropout masks, so a rerun
  changes the MC Dropout numbers in the third decimal; the paper reports the run whose figures are
  in `paper_results.zip`.
- Eight days (24, 25, 28, 29 May and 10, 15, 19, 20 June) are missing from the downloaded AIS data.
- PPO training on the CPU with fixed seeds is reproducible (variant V4 of the ablation repeats the
  Normal scenario seed by seed). LSTM training on a GPU is not bit-exact.
- `download_era5_regions.py` disables TLS certificate verification for the CDS requests (this was
  needed on the authors' university network).

## Licence

Code: MIT, see `LICENSE`. The data keep their own licences (NOAA Marine Cadastre AIS: public
domain; ERA5: Copernicus licence; GLOBE: NOAA). Please cite the paper and the data sets if you use
the code (`CITATION.cff`).
