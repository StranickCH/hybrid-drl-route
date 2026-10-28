@echo off
rem Full recomputation of the paper, in order. Run from the Anaconda Prompt in this folder.
rem All data and results are written into subfolders of this folder (see README.md); the output of
rem every step is shown here and written to logs\. Downloads and merges skip files that already
rem exist; trained models are reused when present, so the script can be stopped and started again.
rem Time on a desktop PC (i9 CPU, RTX 3060): AIS download ~60 GB (many hours), ERA5 a few hours,
rem LSTM ~1.5 h, PPO 4 scenarios x 10 seeds ~4 h, ablation ~2 h, SAC ~4 h.
rem Before the ERA5 step: create a free CDS account and put your API key into %USERPROFILE%\.cdsapirc
set PYTHONUTF8=1
cd /d "%~dp0"
if not exist logs mkdir logs

rem ---- 1. data
python -u tee_run.py logs\01_download_ais.log python -u download_ais.py --start 2024-01-01 --end 2024-06-30 --out ais_raw || goto fail
python -u tee_run.py logs\02_filter_ais.log python -u filter_ais.py --in-dir ais_raw --out-dir ais_filtered --start 2024-01-01 --end 2024-06-30 --bbox 28.5,-95.6,30.0,-94.0 --vessel-types 70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89 || goto fail
python -u tee_run.py logs\03_download_ais_regions.log python -u download_ais_regions.py || goto fail
python -u tee_run.py logs\04_download_era5.log python -u download_era5_regions.py || goto fail
python -u tee_run.py logs\05_merge.log python -u merge_ais_era5.py || goto fail
python -u tee_run.py logs\06_merge_regions.log python -u merge_regions.py || goto fail

rem ---- 2. LSTM environment predictor (temporal split), MC Dropout, cross-region test
python -u tee_run.py logs\07_lstm.log python -u lstm_predictor_temporal_split.py || goto fail
python -u tee_run.py logs\08_plot_predictions.log python -u plot_predictions.py || goto fail
python -u tee_run.py logs\09_mc_dropout.log python -u mc_dropout_generalization.py || goto fail

rem ---- 3. cost map; navigable_mask.npy is in the repository (make_navigable_mask.py rebuilds it)
python -u tee_run.py logs\10_cost_map.log python -u cost_map.py || goto fail

rem ---- 4. PPO: 4 scenarios x 10 seeds (parallel training), evaluation and statistics
python -u tee_run.py logs\11_train_ppo.log python -u train_parallel.py experiments || goto fail
python -u tee_run.py logs\12_run_experiments.log python -u run_experiments.py || goto fail

rem ---- 5. ablation: 4 variants x 10 seeds
python -u tee_run.py logs\13_train_ablation.log python -u train_parallel.py ablation || goto fail
python -u tee_run.py logs\14_ablation.log python -u ablation_study.py || goto fail

rem ---- 6. SAC-Discrete baseline: 4 scenarios x 10 seeds
python -u tee_run.py logs\15_sac.log python -u sac_maritime_baseline.py all || goto fail

rem ---- 7. figures and tables of the paper, archives for Zenodo
python -u tee_run.py logs\16_figures.log python -u make_paper_figures.py || goto fail
python -u tee_run.py logs\17_pack.log python -u pack_results.py --data || goto fail
echo Done. Results: results\ and sac_results\
pause
exit /b 0
:fail
echo Stopped with an error at the step above (see logs\).
pause
exit /b 1
