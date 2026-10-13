# Heat Resilience in Sight

Climate-conditioned thermal affordance: treat Thermal Comfort in Sight (TCIS) VATA as streetscape **supply**, city-month UTCI as **demand**, and report streetscape heat resilience (SHR).

## Layout

- `data/svi_vata/` — eight-city aligned VATA CSVs (image-level, about 1.3 GB, derived from Google Street View; **not included in this repository**). File key `svi_jh` is Johannesburg (coordinates in the CSV), not Johor Bahru. Singapore here is the 21,772-image panel, not the 92,233-image TCIS survey-design set.
- `data/svi_vata_distributions/` — leftover exploratory histograms; paper figures are written by Code 4.
- `data/climate_cache/hourly_v1/` — Open-Meteo ERA5-land hourly series, 2015–2024 (`hris_cache=hourly_v1`). Old daily JSON is never reused.
- `data/outputs/` — city VATA stats, monthly/annual SHR, calibration constants, robustness and domain-shift tables.
- `figures/` — `fig_framework`, `fig_mapping`, Code 3 climate figures, Code 4 `fig_vata_distributions` / `fig_vata_maps`, and `fig_robustness`.
- `paper/` — workshop tex/pdf (local draft; gitignored). Compile from that folder (`latexmk -pdf paper.tex`).
- `scripts/` — shared Python used by the notebooks.
- `code_1_eight_city_vata_panel.ipynb` — load the eight-city panel.
- `code_2_utci_climatology.ipynb` — hourly radiation-aware UTCI demand.
- `code_3_streetscape_heat_resilience.ipynb` — Hong Kong summer calibration, SHR, figures.
- `code_4_vata_maps_distributions.ipynb` — eight-city VATA distributions and hex maps vs Hong Kong mean.
- `code_5_robustness_checks.ipynb` — calibration sensitivity and Singapore-relative domain diagnostics.
- `scripts/baselines.py` — green-view and 1 − sky-fraction baselines through the same Hong Kong calibration, street-level disagreement with VATA, and the capture-date table (`python -m scripts.baselines`).
- `scripts/baseline_figures.py` — `fig_street_agreement` (paper Figure 4) and `fig_baselines` (`python -m scripts.baseline_figures`).
- `scripts/planning.py` — indicative planting, shading, and protection strategies from the hottest-quarter clusters (`python -m scripts.planning`).

The TCIS perception survey, 500-image training table, and MTNNL weights stay in [Thermal Comfort in Sight](https://github.com/Sijie-Yang/Thermal-Comfort-In-Sight).

## Run

```bash
pip install -r requirements.txt
# from this folder, with the project root on PYTHONPATH
jupyter notebook code_1_eight_city_vata_panel.ipynb
```

Adequacy bar: Hong Kong mean VATA and June–August heat-degree demand (`max(UTCI−26, 0)`), not equatorial Singapore and not Hong Kong’s annual mean.

## Citation

Yang, S., Chong, A., Liu, P., Li, W., Wei, X., Li, J., Sun, H., Reséndiz García, D., and Biljecki, F. (2026). Heat Resilience in Sight: Climate-Conditioned Thermal Affordance of Streetscapes Across Cities and Seasons. *Tackling Climate Change with Machine Learning workshop at NeurIPS 2026*.

## License

Code is released under the MIT License (see `LICENSE`).
