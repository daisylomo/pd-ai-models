# Dataset inventory and research boundaries

The active experiments use two local datasets: spiral/wave drawing images and the i-PROGNOSIS virtual touchscreen typing development dataset. Raw data are ignored by Git and remain in their original directories.

## Spiral and wave drawings

- **Local source:** `data/Parkinson's drawings dataset/{spiral,wave}/{training,testing}/{healthy,parkinson}/`.
- **Verified local counts:** 204 PNG images after exact duplicate checking: 102 spiral (51 per class) and 102 wave (51 per class). Each task/class contains 36 files in `training/` and 15 in `testing/`.
- **Image properties:** spiral images are mostly 256×256 RGB/RGBA; wave images are 512 pixels wide with varying heights (observed 138–516 pixels). Images show grayscale-like strokes and variable paper/background texture despite RGB/RGBA encodings.
- **Labels:** class is inferred from the `healthy` or `parkinson` directory. There are no clinical records or verified person crosswalks locally.
- **Filename identity:** names begin with tokens such as `V01`, followed by varying task/class-like suffixes. `V##` meaning is not verified by local source documentation. The same tokens occur in source `training/` and `testing/` folders; those folders are not participant-disjoint. Results using them as groups are explicitly exploratory candidate-group estimates, not confirmed participant-level validation.
- **Duplicates:** no identical file hashes were found among the 204 files consumed by the trainer. A qualitative image review confirmed scan/background variation. A systematic near-duplicate audit has not been completed.
- **Features:** the active baseline directly resizes grayscale images to 32×32 (1,024 pixels), which distorts aspect ratio. It uses static image values and does not measure movement time, speed, pauses, or true temporal smoothness. Historical comparison outputs are archived under `results/drawing_representation_comparison/`; that experiment is no longer active.
- **Source/licence:** no dataset licence, citation, source article, participant metadata, or acquisition protocol was found in the local drawing source tree inspected. Do not infer redistribution rights or clinical population characteristics from folder labels.

## i-PROGNOSIS virtual touchscreen typing

- **Local source:** `data/virtual keyboard data/iprognosis_typing_dev/` (`Data/`, `Demographics_Clinical_Characteristics.csv`, and `ReadMe.txt`).
- **Verified local cohort:** 33 participants (18 PD, 15 controls), with 338 text-excerpt event files. Excerpts are nested under participant folders and link to `Subject ID`; the diagnostic label comes from the clinical CSV `Group` column (`PD`/`Control`).
- **Documented protocol:** the local ReadMe describes typing on a touchscreen LG Nexus 5X (Android 7.0, 1080×1920 display), using a custom input method. Participants transcribed up to 11 short excerpts: a shared initial 200-character excerpt and further excerpts of 40–115 characters. The content is from *The Little Prince*; event rows used by the pipeline do not include the text.
- **Event format:** each nonblank event line has four comma-separated fields: `Press`, press timestamp in milliseconds, `Release <timestamp>` in milliseconds, and normalized pressure documented as 0–1. Raw text/key content is not included in the event rows used here.
- **Features/unit:** hold time, inter-press timing, press-to-next-press gap, long-pause summaries and in-range pressure summaries are extracted per excerpt, then median-aggregated to one participant row. Event/excerpt counts are audit-only, not predictors. One reversed timestamp pair and seven out-of-range pressure values were audited; malformed or reversed events are not silently treated as valid holds.
- **Evaluation:** fixed-seed participant holdout (26 train, 7 test); five-fold grouped CV on training participants only. Imputation, missingness indicators, scaling and estimators are inside sklearn pipelines and refit in folds. The 7-person test results are highly unstable.
- **Licence and attribution:** local ReadMe documents CC BY 4.0 and requests citation of Iakovakis et al. (2018), *Scientific Reports* 8, 7663, DOI `10.1038/s41598-018-25999-0`. Verify the source terms and citation before redistribution.
- **Limitations:** retrospective known-status classification, small single-study cohort, and no external validation. The local ReadMe notes that 10 participants did not complete the shared initial 200-character excerpt; because excerpt text is absent, completeness cannot be independently checked from event data. Possible confounding remains. This is not prospective risk prediction or a clinical test.

## Boundaries

The drawing and typing cohorts have no verified participant crosswalk, harmonized labels, or common feature space. They are evaluated separately; no fused multimodal model is supported. Current labels describe existing diagnostic groups, not future disease conversion.
