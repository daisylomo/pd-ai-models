# Dataset inventory and research boundaries

This page distinguishes what the repository contains from what current trainers consume. Raw data remain local and unchanged; counts below describe this checkout and may differ from upstream distributions.

## Modelled datasets

### Spiral and wave drawings

The trainer reads `data/Parkinson's Datasets/Parkinson's drawings dataset/drawings/{spiral,wave}/{training,testing}/{healthy,parkinson}`. Folder names define the class label. `V##` filename prefixes are treated as class-scoped participant-group candidates. Exact-content hashes remove duplicate image files. The active feature representation is a 32x32 grayscale image, flattened to 1,024 normalized pixel values. The local processed manifest contains 204 unique images and 28 candidate groups. Test results are image-level and group-partitioned, but the group tokens have not been verified against source participant records.

The same source tree also contains sibling `spiral/` and `wave/` directories and a second set of drawings under `Park's UCI kaggle dataset/hw_drawings`; these are not read by the current image trainer. Do not merge them based on similar task names without checking provenance and subject overlap.

### Pen trajectories

The trainer reads only the two `hw_dataset` roots under `Parkinson's UCI dataset` and `Park's UCI kaggle dataset`. Rows include `X`, `Y`, `Z`, `Pressure`, `GripAngle`, `Timestamp`, and `Test ID`. Folder labels define control/Parkinson classes; Test ID is treated as task category (0 static spiral, 1 dynamic spiral, 2 circular motion, according to the local readme). The trainer summarizes point geometry, step and turn behavior, pressure, grip angle and z. It does not use raw timestamps as a numeric predictor. Local audit: 80 files scanned, 40 unique recording contents, 110 task samples, 15 control and 10 Parkinson candidate groups. `C_####` and truncated `P_####` filename tokens are provisional group identifiers and need verification in the source metadata. A test fold containing one PD candidate group makes the high test score particularly unstable.

`hw_drawings/` and `new_dataset/` copies are present but are not used by this trainer. They may overlap with the active recordings and should be provenance-checked before analysis.

### PhysioNet Tappy Keystroke

The local root is `data/Parkinson's Datasets/tappy-keystroke-data-1.0.0/`. It contains `Archived-users.zip` and an expanded `Archived-Data/Tappy Data/` folder. The separately referenced `Archived-Data.zip` is absent locally; it is not needed by the current parser because the event files are expanded. The eight event columns are `UserKey`, `Date`, `Timestamp`, `Hand`, `Hold time`, `Direction`, `Latency time`, and `Flight time`. User profile records provide a `Parkinsons` self-report and demographic, symptom, severity/impact and medication fields.

The current audit found 227 profiles and 622 monthly event files. After profile matching and row rules, the participant-level cohort has 217 participants (55 control, 162 self-reported PD), 558 matched monthly files, and 8,835,813 usable events. The trainer's target is profile self-report; it is not clinician-adjudicated. UPDRS is missing/unknown for every control and 154 of 162 PD profiles, so it is not a supported cohort-wide target. Monthly features are combined by participant before splitting. The project calls this Tappy Keystroke, not a standardized tapping assessment.

## Present but not modelled by current training scripts

The local tree contains substantial Parkinson's kinetics data (including C3D files and gait-cycle exports, plus `PDGinfo.xlsx`), motion/UDysRS-UPDRS CSV exports, and other `new_dataset` / `hw_drawings` data. The EDA notebook inventories or inspects some of these areas, but none of the three current trainer modules loads them for model fitting. Their schemas, participant identifiers, labels, linkage, and intended outcome need a dedicated audit before adding a pipeline. Similar column names do not establish equivalence between datasets or clinical scores.

## Shared limitations and target boundary

Current labels support retrospective known-status comparisons. None of the active sources establishes that a participant was pre-diagnostic, supplies a prospective conversion outcome, or supports a time horizon for future risk. Therefore “early PD risk assessment” remains a longer-term research motivation, not the task performed by these models. A valid risk study would need appropriately timed baseline measurements, verified follow-up/outcomes, a prespecified horizon, and participant-level or temporal holdouts.

No reliable participant-level crosswalk between the image, pen, Tappy, gait or motion sources is present. Independent evaluation is justified; paired multimodal fusion is not. Dataset-level transfer could only be assessed after target definitions, populations, and feature semantics are made comparable, and would measure domain transfer rather than personal multimodal prediction.
