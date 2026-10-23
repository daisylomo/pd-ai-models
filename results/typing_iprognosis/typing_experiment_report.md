# i-PROGNOSIS smartphone typing baseline

**Scope:** Retrospective classification of known PD versus control status from smartphone touchscreen typing; not prospective risk prediction.

- Cohort: 33 participants (15 controls, 18 PD); 338 excerpt files.
- Event audit: 26343 valid events; 0 malformed lines; 1 timestamp reversals; 7 pressure values outside documented [0,1].
- Split: 26 training and 7 held-out participants; model selection used 5-fold grouped CV on training participants only.
- Selected model: random_forest, parameters `{"classifier__max_features": "sqrt", "classifier__min_samples_leaf": 2}`.
- Test support: {'control': 3, 'pd': 4}; accuracy 0.857; PD precision 1.000; PD recall 0.750; PD F1 0.857; ROC-AUC 1.0.
- Confusion matrix, rows=true and columns=predicted [control, PD]: `[[3, 0], [1, 3]]`.
- Class-level metrics: {"accuracy": 0.8571428571428571, "control": {"f1-score": 0.8571428571428571, "precision": 0.75, "recall": 1.0, "support": 3.0}, "macro avg": {"f1-score": 0.8571428571428571, "precision": 0.875, "recall": 0.875, "support": 7.0}, "pd": {"f1-score": 0.8571428571428571, "precision": 1.0, "recall": 0.75, "support": 4.0}, "weighted avg": {"f1-score": 0.8571428571428571, "precision": 0.8928571428571429, "recall": 0.8571428571428571, "support": 7.0}}
- Estimates are unstable due to the small single-study cohort. No external validation or clinical readiness is claimed.
- Dataset license: CC BY 4.0. Cite Iakovakis et al. (2018), Scientific Reports 8, 7663, doi:10.1038/s41598-018-25999-0.
