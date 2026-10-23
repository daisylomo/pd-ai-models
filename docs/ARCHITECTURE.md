# Active pipeline architecture

The repository contains independent drawing and virtual-touchscreen-typing experiments. No shared participant crosswalk or harmonized feature space supports fusion.

```mermaid
flowchart TD
  subgraph Drawing
    A[Labeled spiral/wave PNGs] --> B[Hash manifest and provisional class-scoped V groups]
    B --> C[Fixed grouped train/validation/test assignments]
    C --> D[Grouped CV over pixels, descriptors, combined representation]
    D --> E[CV-selected pipeline and reserved candidate-group test]
  end
  subgraph Typing
    F[Excerpt event files] --> H[Validate schema and extract excerpt features]
    G[Clinical CSV Subject ID and Group] --> H
    H --> I[Median aggregate excerpts to one row per participant]
    I --> J[Participant holdout and grouped CV on training only]
    J --> K[Selected pipeline and seven-person held-out test]
  end
  E --> O[Per-experiment models, metrics, assignments, predictions, reports]
  K --> O
```

Drawing image transforms are deterministic and independent per image; scaling and estimators are inside sklearn cross-validation pipelines. `V##` tokens are unverified candidate groups, so even group-disjoint evaluation does not prove participant independence. The typing loader links event folders and excerpt names to the clinical `Subject ID`; excerpts are aggregated before participant-level splitting. Typing imputation, missingness indicators and scaling are fitted inside each training fold.

The legacy handwriting and Tappy source trainers were removed from active source. Their existing model/result artifacts and local raw datasets are retained unchanged as historical material.
