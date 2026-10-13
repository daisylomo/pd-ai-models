# Architecture and data flow

Each dataset has a separate pipeline. Images become grayscale pixel vectors, pen traces become task-level features, and Tappy event logs become monthly summaries then one participant row. Each trainer owns splitting, model selection, evaluation and artifact writing.

```mermaid
flowchart TD
  subgraph Drawing
    DI[Raw labeled images] --> DM[Manifest, hash dedupe, provisional groups]
    DM --> DV[Grouped train / validation / test]
    DV --> DX[32 by 32 grayscale vectors]
    DX --> DCV[Grouped CV and validation selection]
    DCV --> DT[Held-out image test]
  end
  subgraph Pen
    PI[Raw trace files] --> PH[Parse and deduplicate]
    PH --> PF[Task trajectory features]
    PF --> PS[Provisional group splits]
    PS --> PCV[Grouped CV and validation selection]
    PCV --> PT[Held-out task test]
  end
  subgraph Tappy
    TE[Monthly event files] --> TV[Validate, match, audit]
    TP[User profiles] --> TV
    TV --> TM[Monthly timing, hand, direction summaries]
    TM --> TA[Median aggregation per participant]
    TA --> TS[Stratified participant holdout]
    TS --> TCV[Train-only CV and pipeline selection]
    TCV --> TT[Held-out test and bootstrap]
    TA --> INF[Saved pipeline input]
    INF --> PRED[Class and ranking score]
  end
  DT --> ART[Models, metrics, tables and plots]
  PT --> ART
  TT --> ART
```

For Tappy, monthly records are aggregated before the participant split. Imputation and scaling are inside sklearn pipelines and learned within training folds. Drawing and pen groups are filename-derived provisional tokens; source-level identity verification is still needed. The saved estimators are independent: no shared feature space or verified cross-dataset participant key exists for fusion.
