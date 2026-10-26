# Drawing data quality and identity audit

## Verified local inventory

The active trainer reads only `data/Parkinson's drawings dataset/`. It finds 204 PNG files after exact-hash deduplication: 102 spiral images and 102 wave images, each task containing 51 healthy and 51 Parkinson's-labelled images. For each task/class cell, the source folders contain 36 files in `training/` and 15 in `testing/`.

Spiral files are generally 256×256, with a few small dimension variations. Wave files are 512 pixels wide with heights observed between 138 and 516. Files are RGB or RGBA although visible content is grayscale-like. A visual sample showed faint strokes and variable paper/background texture. These acquisition differences are plausible nuisance signals; whether they correlate with class has not been established.

No byte-identical duplicate files occur in the 204-image active tree, and the manifest checks conflicting and repeated SHA-256 hashes before fitting. A coarse 63-bit perceptual hash over 32×32 grayscale thumbnails found no pairs with Hamming distance ≤5 (checked separately by task and across all images); this does not rule out more subtle near-duplicates. The source directory names `training` and `testing` do not encode independent groups. A source-level identity crosswalk remains absent.

## Evaluation implications

`src.train_drawing_models` uses class-scoped `V##` strings as provisional groups, and its saved split contains 114/42/48 images in train/validation/test. The test contains 48 images but only four candidate groups: one healthy group contributes all 30 healthy test images, and three Parkinson's groups contribute 18 images. Its scores are therefore sensitive to the identity interpretation and the particular groups selected.

The active drawing experiment is a baseline only. Its grouped split does not establish participant-independent generalization. Historical comparison outputs remain under `results/drawing_representation_comparison/`, but the comparison trainer is no longer part of the active codebase.

## Provenance and licence

No licence, citation, article, clinical metadata, image acquisition protocol or verified identity mapping was found in the local drawing dataset tree or the repository's dataset notes. These are unresolved, not evidence that the data are unrestricted. Do not redistribute raw or derived drawing data until terms are established.
