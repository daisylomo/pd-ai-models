# Drawing data quality and identity audit

## Verified local inventory

The active trainer reads only `data/Parkinson's Datasets/Parkinson's drawings dataset/drawings/`. It finds 204 PNG files after exact-hash deduplication: 102 spiral images and 102 wave images, each task containing 51 healthy and 51 Parkinson's-labelled images. For each task/class cell, the source folders contain 36 files in `training/` and 15 in `testing/`.

Spiral files are generally 256×256, with a few small dimension variations. Wave files are 512 pixels wide with heights observed between 138 and 516. Files are RGB or RGBA although visible content is grayscale-like. A visual sample showed faint strokes and variable paper/background texture. These acquisition differences are plausible nuisance signals; whether they correlate with class has not been established.

No byte-identical duplicate files occur in the 204-image active tree, and the manifest checks conflicting and repeated SHA-256 hashes before fitting. A coarse 63-bit perceptual hash over 32×32 grayscale thumbnails found no pairs with Hamming distance ≤5 (checked separately by task and across all images); this does not rule out more subtle near-duplicates. The source directory names `training` and `testing` do not encode independent groups: candidate `V##` prefixes occur in both folders for both classes. No local source document verifies that `V##` is a real participant identifier or explains its numbering. A source-level identity crosswalk remains absent.

## Evaluation implications

`src.train_drawing_models` uses class-scoped `V##` strings as provisional groups, and its saved split contains 114/42/48 images in train/validation/test. The test contains 48 images but only four candidate groups: one healthy group contributes all 30 healthy test images, and three Parkinson's groups contribute 18 images. Its scores are therefore sensitive to the identity interpretation and the particular groups selected.

The controlled comparison holds the same candidate groups out across spiral, wave and combined experiments. Grouping across tasks by the repeated class-scoped token is a conservative partitioning heuristic to avoid a possible cross-task overlap; it does **not** verify that equal tokens name the same person. Conversely, if equal tokens do not denote the same people, the partition may over-group unrelated samples. Neither result establishes participant-independent generalization.

The original 32×32 preprocessing resizes every image to a square and therefore changes the geometry of non-square wave images. The comparison includes a deterministic aspect-preserving letterbox transform and fixed image-level descriptors. The descriptors quantify thresholded foreground occupancy, spatial distribution, centroid/covariance, radial occupancy and local unsigned gradient orientation. They are image measurements, not true movement speed, rhythm, pauses or temporal smoothness. All learned scaling and model fitting occur inside grouped cross-validation pipelines; the final candidate-group test partition is excluded from model/representation selection.

## Provenance and licence

No licence, citation, article, clinical metadata, image acquisition protocol or verified identity mapping was found in the local drawing dataset tree or the repository's dataset notes. These are unresolved, not evidence that the data are unrestricted. Do not redistribute raw or derived drawing data until terms are established.
