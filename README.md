# Multi-label Classification of Dutch Election Manifestos

Automatic thematic coding of Dutch election manifesto paragraphs, where each paragraph can
carry several themes drawn from a 380-theme inventory with a severe long tail. We compare
classical TF–IDF pipelines, dense sentence embeddings, and a fine-tuned Dutch transformer
(RobBERT-2023), then run a controlled ablation to isolate how much of the difference between
them is caused by the long tail rather than by model capacity.

The headline result is a reversal. On the full label space the classical TF–IDF baseline wins
(Micro-F1 0.508 vs. 0.310 for RobBERT-2023). Once rare labels are removed, RobBERT-2023
overtakes it from a frequency cutoff of 75 onward, reaching Micro-F1 0.633 and Macro-F1 0.598
at cutoff 100. The transformer's weakness in the full setting is supervision per label, not
architecture.

**Full write-up:** [`report.pdf`](report.pdf)

## Data

Dutch election manifestos from 1986, 1994 and 1998, segmented into paragraphs and
thematically annotated by experts. Supplied as one XML file per election year plus a matching
theme taxonomy per year.

After parsing: 2,533 paragraphs, 380 unique themes, 8.5 themes per paragraph on average
(range 1–68), 317 words per paragraph on average. Per-year statistics are in Table 1 of the
report.

The corpus was provided by the course and is not redistributed here. It is described in
Verberne et al. (2014), "Automatic thematic classification of election manifestos",
*Information Processing & Management* 50(4), 554–569.

## Approach

**Classical baselines.** Six TF–IDF representations (unigrams, lemmatised unigrams,
lemmatised without stopwords, and the same three with bigrams) crossed with four classifiers
(Multinomial Naive Bayes, Logistic Regression, balanced Logistic Regression, LinearSVC) —
24 one-vs-rest pipelines, selected on Micro-F1 with Macro-F1 as tie-breaker.

**Dense embeddings.** Paragraph-level SentenceTransformer vectors fed to the same
best-performing classifier, holding the classifier family fixed so the comparison isolates the
representation.

**Transformer.** RobBERT-2023 fine-tuned with focal loss (γ = 2.0, α = 0.25) to counter the
dominance of easy negatives, with a single global decision threshold swept on validation to
maximise Macro-F1.

**Long-tail ablation.** Labels below global frequency cutoffs of 5, 25, 50, 75 and 100 are
progressively removed and both the classical pipeline and RobBERT-2023 are re-trained on each
filtered label set. This is the part that explains the rest of the results.

Splits are 70/15/15, multi-label stratified to preserve co-occurrence. Seed fixed at 42.

## Results

Full label space:

| Method | Precision | Recall | Micro-F1 | Macro-F1 |
|---|---|---|---|---|
| TF–IDF (balanced LR, lemmatised uni+bigrams, no stopwords) | 0.459 | 0.569 | **0.508** | **0.344** |
| Dense embeddings | 0.178 | 0.603 | 0.275 | 0.190 |
| RobBERT-2023 | 0.202 | 0.666 | 0.310 | 0.162 |

Both neural approaches over-predict: high recall, poor precision. Classifiers without
imbalance correction fail the other way — plain Logistic Regression reaches precision 0.908 at
recall 0.058, predicting almost nothing.

Long-tail ablation:

| Cutoff | Labels | Classical Micro-F1 | Classical Macro-F1 | RobBERT Micro-F1 | RobBERT Macro-F1 |
|---|---|---|---|---|---|
| 5 | 354 | 0.509 | 0.360 | 0.370 | 0.177 |
| 25 | 202 | 0.530 | 0.472 | 0.448 | 0.360 |
| 50 | 130 | 0.546 | 0.497 | 0.513 | 0.447 |
| 75 | 81 | 0.571 | 0.530 | 0.581 | 0.533 |
| 100 | 57 | 0.593 | 0.562 | **0.633** | **0.598** |

Both improve as rare labels go, but RobBERT-2023 gains far more, crossing the classical
baseline at cutoff 75. Macro-F1 nearly quadruples for the transformer (0.177 to 0.598) against
a 56% gain for the classical pipeline.

## Running it

`code.ipynb` was written for Google Colab and mounts Drive to reach the corpus. To run it
elsewhere, replace the mount cell and set the data path at the top of the notebook to a
directory holding the manifesto XML and taxonomy files.

The RobBERT-2023 fine-tune uses batch size 2 with gradient accumulation 4, learning rate 2e-5,
10 epochs, cosine schedule with 0.1 warmup, max sequence length 512.

## Limitations

- A single global decision threshold is used for all labels. Frequent and rare themes need
  different operating points, and per-label thresholds would likely close part of the
  precision gap.
- The dense embedding arm uses one off-the-shelf sentence encoder; no Dutch-specific or
  fine-tuned encoder was tried, so it under-tests the representation rather than the idea.
- Label dependencies are ignored — one-vs-rest and independent sigmoids both assume themes are
  predicted separately, though the annotations are clearly correlated.
- Results come from one seed and one split. No variance estimates.

## Context

Final assignment for the Text Mining course, LIACS, Leiden University, 2025. Joint work with
Luis Chial Sanchez: Luis handled the classical and embedding models, I handled RobBERT-2023
and the long-tail experiments, and we finalised the report together.
