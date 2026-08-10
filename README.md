# LUCID-PD

A clinical decision support prototype for Parkinson's disease that treats validation
protocol as the object of study. It detects Parkinson's from a sustained-vowel recording,
explains each decision with SHAP evidence and retrieval-grounded text, routes the result
to a care pathway through auditable rules, and reproduces the popular MRI convolutional
baseline in order to show why its reported accuracy does not survive a correct split.

> **Decision support and research only. This is not a medical device and not a diagnosis.**
> Nothing produced here should be used to make or delay a clinical decision.

MSc Individual Research Project, 7005SCN, Coventry University. Author: Nkese Eyo.

## The problem

Published voice-and-MRI decision support systems for Parkinson's routinely report between
95% and 99% accuracy. Two design choices manufacture most of that number.

The first is splitting on records rather than on people. The standard speech corpora hold
several recordings per participant, so a random split places recordings of the same person
on both sides of the fold boundary. The model can then recognise the speaker instead of
the pathology, and the reported figure measures speaker identification.

The second is fusing corpora that share no subjects. No public dataset pairs voice with
imaging for the same person, so a "multimodal" system built from a voice corpus and an
unrelated imaging corpus is not performing fusion at all.

This project reproduces both designs, quantifies what they add to the headline figure, and
builds the alternative under subject-level validation throughout.

## What is here

- A leakage-controlled speech benchmark on UCI #470, comparing four classifiers under
  subject-level grouped cross-validation against the same models under a record-level split.
- The deployed voice detector: openSMILE eGeMAPS features scored by a gradient boosted
  classifier, serialised in `models/deployed_voice.joblib`.
- A retrieval-grounded explanation layer that constrains a small language model to a
  curated corpus of NICE and NHS guidance, measured for faithfulness against an
  unconstrained control.
- The MRI critique baseline: five convolutional backbones trained under image-level and
  scan-level splits, with Grad-CAM and a dataset audit that identifies the confound.
- A rule-based care-route recommender that maps a risk band to a referral type through
  explicit, inspectable rules rather than a learned policy.
- A FastAPI service and React interface that serve all of the above from one container.

## Results

All tables are reproduced from the CSV files in `results/tables/`, which the commands in
[Reproducing the experiments](#reproducing-the-experiments) regenerate.

### The validation protocol decides the headline number

The same standardised logistic regression, on the same UCI #470 features, evaluated under
two cross-validation protocols. Only the split changes.

| Protocol | Accuracy | ROC-AUC |
| --- | --- | --- |
| Record-level split | 0.827 ± 0.008 | 0.868 ± 0.025 |
| Subject-level grouped split | 0.768 ± 0.063 | 0.775 ± 0.074 |

Splitting on records rather than people adds about 6 accuracy points and 9 AUC points to
an otherwise identical experiment. The variance also collapses under the record-level
split, because folds that share speakers are not independent, which makes the inflated
result look more stable than the correct one.

### Speech detection under subject-level validation

Four classifiers on UCI #470, subject-level grouped cross-validation, 252 participants
(188 with Parkinson's, 64 controls) and 756 recordings.

| Model | Accuracy | Balanced acc. | Sensitivity | Specificity | ROC-AUC | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic regression | 0.759 ± 0.077 | 0.693 | 0.826 | 0.560 | 0.775 ± 0.083 | 0.378 |
| SVM (RBF) | 0.796 ± 0.054 | 0.678 | 0.916 | 0.439 | 0.791 ± 0.070 | 0.406 |
| Random forest | 0.808 ± 0.037 | 0.657 | 0.963 | 0.351 | 0.837 ± 0.070 | 0.423 |
| XGBoost | **0.850 ± 0.046** | **0.745** | 0.957 | 0.533 | **0.870 ± 0.068** | **0.567** |

The specificity column is the part worth reading. Every model is far better at identifying
Parkinson's than at clearing a control, which the 3:1 class imbalance encourages and which
raw accuracy hides. Balanced accuracy and MCC are reported alongside for that reason, and
the leading model still misses close to half of the controls.

Nested subject-level hyperparameter search improved the weaker models and did not improve
the leading one: logistic regression gained 0.038 ROC-AUC, while XGBoost lost 0.008. The
default configuration is therefore reported above.

SHAP attribution over the fitted XGBoost is dominated by cepstral dynamics and tunable-Q
wavelet coefficients rather than the classical jitter and shimmer measures, and the top
features recur across folds.

### Grounding the explanation layer

Three small language models generate a plain-language account of each decision, once
constrained to retrieved guideline text and once without that constraint. Nine cases per
condition. Faithfulness is the proportion of claims supported by the retrieved text, and
"unsupported" counts the claims per response that are not.

| Model | Condition | Faithfulness | Unsupported claims | Reading grade |
| --- | --- | --- | --- | --- |
| Qwen3-4B | grounded | **1.000** | **0.00** | 10.9 |
| Qwen3-4B | ungrounded | 0.702 | 1.78 | 12.2 |
| Gemma 4 E4B | grounded | 0.989 | 0.11 | 16.7 |
| Gemma 4 E4B | ungrounded | 0.524 | 3.44 | 16.4 |
| Phi-4-mini | grounded | 0.831 | 1.33 | 20.3 |
| Phi-4-mini | ungrounded | 0.729 | 1.44 | 16.1 |

Removing the grounding constraint costs every model, and costs the most capable one the
most: Gemma 4 falls from 0.11 unsupported claims per response to 3.44. Qwen3-4B produces
no unsupported claim at all when grounded, at the most accessible reading level of the
three, which is why it is the model the deployment serves.

### The MRI baseline does not survive a scan-level split

Five backbones on a public Parkinson's MRI classification dataset, trained first with
images split at random and then with every image from a scan held to one side.

| Backbone | Image-level acc. | Image-level AUC | Scan-level acc. | Scan-level AUC | Scan-level specificity |
| --- | --- | --- | --- | --- | --- |
| VGG-16 | 0.971 | 0.996 | 0.693 | 0.716 | 0.380 |
| Small CNN | 0.968 | 0.992 | 0.929 | 0.971 | 0.956 |
| ResNet-50 | 0.953 | 0.994 | **0.651** | **0.583** | **0.125** |
| EfficientNet-B0 | 0.950 | 0.987 | 0.821 | 0.857 | 0.483 |
| DenseNet-121 | 0.945 | 0.981 | 0.774 | 0.861 | 0.741 |

Under the image-level split every backbone lands between 0.945 and 0.971, the range the
literature reports. Under the scan-level split ResNet-50 falls to 0.583 AUC, which is
close to chance, and its specificity drops from 0.983 to 0.125: it has stopped
discriminating and begun answering "Parkinson's" to almost everything.

A dataset audit locates the confound. The 3,284 images resolve to 43 distinct acquisition
sequences, 86% of which appear under one class label only. The separable signal is the
scanner protocol, not the pathology, and Grad-CAM agrees: the attended regions frequently
fall outside the brain.

The application exposes this model deliberately, labelled as a leakage-inflated research
baseline, so that the failure can be seen rather than described.

### The deployed voice model

The model the application serves is trained on the Italian Parkinson's Voice and Speech
corpus with eGeMAPS features, evaluated at subject level. Six speakers are excluded from
training entirely: their recordings ship as the demonstration set, and a demonstration
proves nothing if its clips were also training data, so every demo run is a prediction
about a subject the model has never seen.

| Metric | Value |
| --- | --- |
| Accuracy | 0.959 ± 0.045 |
| Balanced accuracy | 0.958 |
| Sensitivity | 0.977 |
| Specificity | 0.940 |
| ROC-AUC | 0.992 ± 0.014 |
| MCC | 0.920 |

On the six held-out speakers, five fall on the correct side of 0.5 when their recordings
are averaged, and one person with Parkinson's is borderline, which is what genuine
prediction on unseen subjects looks like (`results/tables/ipvs_demo_holdout.csv`).

These figures are higher than the UCI #470 results above and should not be read as the
better result. No recording from one person crosses a fold boundary, but the control and
Parkinson's recordings come from different cohorts whose age distribution and recording
conditions plausibly separate them on their own. The validated findings of this project
are the UCI #470 results; this model exists so that the demonstrator can accept live audio.

## The application

One FastAPI process serves the compiled React interface and the API from the same origin.

| Endpoint | Behaviour |
| --- | --- |
| `POST /predict/voice` | eGeMAPS features, gradient boosted prediction, SHAP evidence, grounded explanation, rule-based care route |
| `POST /predict/mri` | Image-level CNN with a Grad-CAM overlay, returned with an explicit leakage caveat |
| `POST /predict/combined` | Decision-level average of the two, labelled illustrative because the corpora are unpaired |
| `GET /health` | Liveness check |
| `GET /docs` | Interactive API documentation |

Uploads are written to a temporary directory that is destroyed as the response is built,
so no recording or image outlives the request that carried it. Audio type is resolved from
container magic bytes before falling back to the filename, because browsers post captured
recordings under generic names.

## Installation

Requires Python 3.11 and Node 20.

```bash
conda env create -f environment.yml
conda activate parkinsons_cdss
pip install -e ".[app,dev]"
```

Additional extras are installed per strand: `.[audio]` for openSMILE feature extraction,
`.[mri]` for the imaging baseline, `.[slm]` for a local explanation backend, and `.[data]`
for the dataset download helpers.

## Running the application

```bash
# backend, from the repository root
PYTHONPATH=src uvicorn app.backend.main:app --port 8000

# frontend, in a second terminal
cd app/frontend && npm install && npm run dev
```

The interface opens on the URL Vite prints. `models/deployed_voice.joblib` is included in
this repository, so voice mode works from a fresh clone. MRI mode additionally needs
`models/deployed_mri.pt`, which is rebuilt with `python -m pdcdss.mri.deploy_mri`.

The explanation layer is selected by `SLM_BACKEND`: `hf` calls Hugging Face Inference
Providers and needs `HF_TOKEN`, `ollama` reaches a local daemon, and `off` disables
generation. With no backend available the service still answers every request, returning a
deterministic summary that states the probability, the risk band, the SHAP feature
families and the retrieved guideline text verbatim.

## Reproducing the experiments

The datasets are not redistributed here. Download them first:

```bash
python src/pdcdss/data/download_speech.py --dataset 470   # primary speech corpus
python src/pdcdss/data/download_speech.py --dataset 174   # external benchmark
python src/pdcdss/data/download_ipvs.py                   # audio for the deployed model
python src/pdcdss/data/download_mri.py --kaggle owner/slug # imaging critique baseline
```

Then run the strands. Each writes its tables to `results/tables/` and its figures to
`results/figures/`.

```bash
python -m pdcdss.speech.eda                # corpus description
python -m pdcdss.speech.experiments        # leakage comparison and model comparison
python -m pdcdss.speech.tune               # nested subject-level hyperparameter search
python -m pdcdss.speech.shap_analysis      # SHAP attribution and cross-fold stability

python -m pdcdss.mri.audit                 # sequence and duplicate audit, no GPU needed
python -m pdcdss.mri.leakage               # five backbones, both split protocols
python -m pdcdss.mri.ablation              # fine-tuning depth against the leakage gap

python -m pdcdss.explain.cases             # build cases from real leakage-free predictions
python -m pdcdss.explain.rq3_experiment    # grounded against ungrounded explanation

python -m pdcdss.speech.build_ipvs_features
python -m pdcdss.speech.deploy_voice       # rebuild models/deployed_voice.joblib
python -m pdcdss.mri.deploy_mri            # rebuild models/deployed_mri.pt
```

`pdcdss.mri.leakage` and `pdcdss.mri.ablation` need a CUDA-capable GPU to finish in
reasonable time. Everything else runs on a laptop CPU.

## Deployment

The whole system runs as a single Docker container on Hugging Face Spaces: a Node stage
compiles the React bundle and the Python stage serves it alongside the API on port 7860.
The frontend is not hosted separately. See [deploy/README.md](deploy/README.md) for the
procedure and the required configuration.

## Data and ethics

Every corpus used here is public, de-identified secondary data. No data are collected from
people and no identifiable information is processed, consistent with ethics approval
P194723. Nothing under `data/` is redistributed in this repository; the download scripts
above re-create it. Licences, provenance and the known limitations of each corpus are
recorded in [data/README.md](data/README.md).

## Limitations

- No clinician or patient study was run. Explanation quality is measured computationally
  through faithfulness, readability and consistency, never through expert judgement, and a
  clinician evaluation remains necessary before any claim about usefulness.
- The imaging corpus has opaque provenance. That opacity is part of the critique rather
  than an accident, but it means the MRI results characterise this dataset and cannot be
  generalised to imaging for Parkinson's in general.
- Combined mode averages two predictions over unpaired corpora. It illustrates an interface
  pattern and is not trained joint fusion.
- The deployed voice model is cohort-confounded, as set out above.
- Specificity is the weak point of every speech model reported here, and a screening tool
  that clears controls poorly is a tool that generates unnecessary referrals.

## Licence

MIT, see [LICENSE](LICENSE). The licence covers the source code in this repository. The
datasets and any trained model derived from them remain subject to the terms of their
original sources.

## Citation

See [CITATION.cff](CITATION.cff).
