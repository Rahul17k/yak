# Yakshagana Vesha Identifier

A React/Vite and FastAPI web app for the provided TensorFlow/Keras MobileNetV2 model. The six trained output labels and their index order come from `yakshagana_work/saved_models/class_mapping.json`.

## Important model limitation

The provided dataset contains six Yakshagana vesha classes and no non-Yakshagana examples. The classifier has no learned “Not Yakshagana” class. The app adds a **best-effort novelty screen**: it compares a normalized intermediate model embedding with the training-image embeddings and rejects images below a similarity cutoff. This screen can still accept an unrelated image that looks similar in embedding space, and it can reject genuine Yakshagana images. It cannot guarantee that every non-Yakshagana photo is rejected.

The novelty cutoff is selected from Yakshagana validation images only, with a 75% target acceptance rate. The present validation set retains 78/103 images (75.7%) at cutoff 0.9120. This intentionally trades coverage of genuine Yakshagana images for stronger rejection; it does not measure non-Yakshagana false acceptance. No non-Yakshagana test images are available, so that rate is unknown.

The app also requires a top-class confidence score of at least 85% before reporting a vesha; lower scores are marked uncertain. This is a user-selected score cutoff, **not 85% accuracy** or an 85% probability that the result is correct. The score is produced by a six-class softmax and can still be high for an unrelated image.

The saved split contains 478 training, 103 validation, and 103 held-out test images. In the current evaluation, six-class classifier accuracy is 90.3%; app-decision Model accuracy is 72.8% (75/103), counting novelty rejections and uncertain results as incorrect. Twenty-one of 103 genuine test images were rejected by the novelty screen, and seven were marked uncertain by the 85% confidence cutoff. These held-out metrics do not guarantee performance on new performances or unrelated image sources. The novelty threshold is selected on validation data; the test split is not used to select thresholds.

## Blur/distance experiment (not deployed)

`train_yakshagana_blur.py` fine-tunes a separate candidate using synthetic zoom-out, downsampling, and Gaussian blur on training images only. It does not modify the dataset or the active model weights. To recreate and evaluate it:

```powershell
python .\train_yakshagana_blur.py
python .\evaluate_yakshagana.py --artifacts .\yakshagana_work\blur_robust_candidate --blur-robust-validation
```

In the current experiment, synthetic blur/distance validation app accuracy improved from 53.1% to 59.2%. However, the candidate accepted the known unrelated HSAM3855 photo as Stree Vesha at 99.2% class confidence, so it was **not deployed**. Raising its novelty cutoff enough to reject that photo would accept only about 37% of mildly degraded and 5% of moderately degraded validation views. These synthetic results are not a substitute for real distant/blurred Yakshagana photos and representative non-Yakshagana negatives.

## Requirements

- Python 3.13 (TensorFlow 2.20)
- Node.js and npm
- The `Yakshagana dataset` directory and the saved model files under `yakshagana_work/saved_models`

Run these commands from the workspace root in PowerShell:

```powershell
python -m pip install -r requirements.txt
npm --prefix .\frontend install
```

## Run the application

Open two PowerShell terminals at the workspace root.

Terminal 1 — API:

```powershell
python -m uvicorn api.main:app --host 127.0.0.1 --port 8017
```

Terminal 2 — web UI:

```powershell
npm --prefix .\frontend run dev -- --host 127.0.0.1
```

Open <http://127.0.0.1:5173>. The API health check is at <http://127.0.0.1:8017/api/health>.

The production frontend can also be served by FastAPI:

```powershell
npm --prefix .\frontend run build
python -m uvicorn api.main:app --host 127.0.0.1 --port 8017
```

Then open <http://127.0.0.1:8017>.

## Evaluate and test

Regenerate validation-selected uncertainty and novelty thresholds, fixed reference photos, training embeddings, and held-out test metrics without training. Run this before starting the API if the evaluation artifacts are missing:

```powershell
python .\evaluate_yakshagana.py
```

Run the API integration tests:

```powershell
python -m unittest discover -s tests -v
```

The generated `evaluation.json` records per-class precision, recall and F1, the decision confusion matrix, class image counts, uncertain predictions, Yakshagana images rejected by the novelty screen, and the unknown non-Yakshagana false-accept rate.

## API

- `POST /api/predict` — send the image bytes as the request body with `Content-Type: image/jpeg`, `image/png`, or `image/webp`. Maximum 10 MB. Image content is validated, decoded in memory, EXIF orientation is applied, and the upload is not written to disk.
- `GET /api/metrics` — held-out test metrics and validation-derived thresholds.
- `GET /api/references` — fixed reference-photo selections from training images.
- `GET /api/references/{index}` — a selected reference image.

Prediction “Confidence” is the six-class model score, not a probability that the image is Yakshagana and not a guarantee that the answer is correct. A “Not a Yakshagana Vesha” result means the novelty screen found the image unlike the available Yakshagana training embeddings; it is not a learned negative-class prediction. Crop group photos to one main performer for the intended use. Uploads are processed in memory and are not stored.
