# PhysioGuard

PhysioGuard is a reliability-aware video forensics prototype that combines three facial-region rPPG signals with an independent visual artifact analysis. It is a screening tool, not a legal or medical instrument.

## What is implemented

- OpenCV face detection and tracked Forehead / Left Cheek / Right Cheek ROIs.
- POS-style rPPG extraction from temporal changes in facial skin reflectance associated with blood-volume changes.
- Per-region signal quality, video-quality measurements, cross-region correlation/frequency consistency, and a 3-level verdict.
- An independent visual module. By default it is a transparent pixel-forensic baseline; an optional Hugging Face image classifier can be enabled with `PHYSIOGUARD_VISUAL_MODEL`.
- Quality-aware evidence fusion, evidence-grounded explanations, temporal quality markers, SHA-256 provenance, and a premium Streamlit dashboard.
- Dataset evaluation for visual-only, rPPG-only, visual+rPPG, and full PhysioGuard, with known-source and held-out unseen-source metrics.

The baseline visual module is intentionally not described as a trained deepfake classifier. To use a compatible pretrained image model, set `PHYSIOGUARD_VISUAL_MODEL` to a model available in your local Hugging Face cache or environment and install the optional dependencies.

## Current demo verdict rule

The dashboard currently uses the requested explicit demo rule: if any of Forehead, Left Cheek, or Right Cheek has reliability below 70%, the verdict is `POTENTIAL DEEPFAKE`; if all three are at least 70%, it is `AUTHENTIC`. This is a heuristic demo rule and can produce false positives from motion, lighting, blur, occlusion, makeup, or skin appearance.

## Install and run

Python 3.10+ is recommended. On resource-constrained machines, the app limits BLAS worker threads automatically; you can override that with environment variables.

```powershell
cd C:\hashcode\rppg-deepfake-detection
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the **Video Lab — Record & Analyze** item in the sidebar to record with the browser MediaRecorder API or upload an existing video. The browser component supports camera permission handling, pause/resume/stop, preview, retake, size and extension checks, and an explicit **Use This Video** action. The selected bytes are validated again by the Python backend, decoded with OpenCV, analyzed through `modules.pipeline.analyze_video`, and deleted from the temporary filesystem after processing.

The current Streamlit backend is local rather than an HTTP API: the Video Lab calls the existing Python analysis pipeline directly. Exported JSON is the actual backend result with image arrays omitted from the downloadable JSON and available in the dashboard visualizations.

Upload a facial video, then select **Run forensic analysis**. If the face is not visible for enough frames or fewer than two regions are reliable, the app returns **INCONCLUSIVE** rather than inventing a decision.

## Evaluation

The checked-in videos are discovered from `real/` and `fake/`; synthetic source is the first folder below `fake/`. No results are included in the repository because evaluation must be run on the actual files.

```powershell
python -m evaluation.evaluate --root . --max-per-class 10 --out evaluation_results.json
python -m evaluation.evaluate --root . --max-per-class 20 --held-out-source Sora
```

The evaluation can be expensive because each sample is decoded and analyzed. The output JSON contains actual counts and metric values, including false-positive rate.

## Scientific limits

rPPG is an additional physiological consistency signal, not a claim that deepfakes have no heartbeat. Lighting, motion, blur, makeup, skin appearance, compression, occlusion, camera frame rate, and face detection errors can reduce reliability. A low-quality clip may therefore be inconclusive.

