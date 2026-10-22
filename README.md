# Jewellery Model 3

A new jewellery image retrieval and matching system.

## Principles

- Independent of Model 1 and Model 2.
- Freshly sourced and documented dataset.
- Validate matching labels before supervised training.
- Establish a measurable retrieval baseline before fine-tuning.
- Keep training separate from production inference.
- Export and test the inference artifact before deployment.
- Use CPU-compatible development and resumable checkpoints.

## Planned workflow

1. Dataset sourcing, licensing review, and validation.
2. Image quality checks and duplicate detection.
3. Establish reliable gold-to-prototype evaluation pairs.
4. Pretrained image-embedding baseline.
5. Retrieval evaluation and justified model improvements.
6. Export and test ONNX inference.
7. Flask API and production startup tests.
8. Deploy as an independent service.

## Environments

Training dependencies: `requirements-train.txt`

Production inference dependencies: `requirements-inference.txt`

The inference requirements target a Linux deployment environment.
Gunicorn is intended for production deployment, not the Windows
development server.

## Dataset

Raw and processed images are intentionally excluded from Git.
Keep dataset provenance and any matching labels documented.
Do not treat unrelated gold and prototype images as positive pairs.

## Configuration

See `configs/config.yaml`. Adjust settings only after baseline tests.
