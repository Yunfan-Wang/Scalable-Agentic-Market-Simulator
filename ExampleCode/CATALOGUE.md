# Scalable Latent-Agent Market Simulator — File Catalogue (Originally by Yunfan-Wang@Github)

1. Purpose of this package

This package contains the structural initial-training stage of the latent-agent market simulator.

The initial training is deliberately expensive. It jointly optimizes the centralized Transformer, the configurable population of recurrent participant agents, the learned action gate, and the stochastic market-transition head.

Daily participant fine-tuning and runtime on-board adaptation are separate stages and are not implemented in this package.

2. Files

2.1 `01_full_market_simulator_initial_training_optiver.ipynb`

This is the principal training Notebook.

It performs the following operations:

I. It reads the Optiver Realized Volatility Prediction book and trade partitions.

II. It validates the raw Optiver schema.

III. It resamples each bucket to a one-second grid covering seconds 0 through 599.

IV. It constructs the current 19-feature input schema.

V. It constructs the six next-step transition targets.

VI. It retains 599 valid processed rows per stock and time bucket.

VII. It creates 129-row training windows consisting of 96 warmup positions, 32 supervised positions, and one alignment row.

VIII. It jointly trains the configurable Transformer, K independent recurrent agents, the softmax action gate, and the transition head.

IX. It evaluates the held-out test split.

X. It exports the complete simulator and separate encoder, participant-population, and transition checkpoints.

This file does not perform daily participant recalibration or runtime on-board adaptation.

2.2 `run_initial_training.py`

This is the external execution runtime.

Its responsibilities are:

I. Validate that the Notebook still follows the current Optiver contract.

II. Confirm the 19 features, six targets, 599-row bucket structure, 96-position warmup, 32 supervised positions, and 129-row window.

III. Confirm that the full Transformer-agent architecture remains present.

IV. Reject initial-training Notebooks that contain runtime on-board adaptation functions.

V. Patch deployment-only settings without rewriting the model architecture.

VI. Execute the Notebook through `nbclient`.

VII. Save the patched Notebook, executed Notebook, runtime log, and JSON manifest.

VIII. Record hashes and runtime metadata for reproducibility.

The runtime can override batch size, worker count, epoch count, training-step limits, stock limit, dataset fraction, and `torch.compile`. These are execution settings. It does not change the feature schema, target schema, agent communication structure, or transition design.

2.3 `runtime_config.example.json`

This is a human-readable example of the main runtime settings.

The execution script uses command-line arguments rather than reading this file automatically. External users can copy values from the example when preparing a run.

2.4 `CATALOGUE.md`

This document explains the role and boundary of every file in the package.

3. Required project storage layout

The default Colab project root is:

```text
/content/drive/MyDrive/bigalpha_market_project
```

The runtime expects the following layout:

```text
bigalpha_market_project/
    credentials/
        kaggle.json
    data/
        downloads/
            optiver-realized-volatility-prediction.zip
        processed/
            optiver_1s_six_target_v2/
    checkpoints/
    runs/
```

Only one of the archive or processed dataset is required for an offline run. The processed dataset is preferred when it has already been generated and validated.

4. Environment requirements

The Notebook expects Python 3 and the following principal packages:

```text
numpy
pandas
pyarrow
torch
tqdm
scikit-learn
matplotlib
psutil
kaggle
```

The external runtime additionally requires:

```text
nbformat
nbclient
jupyter-client
ipykernel
```

A CUDA GPU is strongly recommended. The current aggressive configuration was designed around a large-memory GPU. External users should reduce `--batch-size` when using smaller hardware.

5. Validate without training

Run:

```bash
python run_initial_training.py \
  --notebook 01_full_market_simulator_initial_training_optiver.ipynb \
  --project-root /content/drive/MyDrive/bigalpha_market_project \
  --output-dir /content/drive/MyDrive/bigalpha_market_project/runs/validation \
  --offline \
  --validate-only
```

The runtime will produce:

```text
initial_training_patched.ipynb
initial_training_manifest.json
```

No training process will begin.

6. Execute the full structural training run

Run:

```bash
python run_initial_training.py \
  --notebook 01_full_market_simulator_initial_training_optiver.ipynb \
  --project-root /content/drive/MyDrive/bigalpha_market_project \
  --output-dir /content/drive/MyDrive/bigalpha_market_project/runs/initial_training_001 \
  --offline \
  --batch-size 3800 \
  --num-workers 12 \
  --max-epochs 30 \
  --max-train-steps 889 \
  --max-validation-batches 166
```

(it is tested that 3800 works on Nvidia A100 High Ram with memory efficiency of 92%.)

The runtime will produce:

```text
initial_training_patched.ipynb
initial_training_executed.ipynb
initial_training_manifest.json
initial_training_runtime.log
```

The trained checkpoints are saved by the Notebook under the configured project checkpoint directory and under `/content` for the structural module exports.

7. Online and offline modes

7.1 Offline mode

Use `--offline` for competition or controlled execution.

The Optiver archive or the processed dataset must already exist. The runtime prevents a new Kaggle download when the archive is missing.

7.2 Online preparation mode

Omit `--offline` only during authorized data preparation.

The Notebook may use the configured Kaggle credential to download the Optiver archive when it is absent.

Final competition execution should use approved local data and should not depend on network access.

8. Checkpoint responsibilities

The Notebook exports four checkpoint categories.

8.1 Full simulator checkpoint

This contains all model parameters and is used to reproduce the complete structural model.

8.2 Encoder checkpoint

This contains the centralized Transformer market-interpretation layer.

It changes only during a full structural retraining cycle.

8.3 Participant-population checkpoint

This contains the recurrent participant agents and the learned gate.

A later daily fine-tuning workflow can load this checkpoint together with the frozen encoder.

8.4 Transition checkpoint

This contains the six-target stochastic transition head.

It may be updated together with the participant side in later adaptation workflows.

9. Architecture boundary

At one model step:

```text
market and configured external inputs
    -> centralized Transformer
    -> implicit market state h_t
    -> K independent recurrent participant agents
    -> learned softmax weights
    -> weighted aggregate action
    -> six-target stochastic market transition
```

Agents do not communicate directly with one another at the same step.

Their interaction is market-mediated. Their actions contribute to the aggregate transition, and the resulting market state is observed at the next step.

10. Files intentionally not included yet

The following files belong to later development stages:

I. A daily participant-side fine-tuning runtime.

II. A live inference and recursive simulation runtime.

III. A bounded on-board participant adaptation runtime.

IV. A deployment service or API.

V. A firm-specific external-information adapter.

These later files should load the structural checkpoints created by this package. They should not be merged into the initial-training Notebook.
