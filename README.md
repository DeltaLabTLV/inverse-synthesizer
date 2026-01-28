# InverSynth-II Extensions: U-Net MAE + Transformer Contrastive Pretraining

This repository contains a PyTorch implementation of two complementary extensions to **InverSynth II** (Barkan et al., 2023) for automatic synthesizer sound matching:

1. **U-Net Autoencoder (1D Conv along time)** with **MAE-style spectrogram masking** pretraining.
2. **Transformer Encoder/Decoder** with **contrastive pretraining** (InfoNCE) and masked reconstruction training.

The implementation is designed around a **unified STFT dataloader**, multiple training stages (pretrain → finetune), TensorBoard logging, and deterministic debugging utilities.

---

## References

Please cite / reference these works when using this code:

- **InverSynth II**: Barkan et al., 2023. *InverSynth II: ...* (automatic synthesizer sound matching; proxy decoders; inference-time fine-tuning).  
- **Diffusion Beat GAN / diffusion U-Net design principles**: Dhariwal & Nichol, 2021. *Diffusion Models Beat GANs on Image Synthesis*.  
- **Masked Autoencoders (MAE)**: He et al., 2022. *Masked Autoencoders Are Scalable Vision Learners*.  
- **Transformer**: Vaswani et al., 2017. *Attention Is All You Need*.  
- **Contrastive audio representation learning (CLMR-style)**: Spijkervet et al., 2021. *Contrastive Learning of Musical Representations*.  
- **Perceptual / STFT losses** (optional): Yamamoto et al., 2020. *Parallel WaveGAN*; Wright & Välimäki, 2019; Févotte et al., 2009 (Itakura–Saito).

[//]: # (> Refence to the paper.)

---

[//]: # (## Project Structure)

[//]: # (├── configs/)

[//]: # (│ └── experiments.yaml # multi-stage experiment presets)

[//]: # (├── data_stft_unified.py # unified audio→STFT dataset + masking + padding collate)

[//]: # (├── losses.py # InverSynth-II style losses + low-mag weighted loss)

[//]: # (├── unet_1d.py # 1D U-Net autoencoder &#40;time axis conv, freq as channels&#41;)

[//]: # (├── transformer_contrastive.py # transformer encoder + contrastive pretrain &#40;InfoNCE&#41;)

[//]: # (├── full_transformer.py # transformer encoder-decoder for masked reconstruction)

[//]: # (├── train_unified_tb.py # trainer w/ TensorBoard + optional param accuracy)

[//]: # (├── main.py # pipeline runner &#40;pretrain/train/test all stages&#41;)

[//]: # (└── README.md)

---

## Installation
### Create virtual environment
1. Create the environment
conda env create -f environment.yml
2. conda activate inversynth-extensions


[//]: # (### Create environment)

[//]: # (Python 3.10+ recommended.)

[//]: # ()
[//]: # (python -m venv .venv)

[//]: # ()
[//]: # (source .venv/bin/activate   # &#40;Linux/macOS&#41;)

[//]: # ()
[//]: # (pip install -U pip)

## Training Stages

This project supports five main stages plus testing:

✅ 1) Autoencoder pretraining (MAE masking)

Randomly masks 45% of the spectrogram using 5×5 time-frequency patches.

U-Net reconstructs missing regions.

✅ 2) Transformer pretraining (contrastive)

Patchify spectrogram into tokens.

Mask a subset and apply InfoNCE: masked embeddings should match their true targets vs negatives sampled from other tokens in the batch.

✅ 3) Train transformer decoder (freeze encoder)

Load pretrained encoder weights.

Train decoder reconstruction (masked patches).

✅ 4) Train full transformer (encoder + decoder)

Fine-tune end-to-end reconstruction.

Optional: add spectral + perceptual weighting losses if unpatchify() is available.

✅ 5) Autoencoder finetune (no masking)

Load AE pretrained weights.

Train on full spectrogram reconstruction without masking.

✅ 6) Test (autoencoder / transformer)

Evaluates average losses and logs to TensorBoard.

[//]: # (### Running)
## Run everything (pretrain → train → test)
python main.py --config configs/experiments.yaml --run all

Only pretraining
python main.py --config configs/experiments.yaml --run pretrain

Only training (finetune)
python main.py --config configs/experiments.yaml --run train

Only tests
python main.py --config configs/experiments.yaml --run test

Run a single stage preset
python main.py --config configs/experiments.yaml --run stage --stage pretrain_transformer_contrastive

[//]: # (# TensorBoard)

[//]: # ()
[//]: # (The trainer writes TensorBoard logs to:)

[//]: # ()
[//]: # (runs/<experiment_name>/tb/)

[//]: # ()
[//]: # (Logged scalars include &#40;depending on stage&#41;:)

[//]: # ()
[//]: # (train/loss, train/lr)

[//]: # ()
[//]: # (train/contrastive_loss)

[//]: # ()
[//]: # (train/patch_loss, train/total_loss)

[//]: # ()
[//]: # (test/avg_loss, test/avg_patch_mse)

[//]: # ()
[//]: # (optional parameter metrics if provided:)

[//]: # ()
[//]: # (train/param/cont_mse)

[//]: # ()
[//]: # (train/param/cat_acc_mean)

[//]: # ()
[//]: # (train/param/cat_acc/<param_name>)

[//]: # ()
[//]: # (Parameter Accuracy &#40;Optional&#41;)

[//]: # ()
[//]: # (If you want to log synthesizer parameter accuracy like InverSynth II, your dataloader must provide ground truth:)

[//]: # ()
[//]: # (tgt_cont: &#40;B, P_cont&#41; float tensor)

[//]: # ()
[//]: # (tgt_cat: List[Tensor&#40;B,&#41;] or Dict[str, Tensor&#40;B,&#41;])

[//]: # ()
[//]: # (And your model must output predictions:)

[//]: # ()
[//]: # (pred_cont: &#40;B, P_cont&#41;)

[//]: # ()
[//]: # (pred_cat_logits: list/dict of logits for each categorical parameter)

[//]: # ()
[//]: # (The trainer automatically detects and logs:)

[//]: # ()
[//]: # (continuous MSE)

[//]: # ()
[//]: # (categorical accuracy &#40;per parameter + mean&#41;)

[//]: # ()
[//]: # (Reproducible Debugging)

[//]: # ()
[//]: # (To reproduce training issues reliably:)

[//]: # ()
[//]: # (seed global RNGs)

[//]: # ()
[//]: # (seed DataLoader workers)

[//]: # ()
[//]: # (use per-sample deterministic generators for masking)

[//]: # ()
[//]: # (See the recommended seeding snippet in your training scripts and dataset.)

[//]: # ()
[//]: # (Notes & Common Pitfalls)

[//]: # (1&#41; Channel mismatch &#40;U-Net&#41;)

[//]: # ()
[//]: # (U-Net treats frequency bins as channels: input should be &#40;B, F, T&#41;.)

[//]: # (Make sure:)

[//]: # ()
[//]: # (your STFT produces F = n_fft//2 + 1)

[//]: # ()
[//]: # (models.autoencoder_unet.in_channels == F)

[//]: # ()
[//]: # (Example: n_fft=1024 → F=513.)

[//]: # ()
[//]: # (2&#41; Determinism)

[//]: # ()
[//]: # (For strict reproducibility, set:)

[//]: # ()
[//]: # (torch.use_deterministic_algorithms&#40;True&#41;)

[//]: # ()
[//]: # (cudnn.deterministic = True, cudnn.benchmark = False)

[//]: # ()
[//]: # (&#40;Some ops may not have deterministic CUDA kernels; disable strict determinism if needed.&#41;)