# IFT6135 Winter 2025 — Assignment 3

## Variational Autoencoders, DDPMs, and Classifier-Free Guidance

## Project goal

This project develops three generative-modeling methods for MNIST images with PyTorch:

1. a **Variational Autoencoder (VAE)** that learns a probabilistic latent representation;
2. a **Denoising Diffusion Probabilistic Model (DDPM)** that generates digits by reversing a gradual noising process; and
3. a **Classifier-Free Guidance (CFG)** diffusion model that generates digits conditioned on their class labels.

The main goal is to connect the mathematical foundations of latent-variable and diffusion models to working implementations. The project covers likelihoods, KL divergence, the VAE evidence lower bound, forward and reverse diffusion, noise-prediction training, exponential moving averages, conditional generation, and guidance-scale interpolation.

By the end of the project, the models should reconstruct or generate recognizable MNIST digits, and the report should explain how image quality changes during training and throughout the reverse diffusion process.

## End-to-end workflow

```text
Install dependencies and select a device
                    ↓
Implement VAE probability utilities and ELBO loss
                    ↓
Train the VAE on MNIST
                    ↓
Implement DDPM forward diffusion, reverse diffusion, and loss
                    ↓
Train a time-conditioned U-Net to predict noise
                    ↓
Generate unconditional digits from Gaussian noise
                    ↓
Implement the continuous noise schedule and CFG sampler
                    ↓
Train with conditional-label dropout
                    ↓
Generate label-controlled digits and compare guidance scales
                    ↓
Save figures, analyze results, and complete the report
```

## Repository structure

```text
IFT6135-Assignment3-H25-main/
├── README.md                    # Project goal and workflow
├── requirements.txt             # Python dependencies
├── q1_vae.py                    # Likelihood, log-mean-exp, and KL utilities
├── q1_train_vae.py              # MNIST VAE model and training loop
├── q2_ddpm.py                   # DDPM forward/reverse process and loss
├── q2_trainer_ddpm.py           # DDPM training, EMA, sampling, and plots
├── 02 - DDPM.ipynb              # Complete DDPM workflow notebook
├── ddpm_utils/
│   ├── args.py                  # DDPM configuration
│   ├── dataset.py               # Unconditional MNIST dataset
│   └── unet.py                  # Time-conditioned U-Net
├── q3_cfg_diffusion.py          # CFG diffusion schedule, process, and loss
├── q3_trainer_cfg.py            # Conditional training and guided sampling
├── 03 - ClassifierFreeGuidance.ipynb
├── cfg_utils/
│   ├── args.py                  # CFG configuration
│   ├── dataset.py               # MNIST images and labels
│   └── unet.py                  # Label-conditioned U-Net
├── images/                      # Diagrams and generated samples
└── *_old.py, __q2_ddpm.py       # Older/reference copies; not primary files
```

Use `q2_ddpm.py`, `q2_trainer_ddpm.py`, `q3_cfg_diffusion.py`, and `q3_trainer_cfg.py` as the primary implementations. The README originally referred to nonexistent `q2_cfg_*` files; the actual Question 3 files use the `q3_*` names shown above.

## Step-by-step project guide

### 1. Create the Python environment

From the project directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Check PyTorch and device availability:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

A GPU is strongly recommended for full DDPM and CFG training. Use CPU for short mathematical and shape checks.

### 2. Understand the MNIST pipeline

All three tasks use MNIST:

- the VAE uses the original `28 × 28` images and flattens them to 784 values;
- the DDPM utilities resize images to `32 × 32` and return images only; and
- the CFG utilities resize images to `32 × 32` and retain digit labels.

The datasets download automatically on the first run. Generated data and trained weights should remain outside version control.

### 3. Complete the VAE probability functions

In `q1_vae.py`, implement and verify:

1. `log_likelihood_bernoulli` for the reconstruction distribution;
2. `log_likelihood_normal` for diagonal Gaussian densities;
3. `log_mean_exp` using a numerically stable maximum shift;
4. analytic `KL(q || p)` between diagonal Gaussians; and
5. a Monte Carlo estimate of the same KL divergence.

Important numerical details:

- flatten every example while preserving the batch dimension;
- sum log-probabilities over feature dimensions;
- clamp Bernoulli probabilities away from 0 and 1;
- use `logvar` consistently as the logarithm of variance; and
- create random samples on the same device and with the same dtype as the model tensors.

### 4. Verify the VAE mathematics

Before full training, test basic properties in a Python session:

```python
import torch
from q1_vae import (
    log_likelihood_bernoulli,
    log_likelihood_normal,
    log_mean_exp,
    kl_gaussian_gaussian_analytic,
    kl_gaussian_gaussian_mc,
)

mu = torch.zeros(8, 20)
logvar = torch.zeros_like(mu)

assert torch.allclose(
    kl_gaussian_gaussian_analytic(mu, logvar, mu, logvar),
    torch.zeros(8),
    atol=1e-6,
)

y = torch.randn(8, 10)
assert log_mean_exp(y).shape == (8,)
print("VAE utility checks passed")
```

The Monte Carlo KL estimate should approach the analytic value as `num_samples` increases, although it will not match exactly because of sampling noise.

### 5. Build and train the VAE

`q1_train_vae.py` defines:

- a 784-to-400 encoder;
- 20-dimensional posterior mean and log-variance vectors;
- reparameterized latent samples;
- a 400-to-784 Bernoulli decoder; and
- the negative ELBO training objective.

The loss is:

```text
negative ELBO = negative reconstruction log-likelihood + KL(q(z|x) || p(z))
```

Run a short smoke test:

```bash
python q1_train_vae.py --epochs 1 --batch-size 128
```

Then train the requested configuration:

```bash
python q1_train_vae.py --epochs 10 --batch-size 128
```

Use `--no-cuda` to force CPU execution. The script saves the trained model as `model.pt`.

### 6. Understand DDPM forward diffusion

Question 2 uses a linear variance schedule:

```text
βₜ from 0.0001 to 0.02
αₜ = 1 − βₜ
ᾱₜ = ∏ₛ₌₁ᵗ αₛ
```

In `q2_ddpm.py`, implement:

1. `gather` to broadcast timestep coefficients across an image batch;
2. `q_xt_x0` to compute the mean and variance of `q(x_t | x_0)`; and
3. `q_sample` to create a noisy image directly at an arbitrary timestep.

The forward sample is:

```text
xₜ = √ᾱₜ x₀ + √(1 − ᾱₜ) ε,    ε ~ N(0, I)
```

Verify that `t=0` leaves substantial image structure and a late timestep produces an image close to Gaussian noise.

### 7. Implement DDPM reverse sampling

Complete the reverse-process methods:

1. use the U-Net to predict noise `εθ(x_t, t)`;
2. compute the reverse Gaussian mean in `p_xt_prev_xt`;
3. use `β_t` as the configured reverse variance;
4. sample `x_{t-1}` in `p_sample`; and
5. do not add random noise at the final denoising step.

Check the following for a batch of images:

- the returned mean and variance broadcast correctly;
- `p_sample` preserves image shape;
- all coefficients are on the selected device; and
- fixed seeds reproduce debugging samples.

### 8. Implement the DDPM noise-prediction loss

For each batch:

1. sample a timestep independently for each image;
2. sample Gaussian noise;
3. construct `x_t` with the forward process;
4. predict the noise with the time-conditioned U-Net; and
5. minimize squared error between true and predicted noise.

Be consistent about reduction. The current implementation sums over image dimensions and averages over the batch; changing this to a full mean changes the reported loss scale.

### 9. Run the DDPM notebook

The main executable workflow for Question 2 is `02 - DDPM.ipynb`. Start Jupyter:

```bash
jupyter notebook
```

Then open the notebook and run cells in order. It:

1. imports the DDPM configuration, dataset, and U-Net;
2. defines or loads the diffusion and trainer classes;
3. creates `UNet(c_in=1, c_out=1)`;
4. creates a `DenoiseDiffusion` instance;
5. loads MNIST;
6. trains the model; and
7. generates samples and intermediate denoising images.

Default DDPM settings in `ddpm_utils/args.py` include 1,000 diffusion steps, batch size 256, learning rate `2e-4`, and 20 epochs.

### 10. Understand the DDPM trainer

`q2_trainer_ddpm.py` provides:

- Adam optimization;
- optional automatic mixed precision;
- a StepLR learning-rate schedule;
- exponential moving average (EMA) weights;
- checkpoint saving;
- reverse-process sampling; and
- intermediate-image visualization.

During training, confirm that:

- loss decreases over time;
- recognizable digit structure begins to appear;
- images are saved in `images/`;
- checkpoints are written to the configured `MODEL_PATH`; and
- the reverse sequence becomes progressively less noisy.

The trainer maintains an EMA model, but sampling currently calls `self.diffusion`, whose noise predictor is the original model. If the assignment requires EMA sampling, explicitly connect the diffusion sampler to `ema_model` and document that choice.

### 11. Generate DDPM intermediate samples

After training, use `generate_intermediate_samples` to capture selected reverse steps. Include at least:

- initial Gaussian noise;
- one or more middle steps; and
- the final generated images.

Use the same seed when comparing sampling behavior. Do not accumulate gradients during generation; run under `torch.no_grad()` or detach tensors between steps.

### 12. Understand classifier-free guidance

Question 3 adds class conditioning without training a separate classifier. During training, labels are randomly dropped for part of the batches. The same U-Net therefore learns both:

- conditional noise prediction `εθ(z_t, y)`; and
- unconditional noise prediction `εθ(z_t, ∅)`.

At sampling time these predictions are combined:

```text
ε̂ = (1 + w) εθ(z_t, y) − w εθ(z_t, ∅)
```

where `w` is the guidance scale. Larger guidance can strengthen label consistency but may reduce sample diversity or introduce artifacts.

### 13. Implement the CFG diffusion schedule

In `q3_cfg_diffusion.py`, complete and verify:

1. conversion from discrete timestep `t` to normalized time `u`;
2. the bounded log signal-to-noise schedule `λ(t)`;
3. `α(λ)` and `σ(λ)`;
4. forward sampling `z_λ = α(λ)x + σ(λ)ε`;
5. transition standard deviations/variances;
6. the reverse-process mean and variance;
7. reverse sampling; and
8. the noise-prediction loss.

Keep the distinction between variance and standard deviation explicit. A method named `sigma_*` should have a clearly documented return type, and `p_sample` should apply the square root exactly once when converting variance to sampling scale.

### 14. Complete CFG training and guided sampling

`q3_trainer_cfg.py` trains `UNet_conditional`. The workflow is:

1. load images and labels;
2. move both to the selected device;
3. replace labels with `None` for approximately 10% of batches;
4. optimize the noise-prediction objective;
5. predict conditional and unconditional noise during sampling;
6. interpolate the predictions with `cfg_scale`;
7. estimate the clean image; and
8. sample the next reverse-process state.

Ensure generated labels cover `0` through `9`. The current random-label expression uses an exclusive upper bound of 9, which omits digit 9; use an upper bound of 10 when labels are generated automatically.

### 15. Run the CFG notebook

Open and execute:

```text
03 - ClassifierFreeGuidance.ipynb
```

The notebook:

1. imports the CFG configuration and conditional U-Net;
2. creates `UNet_conditional(c_in=1, c_out=1, num_classes=10)`;
3. builds the CFG diffusion process;
4. trains on labeled MNIST; and
5. generates label-conditioned samples.

Default settings in `cfg_utils/args.py` include 1,000 steps, batch size 256, learning rate `2e-4`, 16 epochs, 10 classes, and a configurable guidance scale.

### 16. Compare guidance scales

Generate the same labels and seeds at several guidance scales, for example:

```text
w ∈ {0.0, 0.3, 1.0, 3.0, 5.0}
```

For each value, assess:

- whether the requested digit class is recognizable;
- sharpness and visual quality;
- diversity among samples of the same class; and
- instability or artifacts at high guidance.

Keep the trained checkpoint, labels, starting noise, and number of reverse steps fixed so guidance scale is the only changing variable.

### 17. Record results for the report

Use a table like this for every important run:

| Run | Model | Epochs | Steps | Batch size | Learning rate | Guidance | Final loss | Output/checkpoint | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| 1 | DDPM | 20 | 1000 | 256 | 2e-4 | — | — | — | — |

The report should include:

- the implemented equations and key design decisions;
- VAE training behavior;
- DDPM samples across epochs;
- intermediate DDPM reverse-process images;
- an explanation of why CFG is “classifier-free” and provides “guidance”;
- the classifier-guidance alternative and how its objective differs;
- CFG samples across epochs;
- a controlled guidance-scale comparison; and
- failure cases, limitations, and conclusions.

## Recommended validation checklist

Before a long training run:

- verify every output shape;
- test timestep 0 and the final timestep;
- verify CPU/GPU device consistency;
- confirm losses are finite;
- confirm causal sampling formulas do not produce NaNs;
- run one small batch through forward and backward passes;
- test sampling with a small number of reverse steps; and
- verify the output directory exists.

After training:

- reload the saved checkpoint;
- generate samples with a fixed seed;
- inspect samples rather than relying only on loss;
- preserve representative and failed outputs;
- label every figure with epoch, timestep, model, and guidance scale; and
- record the exact configuration used.

## Common pitfalls

- Running 1,000-step sampling before completing small smoke tests.
- Confusing variance with standard deviation in Gaussian sampling.
- Adding noise during the final reverse step.
- Using coefficients located on CPU with images located on GPU.
- Enabling FP16 on an unsupported device.
- Sampling from the current model while assuming the EMA model is being used.
- Forgetting to switch the model back to training mode after sampling.
- Omitting digit 9 when randomly generating labels.
- Comparing guidance scales with different initial noise or labels.
- Using older files such as `q3_*_old.py` instead of the primary implementations.
- Committing MNIST data, checkpoints, generated images, or a personal report to Git.

## Definition of done

The project is complete when:

- all VAE likelihood and KL functions pass mathematical and shape checks;
- the VAE trains with a correct negative-ELBO objective;
- DDPM forward and reverse equations are implemented correctly;
- the DDPM noise predictor trains and produces recognizable digits;
- intermediate reverse-process images are saved and explained;
- the CFG schedule and reverse sampler are correct;
- conditional and unconditional predictions are combined correctly;
- labels 0–9 can be generated intentionally;
- guidance-scale effects are compared under controlled conditions; and
- the report contains reproducible settings, figures, analysis, and conclusions.
