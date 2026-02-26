# MINE — Mutual Information Neural Estimators

Implementation and validation of **MINE** (Mutual Information Neural Estimation) from [Belghazi et al., 2018](https://arxiv.org/abs/1801.04062). MINE uses a neural network to compute a tight lower bound on mutual information between arbitrary random variables, without any assumptions about their distributions.

## Background

Mutual information $I(X; Y)$ measures how much knowing $X$ reduces uncertainty about $Y$. It's a fundamental quantity in information theory but notoriously hard to estimate in high dimensions. MINE gets around this by optimising the **Donsker–Varadhan (DV) lower bound**:

$$I(X; Y) \geq \sup_{T: \Omega \to \mathbb{R}} \; \mathbb{E}_{p(x,y)}[T_\theta(x,y)] - \log \mathbb{E}_{p(x)p(y)}\!\left[e^{T_\theta(x,y)}\right]$$

A neural network $T_\theta$ is trained to maximise this bound. Joint samples $(x, y) \sim p(x,y)$ are drawn directly from the data; marginal samples are formed by shuffling $y$ independently of $x$ to break the dependence.

## Notebooks

### `MINE_check.ipynb` — Validation on Bivariate Gaussians

Verifies that MINE recovers the correct MI for bivariate Gaussians where the ground truth is known analytically:

$$I(X; Y) = -\frac{1}{2} \log(1 - \rho^2)$$

Runs MINE for correlation $\rho = 0.8$ and plots the estimated MI against the true value over training.

### `MINE_XOR.ipynb` — MINE on the XOR Problem

Applies MINE to the XOR dataset from [`torch-explain`](https://github.com/pietrobarbiero/pytorch_explain). The true MI between the XOR inputs and its label is exactly $1$ bit $= \ln 2 \approx 0.693$ nats. The notebook trains MINE and tracks convergence to this ground truth.

## Setup

```bash
pip install -r requirements.txt
jupyter notebook
```

## Results

| Experiment | True MI (nats) | MINE estimate |
|------------|:--------------:|:-------------:|
| Bivariate Gaussian (ρ = 0.8) | 0.5108 | ~0.50 |
| XOR | 0.6931 (ln 2) | ~0.69 |

