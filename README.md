# MINE — Mutual Information Neural Estimators

Notebooks implementing and validating **MINE** (Mutual Information Neural Estimation), the method introduced by [Belghazi et al. (2018)](https://arxiv.org/abs/1801.04062).

## Notebooks

| Notebook | Description |
|----------|-------------|
| `MINE_check.ipynb` | Validates MINE against the analytical MI of bivariate Gaussians with known correlation ρ |
| `MINE_XOR.ipynb` | Applies MINE to the XOR dataset, estimating MI between inputs and labels (ground truth: 1 bit = ln 2 nats) |

## Method

MINE trains a neural network *T* to maximise the Donsker–Varadhan lower bound on mutual information:

$$I(X;Y) \geq \mathbb{E}_{p(x,y)}[T] - \log \mathbb{E}_{p(x)p(y)}[e^T]$$

Joint samples are drawn from the true distribution; marginal samples are formed by shuffling *Y* independently of *X*.

## Dependencies

```bash
pip install torch torchvision numpy matplotlib scikit-learn torch-explain
```
