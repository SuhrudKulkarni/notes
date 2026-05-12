# Fisher Information again and again

The Fisher Information matrix shows up across optimization, geometry, and statistics. It appears as a preconditioner in natural gradient, a constraint in TRPO, and a lower bound in Cramér-Rao.

## Notation

- $\theta \in \mathbb{R}^d$ are model parameters
- $p_{\theta}(x)$ is the model distribution
- $\ell(\theta) = -\log p_{\theta}(x)$ is the negative log-likelihood
- $g = \nabla_\theta \ell$ is the score (gradient of log-likelihood)
- $H = \nabla^2_\theta \ell$ is the Hessian
- $F = \mathbb{E}[\nabla \log p_{\theta}(x) \cdot \nabla \log p_{\theta}(x)^\top]$ is the Fisher Information matrix, with expectations under $p_\theta$

## §1 Newton's method

Given a loss $\ell(\theta)$, Newton's method takes a quadratic Taylor expansion around $\theta$:

$$\ell(\theta + \Delta) \approx \ell(\theta) + g^\top \Delta + \tfrac{1}{2} \Delta^\top H \Delta$$

Minimizing over $\Delta$ gives

$$\Delta^\ast = -H^{-1} g$$

which has quadratic convergence near a minimum. This is useful for convex problems but runs into issues for non-convex ones, where $H$ might not be PSD and the Newton step might not point downhill. Forming and inverting $H$ is also intractable at scale. Classical algorithms make Newton tractable by approximating the inverse, like the quasi-Newton method BFGS, or by constraining the step, like trust-region methods including Levenberg-Marquardt.

## §2 Gauss-Newton

For negative log-likelihood losses, the expected Hessian equals the Fisher Information matrix under the model:

$$\mathbb{E}[H] = \mathbb{E}[\nabla \log p \cdot \nabla \log p^\top] = F$$

Gauss-Newton replaces $H$ in the Newton step with the empirical version of the right-hand side. The substitute is PSD by construction, and forming it only requires gradients rather than second derivatives.

## §3 Natural gradient

If we want a metric on distributions instead of on parameters, a common choice is the KL divergence $D_{KL}(p_{\theta} \| p_{\theta + \Delta})$. Its second-order Taylor expansion gives:

$$D_{KL}(p_{\theta} \| p_{\theta + \Delta}) \approx \tfrac{1}{2} \Delta^\top F \Delta$$

So $F$ is the metric tensor of the manifold of distributions $\{p_{\theta}\}$, and the natural gradient is steepest descent in this geometry:

$$\Delta_{\text{nat}} = -F^{-1} g$$

[Jake Tae's post on natural gradient](https://jaketae.github.io/study/natural-gradient/) covers the Taylor-expansion derivation and the constrained-optimization framing.

The first-order term in the Taylor expansion vanishes because of the score identity $\mathbb{E}[\nabla \log p_{\theta}] = 0$ under $p_\theta$, the same identity that cancels the baseline out of the policy gradient in [Post 1](1-variance-reduction.md).

KL is one member of the f-divergence family, all of which have the same second-order Taylor expansion $\tfrac{1}{2} \Delta^\top F \Delta$ up to a constant factor. The choice of divergence sets higher-order behavior, but the metric tensor is $F$ either way.

## §4 TRPO

In TRPO (trust-region policy optimization) we take the largest gradient step we can while keeping the new policy within $\delta$ KL of the old:

$$\max_{\Delta} g^\top \Delta \quad \text{s.t.} \quad D_{KL}(\pi_\theta \| \pi_{\theta + \Delta}) \leq \delta$$

By substituting the second-order expansion $D_{KL} \approx \tfrac{1}{2} \Delta^\top F \Delta$, the constraint becomes quadratic, but objective stays linear. Then the solution is $\Delta \propto F^{-1} g$.

TRPO and Levenberg-Marquardt are both trust-region methods that bound the step to stay where the quadratic approximation holds. The metric is swapped from Euclidean ($\|\Delta\|_2 \leq \delta$) to KL ($\tfrac{1}{2} \Delta^\top F \Delta \leq \delta$), so the same $F$ that multiplies the gradient in natural gradient shows up here as the constraint matrix.

## §5 Adam

Adam's update is

$$\Delta = -\eta \frac{m}{\sqrt{v} + \epsilon}$$

where $m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$ and $v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$.

The $v$ term is an exponential-moving-average estimate of the diagonal of $\mathbb{E}[g g^\top]$, which is the same outer product of scores from the Hessian decomposition in §2 and equals the Fisher in expectation. Dividing by $\sqrt{v}$ scales the gradient by the diagonal of an approximate inverse Fisher.

Adam, thus, looks like the natural gradient simplified with only the diagonal of $F$, and a square-root in place.

## §6 Evolution strategies

NES is often used for black-box optimization, where $\ell(\theta)$ is available but $\nabla \ell$ is not, like in a non-differentiable simulator or a real-world experiment. We put a search distribution $p_{\phi}(\theta)$ over parameters, sample $\theta_i \sim p_{\phi}$, evaluate $\ell(\theta_i)$, and update $\phi$ to shift the search distribution toward lower-loss regions.

The gradient of the expected loss in $\phi$ uses the score-function trick:

$$\nabla_\phi \mathbb{E}_{\theta \sim p_{\phi}}[\ell(\theta)] = \mathbb{E}_{\theta \sim p_{\phi}}[\ell(\theta) \nabla_\phi \log p_{\phi}(\theta)]$$

This is the same **REINFORCE**-style identity from [Post 1](1-variance-reduction.md), applied to parameters sampled from a search distribution instead of actions sampled from a policy.

Multiplying that gradient by $F^{-1}$, where $F$ is the Fisher Information of $p_{\phi}$, yields the natural gradient.

## §7 Cramér-Rao

We also see it in statistics. For any unbiased estimator $\hat\theta$ of a parameter $\theta$:

$$\mathrm{Cov}(\hat\theta) \succeq F^{-1} $$

The size of $F$ along a parameter direction measures how informative the data is about that direction.

It also shows up in the score test, where the test statistic $g^\top F^{-1} g$ uses $F$ as the same metric on parameter space that natural gradient uses to scale the step.

## References

- **Amari 1998**, "Natural Gradient Works Efficiently in Learning" (*Neural Computation*).
- **Amari & Nagaoka**, *Methods of Information Geometry*.
- **Schulman, Levine, Moritz, Jordan, Abbeel 2015**, "Trust Region Policy Optimization" (ICML).
- **Kingma & Ba 2014**, "Adam: A Method for Stochastic Optimization" (arXiv:1412.6980).
- **Wierstra, Schaul, Glasmachers, Sun, Peters, Schmidhuber 2014**, "Natural Evolution Strategies" (JMLR).
- **Casella & Berger**, *Statistical Inference* (Cramér-Rao bound).
