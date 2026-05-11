# Variance reduction and GAE

The same variance reduction trick shows up under different names. Monte Carlo calls it the *control variate*, trading calls it the *minimum-variance hedge*, quant finance calls it *excess return*, and policy-gradient RL calls it the *baseline-corrected advantage*. We show the trick once, then apply it to RL: baselines, GAE, GRPO.

The post is based on reading and toy problems.

## §1 A simple formula

I have a random variable $X$ with more variance than I'd like, and a correlated random variable $Y$ with known mean $\mu_Y$. Let:

$$X' = X - \beta(Y - \mu_Y)$$

For any $\beta$, $\mathbb{E}[X'] = \mathbb{E}[X]$, so still unbiased. The variance:

$$\mathrm{Var}(X') = \mathrm{Var}(X) - 2\beta \mathrm{Cov}(X, Y) + \beta^2 \mathrm{Var}(Y)$$

Minimize over $\beta$ and:

$$\beta^\ast = \frac{\mathrm{Cov}(X, Y)}{\mathrm{Var}(Y)}, \qquad \mathrm{Var}(X')\big|_{\beta^\ast} = \mathrm{Var}(X) (1 - \rho_{XY}^2)$$

When $|\rho_{XY}| = 1$, variance goes to zero. When $\rho_{XY} = 0$, no improvement.

## §2 Where it shows up

**Monte Carlo (control variates).** $X$ is a noisy estimator, $Y$ is a related random variable with a known mean. Art Owen's *Monte Carlo theory, methods and examples* covers the full variance-reduction family in Ch. 8 and importance sampling in Ch. 9.

**Trading (minimum-variance hedge).** $X$ is the position's P&L, $Y$ is a hedge instrument's return. $\beta^\ast$ is the minimum-variance hedge ratio, or DV01 in fixed income. $\beta$ is usually re-estimated on a rolling window, and good hedging also weighs tail risk, margin, basis risk, and liquidity.

**Quant finance (excess return).** Here the motivation is attribution, but variance reduction is a side effect. **Excess return**: return minus a chosen reference. The reference depends on what we're controlling for:

- Excess over the risk-free rate: separates return that requires capital from return that doesn't
- Excess over the market: CAPM, strips out broad market beta
- Excess over a factor model: Fama-French and descendants, strips out exposure to value, momentum, size, whatever else.
- Excess over a benchmark portfolio: active management, strips out whatever the benchmark already captures.

We still have $X - \beta^\ast(Y - \mu_Y)$, with $Y$ the reference and $\beta^\ast$ the regression coefficient minimizing residual variance.

The rest of the post is about the policy-gradient version.

## §3 RL quick definitions

$S_t, A_t, R_t$ are state, action, and immediate reward at time $t$. $V_\phi$ is the learned critic, an approximation to $V^\pi$. $\hat{X}$ denotes a sample-based estimator of $X$.

**Return** $G_t$: the discounted sum of immediate rewards from $t$ onward along one trajectory:

$$G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k}$$

with the recursive form $G_t = R_t + \gamma G_{t+1}$.

**State-value function** $V^\pi(s)$: deterministic in $s$ for fixed $\pi$:

$$V^\pi(s) = \mathbb{E}_{\pi}[G_t \mid S_t = s]$$

The expectation is over actions sampled from $\pi$ and environment transitions to the next state.

**Action-value function** $Q^\pi(s, a)$: conditioned on state and action:

$$Q^\pi(s, a) = \mathbb{E}_{\pi}[G_t \mid S_t = s, A_t = a]$$

Expectation is over environment transitions and future actions.

$$V^\pi(s) = \mathbb{E}_{A \sim \pi(\cdot|s)}[Q^\pi(s, A)]$$
$$Q^\pi(s, a) = \mathbb{E}[R_t + \gamma V^\pi(S_{t+1}) \mid S_t = s, A_t = a]$$

<details>
<summary>Derivation: recursion + Markov property</summary>

Using $G_t = R_t + \gamma G_{t+1}$:

$$Q^\pi(s, a) = \mathbb{E}[G_t \mid S_t = s, A_t = a] = \mathbb{E}[R_t + \gamma G_{t+1} \mid S_t = s, A_t = a]$$

Condition on $S_{t+1}$ with the tower property:

$$\mathbb{E}[G_{t+1} \mid S_t, A_t] = \mathbb{E}[\mathbb{E}[G_{t+1} \mid S_t, A_t, S_{t+1}] \mid S_t, A_t]$$

Markov: $\mathbb{E}[G_{t+1} \mid S_t, A_t, S_{t+1}] = \mathbb{E}[G_{t+1} \mid S_{t+1}] = V^\pi(S_{t+1})$.

</details>

## §4 Learning $Q^\pi$ directly is expensive

The policy gradient needs $Q^\pi(S_t, A_t)$ at every step. Training $Q_\phi(s, a)$ directly over $\mathcal{S} \times \mathcal{A}$ needs coverage of every $(s, a)$ pair. On-policy methods avoid this by using a sampled $G_t$ as a one-sample estimator. The cost is high variance from the single sample.

## §5 Policy gradient

> [!NOTE]
> **The §1 control variate, in RL form: $X = g(A_t, S_t) \cdot G_t$, $Y = g(A_t, S_t)$, $\mu_Y = 0$.** Subtracting $\beta Y$ gives the baseline-corrected policy gradient $g(A_t, S_t) \cdot (G_t - \beta)$.

The score-function form of the policy gradient theorem:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi}[ \sum_t g(A_t, S_t) \cdot Q^\pi(S_t, A_t) ], \qquad g(a, s) = \nabla_\theta \log \pi_\theta(a \mid s)$$

The on-policy estimator replaces the inner expectation with a one-sample $G_t$:

$$\hat{\nabla}_\theta J = \sum_t g(A_t, S_t) \cdot G_t$$

<details>
<summary>Why this is unbiased</summary>

By the §3 identity, $\mathbb{E}_{\pi}[G_t \mid S_t, A_t] = Q^\pi(S_t, A_t)$, where the expectation is over the trajectory continuation. Apply the tower property:

$$\mathbb{E}\_{\pi}[g(A_t, S_t) \cdot G_t] = \mathbb{E}\_{\pi}\big[g(A_t, S_t) \cdot \mathbb{E}\_{\pi}[G_t \mid S_t, A_t]\big]$$

Substitute the identity:

$$= \mathbb{E}_{\pi}[g(A_t, S_t) \cdot Q^\pi(S_t, A_t)]$$

So the expectation matches the policy gradient theorem and a single rollout gives an unbiased estimate.

</details>

This is the §1 setup with:

$$X = g(A_t, S_t) \cdot G_t, \qquad Y = g(A_t, S_t), \qquad \mu_Y = 0$$

$\mu_Y = 0$ because:

$$\mathbb{E}_{A \sim \pi}[g(A, s)] = \int \nabla_\theta \pi_\theta(a \mid s)  da = \nabla_\theta \int \pi_\theta(a \mid s)  da = \nabla_\theta 1 = 0$$

Plug into $X' = X - \beta(Y - \mu_Y)$:

$$X' = g(A_t, S_t) \cdot G_t - \beta \cdot g(A_t, S_t) = g(A_t, S_t) \cdot (G_t - \beta)$$

This is the baseline-corrected policy gradient. Since the zero-mean property holds conditional on $S_t$, any state-dependent $b(S_t)$ keeps the gradient unbiased. With $\beta = 1$ and state-dependent $b$, the standard form is $g(A_t, S_t) \cdot (G_t - b(S_t))$. RL calls $b$ the baseline.

## §6 Baselines

**Unbiasedness.** Subtracting a baseline $b(S_t)$ that doesn't depend on $A_t$ doesn't change the gradient's expectation. Same argument as §5: the score has zero mean conditional on $S_t$, so any state-dependent function works as $\beta$.

**The optimal $b$.** Condition on $S_t = s$. The variance of the baseline-corrected estimator is:

$$\mathrm{Var}[g \cdot (G_t - b(s)) \mid s] = \mathrm{Var}(g \cdot G_t \mid s) - 2 b(s) \mathrm{Cov}(g \cdot G_t, g \mid s) + b(s)^2 \mathrm{Var}(g \mid s)$$

Quadratic in $b(s)$, opening upward (since $\mathrm{Var}(g \mid s) > 0$). The minimizer:

$$b^\ast(s) = \frac{\mathrm{Cov}(g \cdot G_t, g \mid s)}{\mathrm{Var}(g \mid s)}$$

<details>
<summary>Greensmith et al. write this as $\mathbb{E}[g^2 G_t \mid s] / \mathbb{E}[g^2 \mid s]$, which is equivalent to the Cov/Var form.</summary>

$\mathrm{Cov}(g G_t, g \mid s) = \mathbb{E}[g^2 G_t \mid s] - \mathbb{E}[g G_t \mid s] \cdot \mathbb{E}[g \mid s]$ and $\mathrm{Var}(g \mid s) = \mathbb{E}[g^2 \mid s] - \mathbb{E}[g \mid s]^2$.

The score identity gives $\mathbb{E}[g \mid s] = 0$, so the cross-terms drop out: $\mathrm{Cov}(g G_t, g \mid s) = \mathbb{E}[g^2 G_t \mid s]$ and $\mathrm{Var}(g \mid s) = \mathbb{E}[g^2 \mid s]$.

</details>

For $b(s) \in (0, 2 b^\ast(s))$ at every state, variance is strictly less than at $b = 0$.

### Plugging in $V^\pi$

The variance-optimal choice is $b^\ast(s)$ with the variance-minimizing $\beta^\ast$. However, the usual choice is $b(s) = V^\pi(s)$ with $\beta = 1$, giving the **advantage function** $A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$ and its sample estimator:

$$\hat{A}^\pi_t = G_t - V_\phi(S_t)$$

where $V_\phi$ is the learned approximation to $V^\pi$ and $G_t$ is the one-sample estimate of $Q^\pi$ via the §3 identity.

This isn't variance-optimal, since $V^\pi$ isn't $b^\ast$, the §1 $\beta$ is 1 instead of $\mathrm{Cov}(X, Y)/\mathrm{Var}(Y)$, and estimating either properly needs $\mathrm{Cov}(g G_t, g)$ from the gradient's own trajectory data, which introduces correlation and bias. People default to the good-enough version, since $V^\pi$ is already learned for TD bootstrapping and GAE.

## §7 The on-policy constraint

When an estimator is noisy, we collect more data, but on-policy RL can't. Each rollout comes from the current $\pi_\theta$, which we update. After a gradient step, prior rollouts came from the old policy and cover the wrong states. So we can't drive variance down with more samples. The post-gradient policy needs its own rollouts, and we have to find variance reductions that work within a single rollout: bootstrapping with $V_\phi$, truncating the trajectory, blending across $n$.

## §8 A stack of approximations from $Q^\pi$ to $\hat{A}^{\pi,(n)}_t$

> [!NOTE]
> **By the time we reach the n-step advantage estimator, we're several approximations away from the policy-gradient theorem.** And we can't fix this by averaging IID samples, because the $\hat{A}^\pi_t$ within one rollout are correlated.

Each step below replaces an expectation or an exact quantity with something cheaper, at the cost of bias, variance, or correlation:

1. The policy-gradient theorem from §5: $\nabla_\theta J = \mathbb{E}[\sum_t g(A_t, S_t) \cdot Q^\pi(S_t, A_t)]$, where $g(a, s) = \nabla_\theta \log \pi_\theta(a \mid s)$ is the score function. Both $g$ and $Q^\pi$ are inside an expectation we can't compute.
2. Replace $Q^\pi(S_t, A_t)$ with the one-sample return $G_t$, which is unbiased given $(s, a)$ via the §3 identity but high variance.
3. Subtract a baseline. The variance-optimal version is $g(A_t, S_t) \cdot \beta^\ast(G_t - b^\ast(S_t))$. Both $b^\ast$ and $\beta^\ast$ depend on quantities ($\mathrm{Cov}(g G_t, g)$, $\mathrm{Var}(g)$) we'd have to estimate from the same trajectory data.
4. Replace $b^\ast(s)$ with $V^\pi(s)$. This is unbiased and variance-reducing if $V^\pi$ lands in $(0, 2b^\ast)$, but not variance-minimizing.
5. Use $\beta = 1$ instead of the variance-minimizing $\mathrm{Cov}(g G_t, g) / \mathrm{Var}(g)$, which avoids estimating $\beta$ from the same data.
6. **The estimator is now $g(A_t, S_t) \cdot (G_t - V^\pi(S_t))$, where $G_t - V^\pi$ is a one-sample estimate of the advantage $A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$.**
7. Replace $V^\pi$ with the learned critic $V_\phi$, which introduces critic bias and critic variance.
8. Truncate the trajectory at $n$ steps and use $V_\phi(S_{t+n})$ for the tail, trading trajectory variance for $\gamma^n$-weighted critic error.

## §9 GAE

GAE (Generalized Advantage Estimation) is built on n-step returns: $n$ real rewards along the trajectory, then $V_\phi(S_{t+n})$ as a stand-in for the rest.

$$G_t^{(n)} = R_t + \gamma R_{t+1} + \cdots + \gamma^{n-1} R_{t+n-1} + \gamma^n V_\phi(S_{t+n})$$

The n-step advantage estimator:

$$\hat{A}^{\pi,(n)}_t = G_t^{(n)} - V_\phi(S_t)$$

The TD residual:

$$\delta_t = R_t + \gamma V_\phi(S_{t+1}) - V_\phi(S_t)$$

GAE is an exponentially-weighted average of n-step advantage estimators. Equivalently, a sum of TD residuals:

$$\hat{A}^{\pi,\text{GAE}(\gamma, \lambda)}_t = \sum_{k=0}^{\infty} (\gamma\lambda)^k \delta_{t+k}$$

$\lambda = 0$ gives $\hat{A}^\pi_t = \delta_t$ (one-step TD), $\lambda = 1$ gives $\hat{A}^\pi_t = G_t - V_\phi(S_t)$ (Monte Carlo), and intermediate $\lambda$ blends them.

## §10 Three sources of error

The GAE paper uses $\lambda$ to trade bias against variance. The standard supervised-learning decomposition splits error into three parts: bias, variance, and irreducible noise. Applied to the advantage estimator:

$$\mathbb{E}[(\hat{A}^{\pi,(n)}_t - A^\pi_t)^2] = \underbrace{\text{trajectory variance}}_{\text{from rollouts}} + \underbrace{(\text{critic bias})^2}_{V_\phi \neq V^\pi} + \underbrace{\text{critic variance}}_{\text{from training data}}$$

**Trajectory variance.** Randomness from rewards, transitions, and policy. More rollouts shrinks it. Within a single rollout the $\delta_{t+k}$ are not IID, since they come from the same trajectory, and we can't rewind to $(s, a)$ and re-roll the future independently.

**Critic bias.** $V_\phi$ might not match $V^\pi$ in expectation. This enters the n-step estimate through $\gamma^n V_\phi(S_{t+n})$, weighted by $\gamma^n$.

**Critic variance.** $V_\phi$ depends on its training data, so a different training run on different data would give a different $V_\phi$, with the same $\gamma^n V_\phi(S_{t+n})$ entry point as the bias.

## §11 Dropping the critic: GRPO

In GRPO (Group Relative Policy Optimization) the critic is dropped entirely. A second LLM-scale value network is expensive to train and serve alongside the policy, and critic variance goes away as a side effect. In place of $V_\phi(S_t)$, the baseline is a group mean over $K$ rollouts from the same prompt:

$$\hat{A}_i = \frac{R_i - \mathrm{mean}(R_1, \ldots, R_K)}{\mathrm{std}(R_1, \ldots, R_K)}$$

$V_\phi$ encoded the usefulness of states. Without it, every token in a rollout gets the same advantage. From what I've read, this seems to work for short sparse-reward tasks like math with a verifier at the end, and I imagine it gets harder when per-step credit matters along a longer trajectory.

## §12 Closing

I found the policy-gradient baseline easier to understand once I saw it as the §1 formula.

Up next:

- **Off-policy correction.** Importance sampling for variance reduction.
- **PPO clipping.** Bound how much the policy reacts to a noisy gradient instead of reducing the noise.

## References

- **Owen**, *Monte Carlo theory, methods and examples*. https://artowen.su.domains/mc/
- **Williams 1992**, "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning" (*Machine Learning*).
- **Greensmith, Bartlett, Baxter 2004**, "Variance Reduction Techniques for Gradient Estimates in Reinforcement Learning" (JMLR).
- **Schulman, Moritz, Levine, Jordan, Abbeel 2015**, "High-Dimensional Continuous Control Using Generalized Advantage Estimation" (arXiv:1506.02438).
- **Hastie, Tibshirani, Friedman**, *The Elements of Statistical Learning*.
- **Shao et al. 2024**, "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models".
