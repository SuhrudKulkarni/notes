# notes

Notes on RL, optimization, and numerical methods. Quant working through modern AI research. Some posts are connections to adjacent fields I've worked in: quant, hypothesis testing, operations research, personalization. These are working thoughts.

---

## Posts

### Up

- **[1 Variance reduction and GAE](posts/1-variance-reduction.md).** The same trick shows up in Monte Carlo (control variates), trading (minimum-variance hedge), quant finance (excess return), and policy-gradient RL (the advantage function). A walk through the unified formula and modern policy gradient methods like GAE and GRPO.

### In flight

- **2 Fisher Information again and again.** Cramér-Rao, natural gradient, evolution strategies, TRPO, KL second-order expansion, Adam-as-loose-natural-gradient. This matrix shows up everywhere. Plus first vs. second order methods.

- **3 Kernels in modern attention.** $QK^\top$ is kernel-y, asymmetric and so not Mercer-PSD. If I squint, attention looks kernel-shaped. Performer applies random Fourier features to the softmax kernel for sub-quadratic attention. RoPE applies Bochner's spectral decomposition to the position kernel.

- **4 Goal programming: hyperparameters as economic coefficients.** ML loss functions combine weighted penalty terms, tuned in validation. Operations Research attacks this head on with goal programming. Thoughts on prices (exchange rates between objectives), dials (tuned because no measurement is available), and constraints (things the optimizer should never violate).

- **5 Double or nothing: Duality** Super egg drop, SVMs, DualDICE. Three settings where I see duality. The dual simplifies egg drop, opens computational access in SVMs, and is nicer math but computationally unloved in DICE.

- **6 From A/B testing to ML eval.** The standard interview question (A/B test, CTR 95% vs 97%, is it significant?) opens onto the Wald-Score-LR trinity, Neyman-Pearson, Wilks' theorem. ML benchmarks have prompt clusters, within-subjects comparisons, contamination, multi-checkpoint reporting. So what test do we use? Thoughts on cluster-robust standard errors, panel methods, multiple-comparison corrections.

- **7 LambdaMART and DPO.** Both convert preference signals into pointwise gradients. RankNet's pairwise model is the same Bradley-Terry model that DPO uses. RankNet to LambdaRank to LambdaMART echoes RLHF to DPO in personalization.

- **8 Clipping as guardrails.** Ratio clipping in PPO, pnl clipping in backtests, gradient clipping in deep learning. When the inputs are messy, we contain the reaction.

- **9 Binary search is just bisection.** Binary search done right, with edge cases handled. The same idea shows up in bisection root-finding, price-yield in fixed income, and inverse CDF sampling.

---

*Suhrud Kulkarni — last updated 2026-05-10*
