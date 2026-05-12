# notes

Notes on RL, optimization, and numerical methods. Quant working through modern AI research. Some posts are connections to adjacent fields I've worked in: quant, hypothesis testing, operations research, personalization. These are working thoughts.

---

## Posts

### Up

- **[1 Variance reduction and GAE](posts/1-variance-reduction.md).** The same trick shows up in Monte Carlo (control variates), trading (minimum-variance hedge), quant finance (excess return), and policy-gradient RL (the advantage function). A walk through the unified formula and modern policy gradient methods like GAE and GRPO.

- **[2 Fisher Information again and again](posts/2-fisher-information.md).** The same matrix appears as a preconditioner in natural gradient, a constraint in TRPO, a diagonal approximation in Adam, an ingredient in NES, and a lower bound in Cramér-Rao.

- **[4 Goal programming: hyperparameters as economic coefficients](posts/4-goal-programming.md).** Lagrangian multipliers as shadow prices, prices we measure, dials we tune, constraints we cannot violate, and some questions about RLHF penalty terms.

- **[7 Personalization and RLHF](posts/7-lambdamart-dpo.md).** Parallels between RankNet/LambdaMART and DPO, both built on Bradley-Terry.

- **[8 Robustness and clipping](posts/8-clipping.md).** A guardrail against bad inputs can go on the input itself, with robust estimators like winsorization, MAD, or MCD, or on the function output, with gradient clipping, PPO ratio clipping, or per-trade PnL clipping.

- **[9 Binary search and bisection](posts/9-binary-search.md).** A one-stop-shop C++23 template that solves integer search, bisection root-finding, bond yield-to-maturity, and inverse CDF sampling.

- **[Hand-rolled multi-head self-attention](code/handrolled_multihead_attention.py).** Causal MHA with fused QKV, raw `nn.Parameter` matrices, and einops + einsum.

- **[Two implementations of RoPE](code/handrolled_rope.py).** Even/odd slicing and a block-diagonal rotation matrix.

- **[Reward hacking: optimizer gaming a misspecified objective](code/tool_use_objective_gaming.py).** Toy REINFORCE with two tasks and an `add()` tool. A bonus allows the optimizer to game the obj and use the tool unnecessarily.

### In flight

- **3 Kernels in modern attention.** $QK^\top$ is kernel-y, asymmetric and so not Mercer-PSD. If I squint, attention looks kernel-shaped. Performer applies random Fourier features to the softmax kernel for sub-quadratic attention. RoPE applies Bochner's spectral decomposition to the position kernel.

- **5 Double or nothing: Duality** Super egg drop, SVMs, DualDICE. Three settings where I see duality. The dual simplifies egg drop, opens computational access in SVMs, and is nicer math but computationally unloved in DICE.

- **6 From A/B testing to ML eval.** The standard interview question (A/B test, CTR 95% vs 97%, is it significant?) opens onto the Wald-Score-LR trinity, Neyman-Pearson, Wilks' theorem. ML benchmarks have prompt clusters, within-subjects comparisons, contamination, multi-checkpoint reporting. So what test do we use? Thoughts on cluster-robust standard errors, panel methods, multiple-comparison corrections.

---

*Suhrud Kulkarni — last updated 2026-05-11*
