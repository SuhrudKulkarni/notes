# Personalization and RLHF

Personalization and RLHF both convert preference data into pointwise gradients with the same Bradley-Terry model: RankNet (2005) for ranking, DPO (2023) for RLHF.

## §1 Bradley-Terry

Given a pair of items $i$ and $j$, Bradley-Terry parameterizes the log-odds $\log \frac{P(i \succ j)}{1 - P(i \succ j)}$ as a score difference $s_i - s_j$. Solving for $P(i \succ j)$ gives the logistic form:

$$P(i \succ j) = \sigma(s_i - s_j) = \frac{1}{1 + \exp(-(s_i - s_j))}.$$

The scores are learned from observed pairwise preferences. With label $y_{ij} = 1$ if $i$ beat $j$ and $0$ otherwise, the log-likelihood is

$$\mathcal{L} = \sum_{(i, j) \in \mathcal{D}} \left[ y_{ij} \log \sigma(s_i - s_j) + (1 - y_{ij}) \log \sigma(s_j - s_i) \right].$$

In the above, preferences are assumed to be transitive: if $s_i > s_j$ and $s_j > s_k$, then $s_i > s_k$. Often times in personalization we see people's preferences don't have a consistent transitive order. Nonetheless, B-T is good enough. Pairs are easier to label than lists, and getting all pairs right would give the right listwise ranking like in bubble sort.

## §2 RankNet, LambdaRank, LambdaMART

Personalization uses Bradley-Terry through ranking models. RankNet puts a neural network scoring function on top of B-T, so pairwise preference data gives pointwise gradients on item scores.

The ranking metric we optimize is the discounted cumulative gain, $\text{DCG} = \sum_i \text{rel}_i / \log_2(i+1)$, where $\text{rel}_i$ is the relevance of the item at position $i$. The denominator penalizes mistakes at the top more than mistakes lower down, since people scroll less. NDCG normalizes DCG so it lies in $[0, 1]$.

LambdaRank keeps the B-T pairwise loss but weights each pair's gradient by the change in NDCG that pair-swapping would produce. Swaps near the top of the ranking get larger gradients.

LambdaMART keeps the same B-T pairwise loss and the NDCG weighting, and uses gradient-boosted trees as the scoring function. XGBoost exposes this as `rank:pairwise` and `rank:ndcg`, which makes LambdaMART very accessible.

## §3 DPO

Similarly, DPO (2023) applies Bradley-Terry to LLM preference pairs, with prompt $s$, chosen response $a_w$, and rejected response $a_l$, using the notation of $S, A$ from Post 1. Applying the B-T assumption on the rewards gives:

$$P(a_w \succ a_l \mid s) = \sigma\big(r(s, a_w) - r(s, a_l)\big).$$

Applying a bandit assumption on Post 1's $Q^\pi(s, a)$ gives the RLHF objective with the reward model $r(s, a)$ in place of $Q$:

$$\max_\pi \mathbb{E}_{a \sim \pi}[r(s, a)] - \beta \cdot D_{KL}(\pi \| \pi_{\text{ref}}),$$

where $\pi_{\text{ref}}$ is a reference policy that keeps $\pi$ from wandering too far away. The sum over timesteps from Post 1 has summed out to one term, with no per-token credit assignment, similar to GRPO. A Lagrangian on the KL constraint gives:

$$\pi^\ast(a \mid s) = \frac{1}{Z(s)} \pi_{\text{ref}}(a \mid s) \exp\left(\frac{1}{\beta} r(s, a)\right),$$

<details>
<summary>Derivation: Lagrangian on the KL constraint</summary>

Expand KL:

$$J(\pi) = \sum_a \pi(a \mid s) r(s, a) - \beta \sum_a \pi(a \mid s) \log \frac{\pi(a \mid s)}{\pi_{\text{ref}}(a \mid s)}.$$

We add the Lagrangian for $\sum_a \pi(a \mid s) = 1$ and take $\partial / \partial \pi(a \mid s)$:

$$r(s, a) - \beta \log \frac{\pi(a \mid s)}{\pi_{\text{ref}}(a \mid s)} - \beta - \lambda = 0.$$

We solve for $\pi$:

$$\pi(a \mid s) = \pi_{\text{ref}}(a \mid s) \exp\left(\frac{r(s, a)}{\beta} - 1 - \frac{\lambda}{\beta}\right).$$

Duality showing up again.

</details>

$$r(s, a) = \beta \log \frac{\pi^\ast(a \mid s)}{\pi_{\text{ref}}(a \mid s)} + \beta \log Z(s),$$

$$r(s, a_w) - r(s, a_l) = \beta \log \frac{\pi^\ast(a_w \mid s)}{\pi_{\text{ref}}(a_w \mid s)} - \beta \log \frac{\pi^\ast(a_l \mid s)}{\pi_{\text{ref}}(a_l \mid s)},$$

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(s, a_w, a_l)} \log \sigma\left(\beta \log \frac{\pi(a_w \mid s)}{\pi_{\text{ref}}(a_w \mid s)} - \beta \log \frac{\pi(a_l \mid s)}{\pi_{\text{ref}}(a_l \mid s)}\right).$$

## §4 Other parallels

**Exploration-exploitation.** In personalization there's a balance between showing what people like and collecting info on new content or things people haven't seen. I've seen epsilon-greedy, LinUCB, and Thompson sampling as some ways to deal with this tradeoff. It seems like RLHF could have at least two versions of the same tradeoff: which prompt-response pairs to send for annotation, and at generation time with sampling temperature.

**Diversity.** When we recommend a list of items, the marginal value of similar items decreases. Personalization utility is often modelled as submodular in the list, which formalizes the idea of diminishing returns through the inequality $f(X \cup Y) \leq f(X) + f(Y)$. So shoe content might be very relevant, but the tenth shoe post doesn't add much for the user. The LLM analogy might be worse than submodular: one paragraph in a given style is good, but the tenth paragraph in the same style could be grating. It seems like a version of this might show up in RLHF, where models start producing homogeneous outputs.

## References

- **Burges 2010**, "From RankNet to LambdaRank to LambdaMART: An Overview".
- **Rafailov et al. 2023**, "Direct Preference Optimization: Your Language Model is Secretly a Reward Model".
