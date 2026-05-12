# Goal programming: hyperparameters as economic coefficients

A loss with multiple weighted terms is a multi-objective optimization, with the weights setting the exchange rates between the objectives. In ML the weights are hyperparameters tuned in validation, while operations research calls this setup goal programming. When a weight has an economic meaning, it can be measured and used directly. In the Lagrangian formulation, the multiplier on a constraint is a shadow price with real units.

## §1 The Lagrangian and the shadow price

For a constrained problem $\min f(x)$ s.t. $g(x) \leq b$, the Lagrangian is

$$L(x, \lambda) = f(x) + \lambda (g(x) - b),$$

and the dual set up is:

$$g(\lambda) = \min_x L(x, \lambda), \qquad \max_{\lambda \geq 0} g(\lambda).$$

At the optimum,

$$\lambda^\ast = \frac{\partial L^\ast}{\partial b},$$

so $\lambda^\ast$ is the marginal return on relaxing the constraint by one unit, with units of "objective per unit of $g$." It is the shadow price of the constraint and the primal price the optimizer pays to use the resource.

The unconstrained penalty form $\min f(x) + \lambda g(x)$ has the same interpretation. $\lambda$'s units correspond to a real-world exchange rate and it can be measured.

For example, in trading, the minimum-variance hedge ratio $\beta^\ast = \mathrm{Cov}(X, Y) / \mathrm{Var}(Y)$ from [Post 1](1-variance-reduction.md) is fit by regression and used directly in the hedge. In tech, economists at large companies run lengthy experiments that produce dollar-value coefficients for things like a click, a percentage point of retention, or the long-term cost of missing a delivery promise. These values are used directly in downstream objectives.

The marginal-return interpretation also lets us compare strategies sharing a limited resource. For example, when allocating capital between two trading strategies, we can give to the one with the higher $\lambda^\ast$ until the two are equal.

## §2 Three categories of coefficients

Goal programming separates the coefficients in an objective into three kinds.

**Prices** are measured exchange rates with units and confidence intervals, like the hedge ratio or the dollar-value-of-a-click.

**Dials** are coefficients without a measurement, tuned on validation as a guess at the exchange rate.

**Constraints** are goals that cannot be violated, either because they are hard limits like a book-size cap or a regulatory rule, or because the cost of violation cannot be priced.

## §3 Weighted goal programming and possible issues

Weighted goal programming writes the objective as

$$\min \sum_i w_i \cdot f_i,$$

where $f_i$ is the $i$th penalty term and $w_i$ is its weight.

Take a trading strategy that maximizes PnL minus a penalty on falling out of the top-N liquidity-provider ranking, $\max \text{PnL} - w \cdot d_{\text{rank}}$. The optimizer can quote more passively, trade less, capture more spread per trade on the trades that do fill, and earn more under the objective. The cost is to long-term client relationships and to future flow, and gets realized over a longer period than the backtest window used to set $w$. It is also sticky, because once flow routes to competitors, winning it back is much harder.

We can instead write this as $\max \text{PnL}$ s.t. $\text{rank} \leq N$, so that the optimizer cannot game the objective by sacrificing rank in a short window.

When the cost of violation is unobservable in the data used to set the weight, that weight can be mispriced, and a constraint is the conservative choice.

## §4 Lexicographic goal programming

Another form of goal programming avoids weight assumptions by solving in strict priority order. We solve for $f_1^\ast = \min f_1$, then $\min f_2$ s.t. $f_1 \leq f_1^\ast$, then $\min f_3$ s.t. $f_1 \leq f_1^\ast, f_2 \leq f_2^\ast$, and so on. Each solved objective becomes a constraint for the next.

This is the same as the weighted version in the limit where each higher-priority weight is infinitely larger than the next. The cost is computational, since now a single optimization is replaced by a sequence. To me, system prompts in LLMs look a lot like a lexicographic ordering, where the model satisfies a hard safety check before optimizing for helpfulness.

## §5 Questions in RLHF

RLHF objectives combine several weighted terms, and it seems like the weights on these are tuned rather than measured.

A length penalty has units of reward per token, which seems measurable from a user-preference study. Sycophancy looks more like the market-maker case, where the cost is to user trust over time and sticky, so a constraint might be safer than a penalty. Lastly, for inflammatory and harmful content, no weight is acceptable and we need to enforce a hard constraint.

I appreciate that we cannot backprop through a hard constraint, so enforcing one means either a code rule, projection, or a soft Lagrangian. The Lagrangian route puts us back at a weighted penalty, with all the mispricing risks from §3 if $\lambda$ is not a price.

## References

- **Boyd & Vandenberghe**, *Convex Optimization*, Ch. 5 (Lagrangian duality and sensitivity).
