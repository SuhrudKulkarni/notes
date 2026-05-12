# Robustness and clipping

There are two places we can put a guardrail against bad data inputs: on the input itself by cleaning the data before a function sees it, or on the function output. Input-side cleaning works when the input is a data matrix sitting in memory. Function-side clipping contains function behavior directly, including when the input only exists transiently inside the function.


## §1 Robust estimators

Simple input-side cleaning, like winsorizing a column, dropping observations more than 2 std. devs. from the mean, or trimming the top and bottom percentiles, protects a data pipeline by capping any observation before it enters.

Robust estimators are built from statistics that are insensitive to outliers. The simplest is the median absolute deviation, where the mean is replaced with the median and the standard deviation with $\mathrm{MAD}(x) = \mathrm{median}(|x_i - \mathrm{median}(x)|)$. The median has breakdown point $1/2$, so up to half the data can be perturbed and the median is still unchanged.

However, for higher-dimensional messy data like bond yields, the simple methods can be insufficient. Prices can be stale because an issue may not trade in size for weeks, and a single issuer can have a systematic dislocation from a debt restructuring or an acquisition. The means, standard deviations, and sample covariance computed from this kind of data reflect the stale prices and dislocations as much as the underlying returns.

For these cases there are several robust estimators, one of which is the Minimum Covariance Determinant (MCD) estimator. MCD returns the covariance of the subset of observations with the smallest sample-covariance determinant, ignoring the rest. Running PCA on an MCD covariance can give loadings closer to factor structure by filtering out corruption.

## §2 Function-side clipping

Gradients, PPO importance ratios, and backtest PnL are all quantities that exist inside a function and can't be cleaned ahead of time. Thus we need to put the guardrail on the function output. Gradient clipping caps the gradient norm at some $c$, PPO ratio clipping caps the policy ratio at $[1-\epsilon, 1+\epsilon]$, and per-trade PnL clipping caps the PnL contribution from any single trade.

Each of these thresholds encodes a prior on how the function ought to behave. The PPO bound $\epsilon$ reflects an expectation that the new policy stays within $1 \pm \epsilon$ of the old. The PnL multiple reflects an expectation that a single trade much larger than the median is corruption rather than signal.

## §3 Assumptions of clipping

Clipping bounds the function's range, and assumes the function's behavior outside that range is unwanted. We further assume that whatever was clipped was an outlier.

For example, we might see a backtest trade with 50x median PnL. It could be from a bad third-party price, a bad fill, or a stale mark on a bond that didn't trade that day, in which case the clip caught a genuine outlier. Or the trade may only fill in simulation because the backtest assumes client flow it wouldn't see in production, in which case the clip hides a structural backtest bug. Both look the same after the clip.

Similarly, a clipped gradient might be one bad minibatch, or a sign that the learning rate is too high and every gradient from here on will get clipped. PPO ratio clipping keeps the updated policy from straying too far from the reference. However, a clipped ratio might be a noisy advantage estimate, or it might be an actual divergence between the new and old policies.

Clustering or a rising fraction of clipped observations can point to a systematic issue, and keeping the raw pre-clip values around is useful for debugging.

## References

- **Rousseeuw & Leroy 1987**, *Robust Regression and Outlier Detection*.
- **Huber 1981**, *Robust Statistics*.
