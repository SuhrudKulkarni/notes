# Binary search and bisection

We can make all binary search problems look the same. The recurring theme is a `while` loop, a midpoint, a predicate, and a window update. We only have to change the predicate. This same template can solve integer search, root-finding on continuous functions, sampling from inverse CDFs, and bond yield-to-maturity.

## §1 The template

```cpp
template <typename Fpredicate>
int binary(int low, int high, Fpredicate&& predicate) {
    while (low < high) {
        const auto midpoint = (low + high) / 2;
        if (predicate(midpoint)) {
            high = midpoint;
        } else {
            low = midpoint + 1;
        }
    }
    return low;
}
```

The loop maintains the invariant that our answer is in `[low, high)`, halving the interval at each step until the window collapses to a single point.

## §2 The monotone predicate

The template works whenever the predicate is monotone: `false` everywhere up to some threshold, then `true` from there on. The return value is the leftmost `x` where the predicate flips. The "sorted array search" formulation is the simplest case where the predicate is `arr[i] >= target`.

## §3 Koko Eating Bananas

A famous leetcode problem: given piles of bananas and `h` hours, find the minimum eating speed that finishes all piles in time. We just have to specify the lambda:

```cpp
class Solution {
public:
    int minEatingSpeed(const std::vector<int>& piles, const int hours_available) {
        const auto can_eat = [&](const auto eating_speed) {
            const auto required_hours = std::ranges::fold_left(piles, 0.0,
                [&](const auto running_hours, const auto pile_size) {
                    return running_hours + std::ceil(pile_size * 1.0 / eating_speed);
                });
            return required_hours <= hours_available;
        };
        return binary(1, *std::ranges::max_element(piles), can_eat);
    }
};
```

## §4 Bisection root-finding

For continuous monotone functions, we replace integer indices with floats, swap `low < high` for `low + TOL < high`, and change the predicate to a sign check on `predicate(x)`.

```cpp
constexpr auto TOL = 1.0e-9;

template <typename Fpredicate>
double bisect(double low, double high, Fpredicate&& predicate) {
    while (low + TOL < high) {
        const auto midpoint = (low + high) / 2.0;
        if (predicate(midpoint) >= 0) {
            high = midpoint;
        } else {
            low = midpoint;
        }
    }
    return low;
}
```

**Bond yield-to-maturity.** Given cash flows $c_1, \ldots, c_n$ at times $t_1, \ldots, t_n$ and market price $P$, the yield $y$ solves $P = \sum_i c_i \cdot e^{-y \cdot t_i}$. Price is monotonically decreasing in yield, so $P - \text{present value}$ goes from negative below the true yield to positive above it. Yields are expressed as decimals (`0.05` = 5%), and we bracket up to 25% to allow distressed high-yield names.

```cpp
const auto price_minus_present_value = [&](const auto yield) {
    const auto discounted = std::views::zip(cashflows, times)
        | std::views::transform([&](const auto& cashflow_and_time) {
            const auto& [cashflow, time] = cashflow_and_time;
            return cashflow * std::exp(-yield * time);
        });
    return market_price - std::ranges::fold_left(discounted, 0.0, std::plus{});
};
const auto computed_yield = bisect(-0.05, 0.25, price_minus_present_value);
```

**Inverse CDF sampling.** Inverse-transform sampling draws from an arbitrary distribution via a uniform random variable: given a CDF $F$ and $u \sim U[0,1]$, the sample is the $x$ with $F(x) = u$. When $F^{-1}$ has no closed form, bisection on $F(x) - u$ gets the sample:

```cpp
const auto cdf_minus_u = [&](const auto x) { return F(x) - u; };
const auto x = bisect(x_low, x_high, cdf_minus_u);
```

In both cases the predicate is just the function shifted by a constant, like market price minus present value or CDF minus the uniform draw, so the search becomes a root-finding problem on the shifted function.

Binary search returns the leftmost $x$ where the predicate flips, which on a continuous function is the leftmost $x$ where the sign changes. This works when the function crosses zero once. If the function only touches zero, like a double root, or is identically zero across an interval, no sign change exists and the bisection result is not well-defined.

## §5 Secant and Newton

Bisection needs only monotonicity and converges linearly with each step halving the interval, but two stronger assumptions on the function give us faster convergence.

**Secant method** Assuming continuity, we can approximate the root by linear interpolation between two recent points. Given $(x_{n-1}, f(x_{n-1}))$ and $(x_n, f(x_n))$:

$$x_{n+1} = x_n - f(x_n) \cdot \frac{x_n - x_{n-1}}{f(x_n) - f(x_{n-1})}$$

We get superlinear convergence when the function is smooth near the root.

**Newton's method** Assuming differentiability, we can use $f'$ directly:

$$x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}$$

Newton converges quadratically near a simple root, so each step roughly doubles the number of correct digits, but we need $f'$ in closed form. [Post 2](2-fisher-information.md) covered Newton in the optimization context, where the goal is finding roots of the gradient.

As we get more problem structure and compute, we can escalate from bisection to secant to Newton.

## References

- **Burden & Faires**, *Numerical Analysis*, chapter 2 (solutions of equations in one variable).