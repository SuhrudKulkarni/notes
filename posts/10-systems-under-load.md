# Systems under load

Software systems, hardware, queues, and trading desks all have to work correctly under load. As any of them fills up, the cost of the next unit of work grows superlinearly, and the slowest requests get much slower before the system stops accepting new ones. The same formulas can be used to analyze all four.

## §1 Little's Law and headroom

At equilibrium, the average number of items in flight equals the arrival rate times the average time each item spends in the system,

$$L = \lambda W.$$

The formula applies to queues at a database, requests through an API, jobs on a cluster, and orders in a market, and since arrivals come in bursts, $\lambda$ over a short window can be much higher than the long-run mean. A system sized for the long-run mean saturates whenever a burst arrives, and we hope the backlog clears on its own, otherwise the system falls over.

## §2 Wait times near capacity

A simple model of an open queue, the M/M/1, has an exact formula for the average wait,

$$W = \frac{\rho}{\mu(1 - \rho)},$$

where $\mu$ is the rate at which the system can serve items and $\rho = \lambda / \mu$ is the utilization, the fraction of capacity in use. At $\rho = 0.5$ the average wait is 1 service time. At $\rho = 0.9$ it is 9. At $\rho = 0.99$ it is 99.

Throughput, the rate at which the system completes work, equals $\lambda$ all the way up to saturation. So throughput stays flat while the tail of the latency distribution is already getting much worse. This is why p99 latency, the time the slowest 1 percent of requests take, is the right metric for a service under load, and why most teams target a utilization of 70 to 80 percent rather than running closer to capacity.

M/M/1 is the cleanest case, with Poisson arrivals and exponential service times. Real systems almost always have correlated arrivals and heavy tails, and the wait blows up faster than the formula predicts.

## §3 Amdahl's law

Suppose a workload has a serial part that cannot be parallelized and a parallel part that can be split across $n$ workers. If $s$ is the fraction of the work that is serial, the speedup from $n$ workers is

$$S(n) = \frac{1}{s + (1 - s)/n}.$$

As $n$ grows, $S$ approaches $1/s$. The serial fraction sets a hard ceiling on speedup no matter how much hardware we add. A workload that is 95 percent parallel can never run more than 20 times faster than serial.

Amdahl and the M/M/1 wait formula are both one over a sum where a small term, $s$ in Amdahl and $1-\rho$ in M/M/1, dominates the cost at the limit.

## §4 Trading in a sell-off

The same pattern shows up on a trading desk. A market maker provides liquidity, meaning it stands ready to buy from anyone who wants to sell and sell to anyone who wants to buy. It earns a small spread for performing that service, and to keep things manageable, it tries to keep its portfolio flat. When the market sells off, everyone is trying to sell at once, and the market maker is at risk of buying far too much. Since everyone else is selling, buying more later will be easy, so what traders often do is step back on the buy side, find out how aggressively they need to sell down the inventory they already have, and only then start buying again. The inventory is the queue, the selling pressure is the arrival rate, and stepping back on the buys is reducing $\lambda$.

The trader and the load balancer are running the same algorithm. When the cost of taking on one more unit of work rises faster than the value of doing so, we back off and let the queue drain. In both settings, the tail is the leading indicator and the mean is the lagging one.