# Future Improvements to Adaptive Notification Scheduling

## A Technical Roadmap with Mathematical Foundations

This document outlines seven concrete improvements to ProactiveClaw's notification scheduling system, ordered roughly by implementation complexity. Each section includes the mathematical formulation, intuition, and integration notes.

---

## 1. Graded Reward Signal

### Current Limitation

The current system uses a binary reward: respond within 10 minutes → 1.0, otherwise → 0.0. This discards useful information. A 30-second response is a much stronger signal than a 9-minute response, but both receive identical reward.

### Proposed Improvement

Replace the binary reward with a continuous decay function over response latency:

$$r(\Delta t) = \exp\left(-\alpha \cdot \Delta t\right)$$

where $\Delta t$ is the response time in minutes and $\alpha$ controls how quickly the reward drops off.

**Properties at $\alpha = 0.3$:**

| Response time | Reward |
|---------------|--------|
| 0 min (instant) | 1.00 |
| 1 min | 0.74 |
| 3 min | 0.41 |
| 5 min | 0.22 |
| 10 min | 0.05 |
| 15 min | 0.01 |

This naturally extends the window — you still get *some* credit for a 12-minute response, but much less than a 2-minute one. The hard 10-minute cutoff disappears.

**Alternative: linear decay with floor**

$$r(\Delta t) = \max\left(0, \; 1 - \frac{\Delta t}{\tau}\right)$$

where $\tau$ is the maximum response time (e.g., 15 minutes). Simpler, but the exponential version better reflects the intuition that the difference between 0 and 2 minutes matters more than the difference between 8 and 10 minutes.

### Organic Chat Reward

Keep the organic chat reward at 0.5 (or make it configurable). The graded reward only applies to notification responses where we have a measurable $\Delta t$.

### Implementation

In `slack_server.py`, replace:

```python
if time.time() - _last_notification_time <= 600:
    bandit.update_arm(*_last_notification_arm, reward=1.0)
```

with:

```python
delta_t = (time.time() - _last_notification_time) / 60  # minutes
reward = math.exp(-0.3 * delta_t)
if reward > 0.01:  # don't bother updating for negligible rewards
    bandit.update_arm(*_last_notification_arm, reward=reward)
```

No changes needed to `bandit.py` — it already accepts continuous rewards.

---

## 2. Active Exploration via Upper Confidence Bound (UCB)

### Current Limitation

The current system has no exploration strategy. Arms that have never been tried stay at Q = 0 forever. The only exploration comes from organic chat, which is uncontrolled — the user may never organically chat at 6 AM, so we never learn whether a 6 AM notification would work.

### Proposed Improvement: UCB1-style Exploration Bonus

Instead of ranking arms purely by Q-value, add an exploration bonus that favors undersampled arms:

$$\text{score}(a) = Q(a) + c \sqrt{\frac{\ln N}{n_a}}$$

where:
- $Q(a)$ is the current (decayed) Q-value of arm $a$
- $N$ is the total number of updates across all arms
- $n_a$ is the number of updates to arm $a$
- $c$ is an exploration constant (controls explore-exploit tradeoff)

The second term is large when $n_a$ is small relative to $N$ — arms that haven't been tried get a bonus that makes them more likely to appear in recommendations.

**Choosing $c$:** Start with $c = 0.5$. This means an arm with zero data and $N = 100$ total updates gets a bonus of $0.5 \sqrt{\ln 100 / 1} \approx 1.07$ — enough to surface it above most moderately-scored arms. Tune down if recommendations become too exploratory.

### Adaptation for Time Decay

Since we use time-decayed Q-values, the count $n_a$ should also decay. Define an effective count:

$$\tilde{n}_a = \sum_{i=1}^{n_a} \exp(-\lambda \cdot \Delta t_i)$$

where $\Delta t_i$ is the time since the $i$-th update to arm $a$. In practice, maintain a running decayed count analogous to Q:

```
count_decay = exp(-lambda * time_since)
n_effective[a] = count_decay * n_effective_old[a] + 1
```

This ensures that an arm updated 30 days ago is treated as "nearly untested" even if it was updated 50 times back then.

### Alternative: Thompson Sampling

Instead of UCB, model each arm as a Beta distribution:

$$Q(a) \sim \text{Beta}(\alpha_a, \beta_a)$$

At decision time, sample from each arm's posterior and rank by sampled values. Arms with high uncertainty (low $\alpha + \beta$) occasionally sample high, providing natural exploration.

Update rules:
- On reward $r$: $\alpha_a \leftarrow \text{decay} \cdot \alpha_a + r$, $\beta_a \leftarrow \text{decay} \cdot \beta_a + (1 - r)$
- Apply time decay to both parameters to handle non-stationarity

Thompson Sampling is more principled than UCB for non-stationary problems but harder to interpret (the LLM sees sampled scores that vary each time, which may confuse prompt-based reasoning).

### Implementation Notes

Add `n_effective` dict to `bandit_state.json`. Modify `get_recommendations()` to compute UCB scores instead of pure Q-values. The exploration bonus only affects which arms appear in the top-K recommendations — the LLM still makes the final decision.

---

## 3. Gaussian Process Smoothing Over the Time Grid

### Current Limitation

Each arm is independent. If Wednesday 2 PM has score 0.85, Wednesday 3 PM might have score 0.0 simply because the user never happened to chat or receive a notification at that exact hour. But intuitively, availability at 2 PM strongly predicts availability at 3 PM. The current system can't share information across neighboring arms.

### Proposed Improvement

Model the 168-arm Q-value surface as a Gaussian Process (GP) over a 2D grid (day_of_week, hour). The GP uses a kernel function that encodes prior beliefs about how correlated nearby time slots are.

### Kernel Design

The input space is a 2D torus: (day $\in \{0, ..., 6\}$, hour $\in \{0, ..., 23\}$). We need a kernel that respects:
1. Hours close together are correlated (2 PM and 3 PM)
2. Days close together are correlated (Tuesday and Wednesday)
3. Both dimensions wrap around (Sunday is close to Monday; 11 PM is close to midnight)
4. Weekday/weekend distinction may matter

**Periodic kernel on hours:**

$$k_h(h_1, h_2) = \sigma_h^2 \exp\left(-\frac{2 \sin^2\left(\pi |h_1 - h_2| / 24\right)}{\ell_h^2}\right)$$

This is the standard periodic kernel with period 24. The lengthscale $\ell_h$ controls how many neighboring hours share information. With $\ell_h = 3$, an observation at 2 PM influences slots from roughly 11 AM to 5 PM.

**Periodic kernel on days:**

$$k_d(d_1, d_2) = \sigma_d^2 \exp\left(-\frac{2 \sin^2\left(\pi |d_1 - d_2| / 7\right)}{\ell_d^2}\right)$$

Period 7. With $\ell_d = 1.5$, an observation on Wednesday influences Tuesday through Thursday.

**Product kernel:**

$$k\left((d_1, h_1), (d_2, h_2)\right) = k_d(d_1, d_2) \cdot k_h(h_1, h_2)$$

The product kernel assumes day and hour correlations are separable — the smoothing in hours doesn't depend on which day it is.

**Optional: weekday/weekend split**

Add a binary feature $w \in \{0, 1\}$ (weekday vs. weekend) and use:

$$k_w(w_1, w_2) = \begin{cases} 1 & \text{if } w_1 = w_2 \\ \rho & \text{if } w_1 \neq w_2 \end{cases}$$

where $\rho \in [0, 1]$ controls cross-group correlation. With $\rho = 0.3$, weekday observations weakly inform weekend predictions and vice versa.

The full kernel becomes:

$$k = k_d \cdot k_h \cdot k_w$$

### GP Posterior

Given observed rewards $\mathbf{y}$ at arms $\mathbf{X}$, the posterior mean at any arm $x_*$ is:

$$\mu(x_*) = \mathbf{k}_*^T (\mathbf{K} + \sigma_n^2 \mathbf{I})^{-1} \mathbf{y}$$

$$\sigma^2(x_*) = k(x_*, x_*) - \mathbf{k}_*^T (\mathbf{K} + \sigma_n^2 \mathbf{I})^{-1} \mathbf{k}_*$$

where:
- $\mathbf{K}$ is the kernel matrix between observed arms
- $\mathbf{k}_*$ is the kernel vector between $x_*$ and observed arms
- $\sigma_n^2$ is observation noise variance

The posterior mean $\mu(x_*)$ provides smoothed Q-value estimates. The posterior variance $\sigma^2(x_*)$ provides uncertainty — usable for UCB-style exploration (see Section 2).

### Handling Non-Stationarity

Standard GPs assume stationary observations. For time-decayed rewards, two options:

**Option A: Weighted observations.** Weight each observation by its temporal decay $w_i = \exp(-\lambda \cdot \Delta t_i)$. The GP posterior becomes:

$$\mu(x_*) = \mathbf{k}_*^T (\mathbf{K} + \sigma_n^2 \mathbf{W}^{-1})^{-1} \mathbf{y}$$

where $\mathbf{W} = \text{diag}(w_1, ..., w_n)$. Old observations get downweighted.

**Option B: Forget and refit.** Only keep observations from the last $T$ days (e.g., 30 days) and refit the GP periodically. Simpler but loses the smooth decay property.

Option A is more principled and consistent with the current system's philosophy.

### Computational Cost

With $n$ observations, GP inference is $O(n^3)$ due to matrix inversion. For a single user with ~10 interactions/day over 30 days, $n \approx 300$. A $300 \times 300$ matrix inversion is negligible (~1ms). This is not a scaling concern for the single-user case.

### Implementation Notes

Use `scikit-learn`'s `GaussianProcessRegressor` or a lightweight custom implementation. The GP replaces `get_recommendations()` — instead of returning raw Q-values, return posterior means. Store raw observations (arm, reward, timestamp) in addition to Q-values.

The GP is refit on each `get_recommendations()` call (pre-exit only, not on every message). This adds ~10ms of compute at a point where the LLM call takes seconds — negligible.

---

## 4. Contextual Bandits

### Current Limitation

The current bandit selects arms based only on (day, hour). But user availability depends on context: Is their calendar empty? Did they just finish a meeting? Is it a holiday? Are they on vacation? The same hour can have very different response rates depending on what else is happening.

### Proposed Improvement

Extend the bandit to a contextual bandit where each arm's reward depends on a feature vector $\mathbf{x}$ describing the current context.

### Feature Design

At notification scheduling time, construct a feature vector:

$$\mathbf{x} = \begin{bmatrix} \text{calendar\_busy} \\ \text{meetings\_next\_2h} \\ \text{time\_since\_last\_chat (hrs)} \\ \text{unread\_emails} \\ \text{is\_holiday} \\ \text{day\_of\_week (one-hot, 7 dims)} \\ \text{hour\_sin} \\ \text{hour\_cos} \end{bmatrix}$$

**Cyclical encoding for hour:**

$$\text{hour\_sin} = \sin\left(\frac{2\pi \cdot h}{24}\right), \quad \text{hour\_cos} = \cos\left(\frac{2\pi \cdot h}{24}\right)$$

This encodes hour 23 as close to hour 0, which one-hot encoding cannot.

**calendar_busy:** Binary — is there an event overlapping this time slot?

**meetings_next_2h:** Count of calendar events in the next 2 hours from the candidate slot. Captures "about to be busy" vs. "free afternoon."

**time_since_last_chat:** Hours since the user's last message. Captures recency of engagement.

### Model: LinUCB

For each arm $a$ (time slot), maintain a linear model:

$$\hat{r}_a(\mathbf{x}) = \mathbf{x}^T \boldsymbol{\theta}_a$$

The LinUCB algorithm maintains:
- $\mathbf{A}_a = \mathbf{I}_d + \sum_{t: a_t = a} \mathbf{x}_t \mathbf{x}_t^T$ (design matrix)
- $\mathbf{b}_a = \sum_{t: a_t = a} r_t \mathbf{x}_t$ (reward-weighted features)
- $\hat{\boldsymbol{\theta}}_a = \mathbf{A}_a^{-1} \mathbf{b}_a$ (parameter estimate)

The UCB score at decision time:

$$\text{score}_a(\mathbf{x}) = \mathbf{x}^T \hat{\boldsymbol{\theta}}_a + \alpha \sqrt{\mathbf{x}^T \mathbf{A}_a^{-1} \mathbf{x}}$$

The second term is the confidence width — large when the context is unfamiliar for this arm.

### Non-Stationarity via Discounting

Apply the same exponential decay philosophy. At each update, discount existing statistics:

$$\mathbf{A}_a \leftarrow \gamma \mathbf{A}_a + \mathbf{x}_t \mathbf{x}_t^T + (1 - \gamma) \mathbf{I}_d$$

$$\mathbf{b}_a \leftarrow \gamma \mathbf{b}_a + r_t \mathbf{x}_t$$

where $\gamma = \exp(-\lambda \cdot \Delta t)$ as before. The $(1 - \gamma)\mathbf{I}_d$ term prevents the design matrix from collapsing to zero under heavy discounting.

### Practical Considerations

- **Feature collection at scheduling time:** The LLM already checks calendar/email during pre-exit. These tool call results could be parsed to extract features *before* calling the bandit. This requires a two-phase pre-exit: (1) gather context, (2) run bandit with context, (3) LLM makes final decision with bandit recommendations + raw context.
- **Feature collection at reward time:** When recording a reward, the context features from the time the notification was *scheduled* should be stored, not the features at response time. This means persisting the feature vector alongside each notification.
- **Dimensionality:** With ~12 features and 168 arms, each arm needs enough data to estimate 12 parameters. This is slower to converge than the context-free bandit. Consider sharing parameters across arms (a single global model with arm identity as a feature) to pool data.

### Shared-Parameter Variant

Instead of per-arm models, use a single model with arm features:

$$\hat{r}(\mathbf{x}, \mathbf{z}_a) = [\mathbf{x}; \mathbf{z}_a]^T \boldsymbol{\theta}$$

where $\mathbf{z}_a$ encodes the arm (day_sin, day_cos, hour_sin, hour_cos). This pools data across all arms, enabling faster learning at the cost of assuming a shared relationship between context and reward across time slots.

---

## 5. Notification Fatigue Modeling

### Current Limitation

The current system treats each notification independently. But in practice, the 3rd notification in a day is less welcome than the 1st. The user's response probability decreases with notification frequency — a phenomenon called **notification fatigue**.

### Proposed Model

Model the user's response probability as a function of both the time slot quality and the recent notification load:

$$P(\text{respond} \mid a, n_{\text{recent}}) = Q(a) \cdot f(n_{\text{recent}})$$

where $n_{\text{recent}}$ is the number of notifications sent in the last $T$ hours, and $f$ is a fatigue function:

$$f(n) = \exp(-\beta \cdot n)$$

**Properties at $\beta = 0.5$:**

| Notifications in last 24h | Fatigue factor |
|---------------------------|----------------|
| 0 | 1.00 |
| 1 | 0.61 |
| 2 | 0.37 |
| 3 | 0.22 |
| 5 | 0.08 |

This means the effective score of a slot drops rapidly as more notifications have been recently sent. The LLM already has some intuition about this ("don't pad with filler"), but the fatigue model makes it quantitative.

### Integration with Recommendations

Modify `get_recommendations()` to accept a `recent_notification_count` parameter:

$$\text{adjusted\_score}(a) = Q_{\text{effective}}(a) \cdot \exp(-\beta \cdot n_{\text{recent}})$$

The recommendations now reflect fatigue. If the user has already received 3 notifications today, even the best arms will have low adjusted scores, signaling to the LLM that it should schedule fewer or no additional notifications.

### Learning $\beta$ from Data

Instead of fixing $\beta$, learn it from observed response rates:

$$\hat{\beta} = \arg\min_\beta \sum_i \left(r_i - Q(a_i) \cdot \exp(-\beta \cdot n_i)\right)^2$$

where $r_i$ is the observed reward, $a_i$ is the arm, and $n_i$ is the notification count at time of delivery. This can be solved with a simple grid search or gradient descent on the log-likelihood.

### Implementation Notes

Track notification delivery timestamps in a rolling window (e.g., list of timestamps in the last 48 hours). Compute $n_{\text{recent}}$ at recommendation time. Add $\beta$ to `bandit_state.json` alongside $\lambda$.

---

## 6. Multi-Objective Optimization: Engagement vs. Intrusiveness

### Current Limitation

The current system optimizes a single objective: user response probability. But a notification can be well-timed (user responds) and still be annoying (user wishes they hadn't been interrupted). Optimizing purely for response rate could lead to aggressive notification patterns.

### Proposed Framework

Define two objectives:

**Engagement:** $E(a) = Q(a)$ — the probability the user responds. This is what we currently optimize.

**Intrusiveness:** $I(a)$ — the probability the notification is unwelcome. We want to minimize this.

The scheduling objective becomes:

$$\text{score}(a) = E(a) - \eta \cdot I(a)$$

where $\eta$ controls the tradeoff. Higher $\eta$ produces more conservative (less frequent, higher-quality) notifications.

### Measuring Intrusiveness

Intrusiveness is harder to observe than engagement. Possible signals:

1. **Dismissal without response:** Notification delivered, user comes back to the app within 30 minutes but doesn't respond to the notification content (starts a new topic instead). This suggests the notification was seen but not valued.

2. **Rapid session termination:** User responds to a notification but the resulting conversation is very short (1-2 messages). This may indicate the notification interrupted something.

3. **Explicit negative feedback:** User says "stop notifying me" or similar. This is rare but high-signal — set $I(a) = 1$ for that arm and nearby arms.

4. **Time-of-day prior:** Assign a prior intrusiveness based on time of day. Late night (11 PM - 7 AM) gets higher base intrusiveness regardless of response data.

$$I_{\text{prior}}(h) = \begin{cases} 0.8 & \text{if } h \in [23, 24) \cup [0, 7) \\ 0.0 & \text{otherwise} \end{cases}$$

Combine signals with Bayesian updating:

$$I(a) = \gamma \cdot I_{\text{prior}}(a) + (1 - \gamma) \cdot I_{\text{observed}}(a)$$

### Pareto Frontier

For a set of candidate arms, compute the Pareto frontier of (engagement, -intrusiveness). Only recommend arms on or near the frontier — these are the ones that are both high-engagement and low-intrusiveness.

An arm with $E = 0.9, I = 0.8$ (user responds but is annoyed) is dominated by an arm with $E = 0.85, I = 0.1$ (user responds and is happy). The current single-objective system can't distinguish these.

### Implementation Notes

Maintain a second set of Q-values for intrusiveness ($I$-values) with the same decay mechanics. Record dismissal/short-session signals as intrusiveness rewards. Compute $\text{score} = E - \eta \cdot I$ in `get_recommendations()`.

---

## 7. Hierarchical Time Modeling

### Current Limitation

The 168-arm model treats each (day, hour) as independent. Section 3 (GP smoothing) addresses spatial correlation (nearby hours). This section addresses **temporal hierarchy** — the observation that user behavior has structure at multiple time scales:

- **Hourly patterns**: Morning vs. afternoon vs. evening
- **Daily patterns**: Weekday vs. weekend
- **Weekly patterns**: Meetings on specific days
- **Seasonal patterns**: Holidays, vacation periods, daylight savings shifts

### Proposed Model: Hierarchical Bandit

Decompose the Q-value into additive components at different time scales:

$$Q(d, h) = \mu + \alpha_h + \beta_d + \gamma_{d,h}$$

where:
- $\mu$ is the global baseline (overall responsiveness)
- $\alpha_h$ is the hour effect (captures "morning person" regardless of day)
- $\beta_d$ is the day effect (captures "busy on Mondays" regardless of hour)
- $\gamma_{d,h}$ is the interaction effect (captures "specifically available Wednesday 2 PM")

**Regularization:** The interaction term $\gamma_{d,h}$ should be small relative to the main effects. This encodes the prior that most of the signal comes from hour-of-day and day-of-week, with specific (day, hour) combinations adding fine-grained corrections.

$$\text{Loss} = \sum_{i} (r_i - Q(d_i, h_i))^2 + \lambda_1 \sum_h \alpha_h^2 + \lambda_2 \sum_d \beta_d^2 + \lambda_3 \sum_{d,h} \gamma_{d,h}^2$$

With $\lambda_3 > \lambda_1, \lambda_2$ — the interaction terms are more heavily regularized, so they only capture strong (day, hour)-specific effects.

### Advantage Over Flat Model

The flat 168-arm bandit needs to observe each arm independently. The hierarchical model can generalize:

- See the user respond at Monday 2 PM and Tuesday 2 PM → infer $\alpha_{14}$ is high → predict Wednesday 2 PM is also good, even without data there
- See the user respond on Monday at multiple hours → infer $\beta_0$ is high → predict other Monday hours are decent

This is especially valuable early on (cold start) when many arms have zero data. The hierarchical model can make informed predictions after just a few observations.

### Update Rule

Apply updates at all levels simultaneously:

```
# On observing reward r at (day d, hour h):
mu     += lr * decay * (r - Q(d,h))
alpha_h += lr * decay * (r - Q(d,h))
beta_d  += lr * decay * (r - Q(d,h))
gamma_{d,h} += lr * decay * (r - Q(d,h))
```

With different learning rates: $\text{lr}_\mu < \text{lr}_\alpha < \text{lr}_\beta < \text{lr}_\gamma$. The global baseline changes slowly; the interaction terms adapt quickly.

### State Representation

```json
{
  "mu": 0.35,
  "alpha": {"0": 0.1, "1": -0.05, ..., "23": -0.2},
  "beta": {"0": 0.15, "1": 0.1, ..., "6": -0.1},
  "gamma": {"0_7": 0.05, "2_14": 0.3, ...},
  "last_updated": {...},
  "lambda": 0.1
}
```

The recommendation score for arm $(d, h)$ is $\mu + \alpha_h + \beta_d + \gamma_{d,h}$, with decay applied to each component based on its own last-updated timestamp.

---

## Implementation Priority

Ordered by expected impact relative to implementation effort:

| Priority | Improvement | Effort | Impact | Dependencies |
|----------|------------|--------|--------|--------------|
| 1 | Graded reward signal | ~10 lines changed | Medium | None |
| 2 | Notification fatigue | ~30 lines | Medium-High | None |
| 3 | Hierarchical time model | ~100 lines (rewrite bandit) | High | None |
| 4 | UCB exploration | ~40 lines | Medium | Minor state schema change |
| 5 | GP smoothing | ~80 lines + sklearn dep | High | Observation storage |
| 6 | Contextual bandits | ~150 lines | Very High | Two-phase pre-exit, feature extraction |
| 7 | Multi-objective | ~100 lines + new signals | High | Dismissal detection, UI changes |

**Recommended first steps:** Implement #1 (graded rewards) and #2 (fatigue modeling) — they're nearly free and improve the system without architectural changes. Then #3 (hierarchical model) for the biggest single improvement to cold-start and generalization.

**Recommended medium-term:** #4 (UCB) to add principled exploration, then #5 (GP) if the hierarchical model's linear structure proves too rigid.

**Recommended long-term:** #6 (contextual bandits) once there's enough data to justify the complexity, and #7 (multi-objective) once you have reliable intrusiveness signals.

---

## Summary

The current system — a 168-arm bandit with time-decayed rewards feeding recommendations to an LLM — is a strong foundation. These improvements build on it incrementally:

- **Graded rewards** and **fatigue modeling** sharpen the signal quality without changing the architecture
- **Hierarchical modeling** and **GP smoothing** share information across arms, dramatically improving cold-start and sparse-data performance
- **UCB/Thompson Sampling** add principled exploration to discover hidden-good time slots
- **Contextual bandits** incorporate real-time context (calendar, email) into the statistical model, not just the LLM prompt
- **Multi-objective optimization** prevents the system from optimizing for engagement at the cost of user satisfaction

Each improvement is independently valuable and backward-compatible with the existing system. They can be adopted incrementally as usage data accumulates and the system's limitations become empirically apparent.
