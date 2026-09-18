# Correlated Bandits for Notification Scheduling — Learning Guide & Deep Dive

## The Problem

We have a 168-arm bandit (7 days x 24 hours). Each arm represents a (day_of_week, hour) slot. Currently, arms are fully independent — a reward at (Wednesday, 18:00) tells us nothing about (Thursday, 18:00) or (Wednesday, 19:00). But human routines have structure: if a user is free at 6pm on Wednesday, they're probably free at 6pm on Thursday too. We want reward propagation across correlated arms.

---

## Part 0: What You Need to Learn First (Prerequisite Roadmap)

### Learning Order

```
Level 1: Foundations (you likely have most of this)
  ├── Probability & Statistics (Bayes' theorem, conjugate priors, Beta distribution)
  ├── Linear Algebra (matrix multiplication, eigenvalues, positive definiteness)
  └── Standard MAB (UCB1, Thompson Sampling, regret bounds) ← you are here

Level 2: Kernel Methods
  ├── What kernels are and why they matter
  ├── Common kernel functions (RBF, Matern, Periodic)
  └── Kernel smoothing / Nadaraya-Watson estimator

Level 3: Gaussian Processes
  ├── GP regression (prior → posterior)
  ├── Mean and covariance functions
  └── Hyperparameter learning

Level 4: Advanced Bandits
  ├── Contextual bandits (LinUCB)
  ├── Lipschitz bandits (smoothness assumptions)
  ├── Correlated bandits (formal framework)
  ├── GP-UCB (Gaussian Process + Upper Confidence Bound)
  └── Thompson Sampling with correlated priors

Level 5: Our Specific Problem
  ├── Periodic kernels (cyclical time structure)
  ├── Toroidal distance (wrap-around grids)
  └── Composite kernels for (day, hour) space
```

### Concept-by-Concept: What to Learn and Why

| # | Concept | Why You Need It | Estimated Study Time |
|---|---------|-----------------|---------------------|
| 1 | Multivariate Gaussian distribution | GPs are built entirely on this. Need to understand joint, marginal, and conditional distributions | 2-3 hours |
| 2 | Positive semi-definite (PSD) matrices | Kernel matrices must be PSD. Covariance matrices are PSD. This is the bridge between kernels and GPs | 1-2 hours |
| 3 | Kernel functions | The core tool for measuring similarity between arms. Everything else builds on this | 3-4 hours |
| 4 | Gaussian Process regression | The probabilistic model that lets observing one arm update beliefs about all arms | 4-6 hours |
| 5 | Kernel smoothing (Nadaraya-Watson) | The simplest practical approach — you can implement this immediately without full GP machinery | 2-3 hours |
| 6 | GP-UCB algorithm | The theoretically grounded way to do exploration-exploitation with correlated arms | 3-4 hours |
| 7 | Contextual bandits | Alternative framing where (day, hour) becomes a context vector rather than a discrete arm | 2-3 hours |
| 8 | Lipschitz bandits | Understand why smoothness assumptions give you free information about unobserved arms | 2-3 hours |
| 9 | Periodic kernels | Essential for cyclical structure (hour wraps 23→0, week wraps Sun→Mon) | 1-2 hours |
| 10 | Thompson Sampling with correlated priors | If you prefer TS over UCB, this is how to extend it | 2-3 hours |

### Recommended Resources

**Kernels & GPs:**
- Rasmussen & Williams, "Gaussian Processes for Machine Learning" (2006) — the bible, free online at gaussianprocess.org/gpml
- Distill.pub, "A Visual Exploration of Gaussian Processes" (2019) — beautiful interactive visualizations
- David Duvenaud, "The Kernel Cookbook" (2014) — practical guide to choosing kernels

**Bandits:**
- Srinivas et al., "Gaussian Process Optimization in the Bandit Setting" (2010) — the GP-UCB paper
- Russo et al., "A Tutorial on Thompson Sampling" (2018) — covers correlated priors
- Gupta et al., "Correlated Multi-Armed Bandits" (2021) — the formal framework

**Contextual Bandits:**
- Li et al., "A Contextual-Bandit Approach to Personalized News Article Recommendation" (2010) — the LinUCB paper

---

## Part 1: Kernel Functions — Measuring Similarity Between Arms

### Intuition

A kernel function `k(x, x')` answers: "how similar are arms x and x'?" If two arms are similar, observing a reward at one should update our estimate of the other. The kernel encodes our assumptions about the structure of the reward surface.

### Formal Definition

A kernel is a function `k: X × X → R` that is **symmetric** and **positive semi-definite (PSD)**:

```
Symmetric:    k(x, x') = k(x', x)    for all x, x'

PSD:          For any n points x_1, ..., x_n and any real coefficients c_1, ..., c_n:
              ΣΣ c_i c_j k(x_i, x_j) ≥ 0
```

Equivalently, the **kernel matrix** (or Gram matrix) `K` where `K_ij = k(x_i, x_j)` must be PSD — all eigenvalues ≥ 0.

**Why PSD matters:** Kernel matrices are used as covariance matrices in GPs. Covariance matrices must be PSD (you can't have negative variance). If your kernel isn't PSD, you'll get nonsensical probability distributions.

### Common Kernel Functions

#### 1. RBF (Radial Basis Function) / Squared Exponential / Gaussian Kernel

```
k_RBF(x, x') = σ² · exp( -||x - x'||² / (2ℓ²) )
```

- `σ²` = signal variance (overall scale of the function)
- `ℓ` = lengthscale (how far the correlation reaches)
- Infinitely smooth — assumes the reward function is very smooth
- Most commonly used kernel

**For our problem:** With `ℓ = 2` hours, arms 1 hour apart have similarity `exp(-1/8) ≈ 0.88`, arms 3 hours apart have `exp(-9/8) ≈ 0.32`, arms 6 hours apart have `exp(-36/8) ≈ 0.01`. The lengthscale controls the "radius of influence."

#### 2. Matern Kernel

```
k_Matern(x, x') = σ² · (2^(1-ν) / Γ(ν)) · (√(2ν) · d / ℓ)^ν · K_ν(√(2ν) · d / ℓ)

where d = ||x - x'||, K_ν is the modified Bessel function
```

Common special cases:

```
ν = 1/2:   k(x,x') = σ² · exp(-d/ℓ)                              (Ornstein-Uhlenbeck, rough)
ν = 3/2:   k(x,x') = σ² · (1 + √3·d/ℓ) · exp(-√3·d/ℓ)          (once differentiable)
ν = 5/2:   k(x,x') = σ² · (1 + √5·d/ℓ + 5d²/(3ℓ²)) · exp(-√5·d/ℓ)  (twice differentiable)
ν → ∞:     k(x,x') → RBF kernel                                    (infinitely smooth)
```

**For our problem:** Matern-3/2 or 5/2 is often more realistic than RBF. Human routines aren't infinitely smooth — there are sharp transitions (work hours → free time).

#### 3. Periodic Kernel (ExpSineSquared)

```
k_periodic(x, x') = σ² · exp( -2 sin²(π|x - x'|/p) / ℓ² )
```

- `p` = period (24 for hours, 7 for days)
- `ℓ` = lengthscale within one period
- Wraps around: `k(hour=23, hour=0)` is close, not far

**For our problem:** This is critical. Hours are cyclical (23:00 is 1 hour from 0:00, not 23 hours). Days are cyclical (Sunday is 1 day from Monday). The periodic kernel handles this naturally.

**Derivation of why periodic kernel handles wrap-around:**
The periodic kernel is equivalent to first mapping x to the unit circle `(cos(2πx/p), sin(2πx/p))` and then applying an RBF kernel in that space. Points that are `p` apart map to the same point on the circle, so `k(x, x+p) = k(x, x)`.

#### 4. Locally Periodic Kernel

```
k_local_periodic(x, x') = k_periodic(x, x') · k_RBF(x, x')
```

Periodic but with decaying amplitude — the pattern repeats but the correlation weakens over longer time spans. (Not immediately relevant for day-of-week since we always want the periodicity, but useful if you later model longer time spans.)

### Kernel Algebra

Kernels can be combined, and the result is still a valid kernel:

```
k₁ + k₂ is a valid kernel     (captures either pattern)
k₁ · k₂ is a valid kernel     (captures both patterns simultaneously)
c · k₁ is a valid kernel      (for c > 0)
```

**Proof (product):** If K₁ and K₂ are PSD matrices, then K₁ ⊙ K₂ (element-wise / Hadamard product) is PSD. This is the Schur product theorem.

This is how we'll build the composite kernel for our (day, hour) grid.

---

## Part 2: Gaussian Processes — From Kernels to Beliefs Over Functions

### Intuition

A Gaussian Process is a probability distribution over *functions*. Instead of saying "the reward at (Wed, 18) is 0.7", a GP says "I believe the entire reward surface f(day, hour) is drawn from a distribution, and my current best guess plus uncertainty is..."

When you observe a reward at one arm, the GP updates its beliefs about the *entire function* — nearby arms get their estimates pulled toward the observation, with the pull strength determined by the kernel.

### Formal Definition

A GP is fully specified by:
- Mean function: `m(x) = E[f(x)]` (often set to 0)
- Covariance function: `k(x, x') = Cov[f(x), f(x')]` (this is our kernel!)

```
f ~ GP(m, k)

means: for ANY finite set of points {x_1, ..., x_n},
the vector [f(x_1), ..., f(x_n)] is jointly Gaussian:

[f(x_1), ..., f(x_n)]^T ~ N(μ, K)

where μ_i = m(x_i)  and  K_ij = k(x_i, x_j)
```

### GP Regression (The Key Derivation)

Given observed data `D = {(x_i, y_i)}` where `y_i = f(x_i) + ε` and `ε ~ N(0, σ_n²)`, we want to predict `f(x*)` at a new point `x*`.

**Joint distribution (prior + observations):**

```
[ y   ]     ([ m(X)  ]   [ K(X,X) + σ_n²I    K(X,x*)  ])
[ f*  ]  ~  N([ m(x*) ] , [ K(x*,X)           k(x*,x*) ])
```

Where:
- `y = [y_1, ..., y_n]^T` — observed rewards
- `K(X,X)` — n×n kernel matrix between observed points
- `K(X,x*)` — n×1 kernel vector between observed points and query point
- `σ_n²` — observation noise variance

**Posterior (conditioning on observations):**

Using the standard formula for conditional Gaussians `p(a|b) where [a,b] ~ N(...)`:

```
f* | D ~ N(μ*, σ*²)

μ*  = m(x*) + K(x*,X) [K(X,X) + σ_n²I]^(-1) (y - m(X))

σ*² = k(x*,x*) - K(x*,X) [K(X,X) + σ_n²I]^(-1) K(X,x*)
```

**In words:**
- `μ*` = prior mean + (kernel similarity to observations) × (inverse-noise-weighted residuals)
- `σ*²` = prior variance − variance reduction from observations

**For our problem:** After observing rewards at arms (Wed,18), (Thu,18), (Fri,9), the GP posterior gives us updated mean and uncertainty for *all 168 arms*. Arms similar to observed ones (via the kernel) have their means pulled toward observations and their uncertainties reduced.

### Computational Complexity

The bottleneck is inverting `K(X,X) + σ_n²I`, which is `O(n³)`.

For our 168-arm grid, n ≤ 168, so `168³ ≈ 4.7M` operations — trivial on modern hardware. Full GP is perfectly tractable here. (GP approximations are needed when n > ~10,000, which won't happen for us.)

---

## Part 3: GP-UCB — Exploration-Exploitation with Correlated Arms

### Intuition

Standard UCB1 uses `Q(a) + c√(ln(t)/N(a))` — the exploration bonus depends only on how many times *that specific arm* was pulled. GP-UCB replaces this with the GP posterior uncertainty, which decreases not just when you pull that arm, but when you pull *any correlated arm*.

### Algorithm

```
GP-UCB Algorithm:
  Input: Kernel k, noise σ_n², exploration parameters β_t
  Initialize: D = {} (empty dataset)

  For t = 1, 2, 3, ...:
    1. Compute GP posterior for all arms:
       μ_t(x) and σ_t(x) for all x in arm set

    2. Select arm:
       x_t = argmax_x [ μ_t(x) + √β_t · σ_t(x) ]

    3. Observe reward y_t at arm x_t

    4. Update dataset:
       D = D ∪ {(x_t, y_t)}
```

### The Acquisition Function

```
UCB_t(x) = μ_t(x) + √β_t · σ_t(x)
             ↑              ↑
         exploitation    exploration
```

- `μ_t(x)` — posterior mean (exploit: pick arms we think are good)
- `σ_t(x)` — posterior std dev (explore: pick arms we're uncertain about)
- `β_t` — controls the exploration-exploitation tradeoff, grows with t

### Choosing β_t

From Srinivas et al. (2010), for finite arm sets:

```
β_t = 2 ln(|A| · t² · π² / 6δ)
```

Where:
- `|A|` = number of arms (168 for us)
- `t` = round number
- `δ` = failure probability (e.g., 0.1)

**Concrete example for our problem:**
```
At t = 100, δ = 0.1:
β_100 = 2 · ln(168 · 10000 · π²/6 · 10) ≈ 2 · ln(27,600,000) ≈ 2 · 17.1 ≈ 34.3
√β_100 ≈ 5.86
```

In practice, this theoretical value is often too large. A common heuristic is `β_t = 2 ln(t)` or even a fixed `β = 2.0`.

### Regret Bound

```
R_T ≤ O(√(T · β_T · γ_T))
```

Where `γ_T` is the **maximum information gain** — how much information T observations can provide about the function. For an RBF kernel on a D-dimensional space:

```
γ_T = O((log T)^(D+1))
```

**Key insight:** For our 2D grid (day, hour), `γ_T = O((log T)³)`, which is *much* smaller than the `O(168 · log T)` you'd get treating arms independently. Correlated arms give you a massive regret improvement because each observation teaches you about many arms.

### Comparison: GP-UCB vs Independent UCB1

| Aspect | UCB1 (current) | GP-UCB |
|--------|----------------|--------|
| Pulls needed to learn arm (Wed,18) | Must pull (Wed,18) directly | Can learn from (Thu,18), (Wed,19), etc. |
| Information per pull | Updates 1 arm | Updates all arms (weighted by kernel) |
| Regret after T pulls | O(√(168 · T · log T)) | O(√(T · (log T)³)) |
| Cold start for new arm | No information | Inferred from neighbors |
| Computation per step | O(1) | O(n²) where n = # observations |

---

## Part 4: Contextual Bandits — An Alternative Framing

### Intuition

Instead of 168 discrete arms, we can treat (day, hour) as a **context vector** and have a single arm whose reward depends on the context. This is the contextual bandit framing.

### Feature Encoding for Cyclical Time

Naive encoding `[day=3, hour=18]` doesn't capture that hour 23 is close to hour 0. Instead, use cyclical features:

```
φ(day, hour) = [
    sin(2π · day / 7),
    cos(2π · day / 7),
    sin(2π · hour / 24),
    cos(2π · hour / 24),
    sin(2π · day / 7) · sin(2π · hour / 24),    // day-hour interaction
    sin(2π · day / 7) · cos(2π · hour / 24),
    cos(2π · day / 7) · sin(2π · hour / 24),
    cos(2π · day / 7) · cos(2π · hour / 24),
    1                                              // bias term
]
```

This 9-dimensional feature vector:
- Naturally handles wrap-around (sin/cos are periodic)
- Captures day-hour interactions (the cross terms)
- Reduces the problem from 168 arms to a 9-dimensional regression

### LinUCB Algorithm

```
LinUCB:
  Initialize: A = I_d (d×d identity), b = 0_d (d-vector)

  For t = 1, 2, ...:
    1. Observe context x_t = φ(day_t, hour_t)
    2. Compute: θ_hat = A^(-1) b
    3. Compute: UCB = x_t^T θ_hat + α √(x_t^T A^(-1) x_t)
    4. (Select arm / receive reward r_t)
    5. Update: A = A + x_t x_t^T,  b = b + r_t x_t
```

This is `O(d²)` per update where `d = 9` — very fast.

**Limitation:** LinUCB assumes the reward is a *linear* function of features. This works if user availability is roughly sinusoidal (smooth daily/weekly patterns), but can't capture sharp transitions like "free exactly between 6-8pm, busy otherwise."

---

## Part 5: Lipschitz Bandits — Smoothness Gives Free Information

### Intuition

If we assume the reward function is "smooth" (can't change too fast between nearby arms), then observing reward `r` at arm `x` tells us that arms close to `x` have rewards close to `r`. This is formalized by Lipschitz continuity.

### Formal Definition

A function `f: X → R` is L-Lipschitz if:

```
|f(x) - f(x')| ≤ L · d(x, x')    for all x, x'
```

Where `d(x, x')` is a distance metric and `L` is the Lipschitz constant.

**For our problem:** If `L = 0.1` (reward per hour), then observing reward 0.8 at (Wed, 18) means:
- (Wed, 19) has reward in [0.7, 0.9]
- (Wed, 20) has reward in [0.6, 1.0]
- (Wed, 21) has reward in [0.5, 1.0]

### Why This Matters

In standard MAB, to estimate 168 arms to precision ε, you need `O(168/ε²)` samples. With Lipschitz smoothness and a good distance metric, you need `O(C/ε^(2+D))` where `C` depends on the Lipschitz constant and `D` is the effective dimension. For smooth reward surfaces, this is much less.

### Connection to Kernels

Lipschitz bandits and GP bandits are related:
- Lipschitz: hard constraint on how fast f changes (worst-case)
- GP with kernel: soft probabilistic constraint on smoothness (average-case)

The GP approach is richer — it gives you a full posterior distribution, not just bounds.

---

## Part 6: Kernel Smoothing (Nadaraya-Watson) — The Simplest Practical Approach

### Intuition

This is the lowest-hanging fruit. Instead of full GP machinery, just use a weighted average: when you observe a reward, spread it to nearby arms with weights determined by a kernel. No matrix inversions, no posterior distributions — just weighted averaging.

### Mathematical Formulation

The Nadaraya-Watson estimator for the reward at arm `x`:

```
f_hat(x) = Σ_i w_i(x) · y_i  /  Σ_i w_i(x)

where w_i(x) = K((x - x_i) / h)
```

- `y_i` = observed reward at arm `x_i`
- `K` = kernel function (e.g., Gaussian)
- `h` = bandwidth (controls smoothing radius)

### Two Integration Strategies

#### Strategy A: Smoothed Update (modify `update_arm`)

When arm `(d, h)` receives reward `r`, propagate to all arms:

```
For each arm (d', h') in the 7×24 grid:
    weight = kernel((d,h), (d',h'))
    Update Q(d',h') with reward r, scaled by weight
```

The update rule for each arm becomes:

```
Q_new(d',h') = (1 - α·w) · Q_old(d',h') + α·w · r

where w = k((d,h), (d',h')) / k((d,h), (d,h))  ∈ [0, 1]
      α = learning rate
```

**This is what your original "2D Gaussian reward kernel" idea describes.** The kernel-weighted update is the mathematically precise version of "propagate a portion of the reward to nearby arms."

#### Strategy B: Smoothed Readout (modify `get_recommendations`)

Store raw Q-values as before (only update the pulled arm), but when reading Q-values for recommendations, smooth them:

```
Q_smooth(d,h) = Σ_{d',h'} k((d,h),(d',h')) · Q_raw(d',h')  /  Σ_{d',h'} k((d,h),(d',h'))
```

**Trade-off:** Strategy A makes stored Q-values smoother over time (information propagates). Strategy B keeps raw data pure and smooths at read time (more flexible — you can change the kernel without losing data).

### The Kernel for Our (Day, Hour) Grid

```
k((d₁,h₁), (d₂,h₂)) = exp( -d_day²/(2σ_day²) - d_hour²/(2σ_hour²) )
```

Where the distances are **circular** (wrap-around):

```
d_day  = min(|d₁ - d₂|, 7 - |d₁ - d₂|)
d_hour = min(|h₁ - h₂|, 24 - |h₁ - h₂|)
```

**Suggested hyperparameters:**
- `σ_hour = 1.5` — being free at 6pm correlates with ~5pm and ~7pm, weakly with 4pm and 8pm
- `σ_day = 1.5` — Wednesday correlates with Tuesday and Thursday, weakly with Monday and Friday

#### The Full Kernel Matrix (Visualization)

For arm (Wednesday=2, 18:00), the kernel weights across the grid look like:

```
         Mon   Tue   Wed   Thu   Fri   Sat   Sun
  ...
  16:00  0.04  0.14  0.32  0.14  0.04  0.01  0.01
  17:00  0.09  0.32  0.72  0.32  0.09  0.02  0.02
  18:00  0.14  0.47  1.00  0.47  0.14  0.03  0.03  ← center
  19:00  0.09  0.32  0.72  0.32  0.09  0.02  0.02
  20:00  0.04  0.14  0.32  0.14  0.04  0.01  0.01
  ...
```

This is exactly the "tabular calendar with reward bleeding horizontally and vertically" you described.

### Weekday-Weekend Asymmetry

Your intuition about weekday-weekend being different is correct. You can handle this by inflating the distance for weekday-weekend transitions:

```
d_day_adjusted(d₁, d₂):
    raw = min(|d₁ - d₂|, 7 - |d₁ - d₂|)
    if one is weekday (0-4) and other is weekend (5-6):
        return raw * 2.0    // inflate cross-category distance
    else:
        return raw
```

This makes Saturday 10am correlate strongly with Sunday 10am, but weakly with Monday 10am — matching real human behavior.

### Pseudocode Implementation

```python
import math

def circular_distance(a, b, period):
    """Wrap-around distance on a circle of given period."""
    diff = abs(a - b)
    return min(diff, period - diff)

def kernel_weight(d1, h1, d2, h2, sigma_day=1.5, sigma_hour=1.5):
    """Compute kernel weight between two (day, hour) arms."""
    d_day = circular_distance(d1, d2, 7)
    d_hour = circular_distance(h1, h2, 24)

    # Optional: inflate weekday-weekend distance
    is_weekday_1 = d1 <= 4
    is_weekday_2 = d2 <= 4
    if is_weekday_1 != is_weekday_2:
        d_day *= 2.0

    return math.exp(-(d_day**2) / (2 * sigma_day**2)
                    -(d_hour**2) / (2 * sigma_hour**2))

def update_arm_correlated(day, hour, reward, sigma_day=1.5, sigma_hour=1.5,
                          threshold=0.05):
    """Update all arms with kernel-weighted reward propagation."""
    state = load_state()
    now_iso = datetime.now().astimezone().isoformat()

    for d in range(7):
        for h in range(24):
            w = kernel_weight(day, hour, d, h, sigma_day, sigma_hour)
            if w < threshold:  # skip negligible updates
                continue

            key = arm_key(d, h)
            q_old = state["q_values"].get(key, 0.0)

            # Time-decayed update, scaled by kernel weight
            last = state["last_updated"].get(key)
            if last:
                time_since = (now - datetime.fromisoformat(last)).total_seconds() / 86400
            else:
                time_since = 0.0

            decay = math.exp(-state["lambda"] * time_since)
            alpha = (1 - decay) * w  # kernel-scaled learning rate
            q_new = decay * q_old + alpha * reward

            state["q_values"][key] = round(q_new, 4)
            if d == day and h == hour:
                state["last_updated"][key] = now_iso

    save_state(state)
```

---

## Part 7: Correlated Bandits — The Formal Framework

### Definition

A correlated multi-armed bandit is a standard MAB where arm rewards are drawn from a joint distribution:

```
[r_1, r_2, ..., r_K] ~ P(r_1, ..., r_K)
```

The key property: observing `r_i` updates our belief about `r_j` for `j ≠ i`.

In the Gaussian case:

```
[r_1, ..., r_K] ~ N(μ, Σ)
```

Where `Σ` is the covariance matrix (which we specify via our kernel: `Σ_ij = k(arm_i, arm_j)`).

### Why Correlated Arms Help: The Pseudo-Reward Concept

When you pull arm `i` and observe reward `r_i`, you can compute **pseudo-rewards** for all other arms:

```
r_j^pseudo = E[r_j | r_i] = μ_j + Σ_ji / Σ_ii · (r_i - μ_i)
```

This is exactly the GP posterior mean update for arm `j` given observation at arm `i`.

**Information gain:** In independent MAB with K arms, you need `Ω(K)` pulls to learn all arms. With correlated arms, if the correlation structure has effective rank `d << K`, you only need `O(d)` pulls. For our 168-arm grid with smooth kernel, the effective rank might be ~10-20.

### C-UCB Algorithm (Gupta et al. 2021)

```
C-UCB (Correlated UCB):
  Maintains belief μ_hat(a), σ_hat(a) for all arms

  For t = 1, 2, ...:
    1. For each arm a, compute:
       UCB(a) = μ_hat(a) + β · σ_hat(a)

    2. Pull a_t = argmax UCB(a)

    3. Observe r_t

    4. Update beliefs for ALL arms using correlation:
       For each arm a:
         μ_hat(a) += Σ(a, a_t) / (Σ(a_t, a_t) + σ_n²) · (r_t - μ_hat(a_t))
         σ_hat(a)² -= Σ(a, a_t)² / (Σ(a_t, a_t) + σ_n²)
```

**Key result from the paper:** Arms that are "non-competitive" (correlated with a clearly suboptimal arm) require only `O(1)` pulls total — the algorithm learns they're bad from pulling their neighbors.

---

## Part 8: Thompson Sampling with Correlated Priors

### Intuition

Standard Thompson Sampling maintains independent Beta distributions for each arm. Correlated TS maintains a joint distribution where arms are linked — sampling one arm's value constrains what nearby arms can be.

### Gaussian Correlated Thompson Sampling

If we model rewards as jointly Gaussian:

```
θ ~ N(μ, Σ)    where Σ_ij = k(arm_i, arm_j)
```

**Algorithm:**

```
Correlated Thompson Sampling:
  Initialize: μ = 0, Σ = K (prior covariance = kernel matrix)

  For t = 1, 2, ...:
    1. Sample: θ_t ~ N(μ, Σ)                    // sample ALL arm values jointly
    2. Pull: a_t = argmax θ_t(a)                 // pick the highest sample
    3. Observe: r_t
    4. Kalman update:
       k = Σ[:, a_t]                             // column of Σ for pulled arm
       s = Σ[a_t, a_t] + σ_n²                   // scalar
       μ = μ + (k / s) · (r_t - μ[a_t])         // update mean of ALL arms
       Σ = Σ - (k · k^T) / s                     // reduce uncertainty of ALL arms
```

**The Kalman update is the key:** When arm `a_t` is pulled, the mean update for arm `j` is:

```
μ_j += Σ[j, a_t] / (Σ[a_t, a_t] + σ_n²) · (r_t - μ[a_t])
```

Arms with high `Σ[j, a_t]` (high kernel similarity) get large updates. Arms with low similarity are barely affected.

### Practical Heuristic: Correlated Beta Priors

If you want to keep using Beta distributions (Bernoulli rewards) but add correlation, use **kernel-weighted pseudo-counts**:

```
When arm (d, h) gets reward r ∈ {0, 1}:
  For each arm (d', h'):
    w = kernel((d,h), (d',h'))
    α(d',h') += w · r           // pseudo-success
    β(d',h') += w · (1 - r)     // pseudo-failure
```

This is approximate but maintains the simplicity of Beta-Bernoulli Thompson Sampling while adding correlation. The kernel weight `w` controls how much each observation counts as evidence for neighboring arms.

---

## Part 9: Periodic Kernels — Handling Cyclical Time

### The Problem

Standard distance: `|hour=23 - hour=1| = 22` (wrong, they're 2 hours apart)
Circular distance: `min(22, 24-22) = 2` (correct)

But even circular distance + RBF kernel is somewhat ad-hoc. The periodic kernel is the principled solution.

### ExpSineSquared Kernel

```
k_periodic(x, x') = σ² · exp( -2 sin²(π · |x - x'| / p) / ℓ² )
```

**Why sin²?** When we map `x → (cos(2πx/p), sin(2πx/p))` onto the unit circle, the squared Euclidean distance between two points on the circle is:

```
||φ(x) - φ(x')||² = 2 - 2cos(2π(x-x')/p) = 4sin²(π(x-x')/p)
```

So the periodic kernel is equivalent to an RBF kernel applied after mapping to the circle.

### Composite Kernel for (Day, Hour)

Our arms live in a 2D space: `x = (day, hour)`. We need a kernel that handles periodicity in both dimensions. The natural choice is a **product kernel**:

```
k((d₁,h₁), (d₂,h₂)) = k_day(d₁, d₂) · k_hour(h₁, h₂)
```

Where:

```
k_day(d, d')  = σ_d² · exp( -2 sin²(π|d - d'|/7) / ℓ_d² )

k_hour(h, h') = σ_h² · exp( -2 sin²(π|h - h'|/24) / ℓ_h² )
```

**Hyperparameters:**

| Parameter | Meaning | Suggested Value | Reasoning |
|-----------|---------|-----------------|-----------|
| `σ_d²` | Day signal variance | 1.0 | Normalized |
| `σ_h²` | Hour signal variance | 1.0 | Normalized |
| `ℓ_d` | Day lengthscale | 1.5 | Weekday correlation ~2-3 days |
| `ℓ_h` | Hour lengthscale | 2.0 | Time-of-day correlation ~3-4 hours |

### Handling Weekday vs Weekend

A single periodic kernel over 7 days treats all days equally. To encode weekday/weekend structure, add a **categorical component**:

```
k_day(d, d') = k_periodic(d, d') · (1 + γ · same_category(d, d'))

where same_category(d, d') = 1 if both weekday or both weekend, 0 otherwise
      γ > 0 controls the boost for same-category days
```

With `γ = 1.0`, same-category days get 2x the base kernel similarity.

---

## Part 10: Toroidal Distance — The Geometry of Our Grid

### The Grid as a Torus

Our 7×24 grid wraps around in both dimensions:
- Hour 23 is next to hour 0
- Sunday is next to Monday

Topologically, this is a **torus** — a rectangle with opposite edges glued together.

### Toroidal Distance

```
d_torus((d₁,h₁), (d₂,h₂)) = √(d_day² + d_hour²)

where:
  d_day  = min(|d₁ - d₂|, 7  - |d₁ - d₂|)
  d_hour = min(|h₁ - h₂|, 24 - |h₁ - h₂|)
```

**Note:** The two dimensions have different scales (7 days vs 24 hours), so using raw toroidal distance in a single-lengthscale kernel would be wrong. That's why we use separate kernels (or separate lengthscales) per dimension.

### Why Periodic Kernels Handle This Automatically

If you use the product of periodic kernels (Part 9), you don't need to explicitly compute toroidal distance at all. The `sin²(π|x-x'|/p)` term naturally wraps around with period `p`. This is one of the elegances of the periodic kernel approach.

---

## Part 11: Putting It All Together — A Practical Progression

Here's a recommended implementation path, from simplest to most sophisticated. Each level is a complete, working system.

### Level 1: Current System (Independent Arms)
```
What we have now. Each arm is independent.
Update: O(1) per observation
Cold-start: Every arm starts from scratch
```

### Level 2: Kernel-Smoothed Updates (Recommended Starting Point)

Add reward propagation via kernel weights to the existing `update_arm` function. This is the Nadaraya-Watson approach from Part 6.

```
What changes: update_arm() now updates ~20-30 nearby arms per observation
Update: O(168) per observation (loop over all arms, but skip negligible weights)
Cold-start: New arms inherit estimates from neighbors
Complexity: ~30 lines of code change to bandit.py
```

**This is the minimum viable implementation of your idea.** It captures 80% of the benefit with 10% of the complexity of full GP-UCB.

### Level 3: Correlated Beta Thompson Sampling

Replace the decayed Q-values with Beta distributions, add kernel-weighted pseudo-count propagation.

```
What changes: State stores (α, β) per arm instead of Q-values
Update: O(168) per observation
Exploration: Natural via Thompson Sampling
Complexity: ~80 lines of code change
```

### Level 4: Correlated Gaussian Thompson Sampling

Full Kalman filter updates with a kernel-defined prior covariance.

```
What changes: State stores μ (168-vector) and Σ (168×168 matrix)
Update: O(168²) per observation (matrix update)
Exploration: Optimal via joint sampling
Complexity: ~120 lines, needs numpy
```

### Level 5: Full GP-UCB

Full Gaussian Process with posterior inference and UCB acquisition.

```
What changes: Full GP regression per decision
Update: O(n³) where n = total observations (but n ≤ 168)
Exploration: Theoretically optimal with regret bounds
Complexity: ~200 lines, or use GPy/scikit-learn
```

### My Recommendation

**Start with Level 2** (kernel-smoothed updates). It's the direct implementation of your intuition, requires minimal code changes, has no new dependencies, and gives you the core benefit: reward propagation across correlated arms. You can always upgrade to Level 3-5 later if the simple approach isn't enough.

---

## Part 12: Open Questions and Future Directions

### Hyperparameter Learning

How do we know `σ_day = 1.5` is right? With enough data, we can learn hyperparameters by maximizing marginal likelihood:

```
log p(y | X, θ) = -1/2 y^T (K + σ_n²I)^(-1) y - 1/2 log|K + σ_n²I| - n/2 log(2π)
```

This requires gradient-based optimization (the three terms balance data fit, model complexity, and a constant). For Level 2, you can start with reasonable defaults and tune manually.

### Non-Stationary Rewards

User routines change. Someone might switch jobs and their schedule shifts. The current time-decay (`exp(-λ·days)`) handles this somewhat, but a more principled approach is a **non-stationary GP** where the kernel itself changes over time:

```
k_t(x, x') = k(x, x') · exp(-λ · |t - t'|)
```

### Multi-User Extension

If ProactiveClaw ever supports multiple users, you could use a **multi-task GP** where users share a common reward structure but have individual variations:

```
k((user_a, d, h), (user_b, d', h')) = k_user(a, b) · k_day(d, d') · k_hour(h, h')
```

---

## References

1. **Srinivas et al. (2010)**. "Gaussian Process Optimization in the Bandit Setting: No Regret and Experimental Design." [arXiv:0912.3995](https://arxiv.org/abs/0912.3995) — The GP-UCB paper.

2. **Gupta et al. (2021)**. "Correlated Multi-Armed Bandits." [arXiv:1911.03959](https://arxiv.org/abs/1911.03959) — Formal correlated bandits framework, C-UCB algorithm.

3. **Li et al. (2010)**. "A Contextual-Bandit Approach to Personalized News Article Recommendation." [arXiv:1003.0146](https://arxiv.org/abs/1003.0146) — The LinUCB paper.

4. **Rasmussen & Williams (2006)**. "Gaussian Processes for Machine Learning." [gaussianprocess.org/gpml](http://gaussianprocess.org/gpml/) — The comprehensive GP textbook, free online.

5. **Russo et al. (2018)**. "A Tutorial on Thompson Sampling." [Stanford](https://web.stanford.edu/~bvr/pubs/TS_Tutorial.pdf) — Covers correlated priors in Section 5.

6. **Duvenaud (2014)**. "The Kernel Cookbook." [cs.toronto.edu/~duvenaud/cookbook](https://www.cs.toronto.edu/~duvenaud/cookbook/) — Practical guide to choosing and combining kernels.

7. **Distill (2019)**. "A Visual Exploration of Gaussian Processes." [distill.pub](https://distill.pub/2019/visual-exploration-gaussian-processes/) — Interactive GP visualizations.

8. **Valko et al. (2013)**. "Kernelised Contextual Bandits." [arXiv:1309.6869](https://arxiv.org/abs/1309.6869) — Kernel methods for contextual bandits.

9. **Janz et al. (2020)**. "Bandit Optimisation of Functions in the Matern Kernel RKHS." [arXiv:2001.10396](https://arxiv.org/abs/2001.10396) — Regret bounds for smooth kernels.
