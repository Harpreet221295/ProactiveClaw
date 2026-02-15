# Correlated Arms in Multi-Armed Bandits: A Complete Technical Deep Dive

## Applied to the 168-Arm Notification Scheduling Problem

---

## Table of Contents

1. [Motivation and Problem Statement](#1-motivation-and-problem-statement)
2. [Prerequisites](#2-prerequisites)
3. [Recommended Learning Path](#3-recommended-learning-path)
4. [Part I: Kernel Methods -- The Foundation](#4-part-i-kernel-methods)
5. [Part II: Gaussian Processes](#5-part-ii-gaussian-processes)
6. [Part III: GP-UCB Bandits](#6-part-iii-gp-ucb-bandits)
7. [Part IV: Contextual Bandits](#7-part-iv-contextual-bandits)
8. [Part V: Lipschitz Bandits](#8-part-v-lipschitz-bandits)
9. [Part VI: Kernel Smoothing / Nadaraya-Watson Estimator](#9-part-vi-kernel-smoothing)
10. [Part VII: Correlated Bandits -- Formal Theory](#10-part-vii-correlated-bandits)
11. [Part VIII: Thompson Sampling with Correlated Priors](#11-part-viii-thompson-sampling-with-correlated-priors)
12. [Part IX: Periodic Kernels for Time Structure](#12-part-ix-periodic-kernels)
13. [Part X: Toroidal Distance Metrics](#13-part-x-toroidal-distance-metrics)
14. [Part XI: Putting It All Together -- The Notification Scheduling Kernel](#14-part-xi-putting-it-all-together)
15. [Part XII: Intermediate Approaches (Simpler than Full GP)](#15-part-xii-intermediate-approaches)
16. [Complete Reference List](#16-complete-reference-list)

---

## 1. Motivation and Problem Statement

### The Current System

Our notification scheduler uses a **168-arm bandit** (7 days x 24 hours). Each arm `(day, hour)` represents a time slot. The current implementation in `bandit.py` treats each arm **independently**: when the user responds to a Monday-9AM notification, only the `(Monday, 9AM)` arm gets updated. Tuesday-9AM learns nothing. Monday-10AM learns nothing.

The current update rule is a time-decayed running average:

```
Q_new(arm) = decay * Q_old(arm) + (1 - decay) * reward
where decay = exp(-lambda * days_since_last_update)
```

### The Problem with Independence

With 168 independent arms, each arm must be explored individually. If the user tends to respond well at 9AM on weekdays, the system must separately discover this for Monday-9AM, Tuesday-9AM, Wednesday-9AM, etc. This wastes the user's patience (many poorly-timed notifications during exploration) and converges slowly.

### What We Want: Reward Propagation

When the user responds positively to a Monday-9AM notification, we want:
- **Temporal neighbors**: Monday-8AM and Monday-10AM should get partial credit (nearby hours are likely similar)
- **Same-hour-different-day neighbors**: Tuesday-9AM, Wednesday-9AM should get partial credit (the user's 9AM availability likely persists across weekdays)
- **Decaying influence**: Monday-3PM should get very little credit (far from 9AM)
- **Cyclical awareness**: Sunday-11PM and Monday-12AM should be considered neighbors (they are 1 hour apart, not 23 hours apart)

This is the **correlated arms** problem. The rest of this document builds every concept needed to solve it rigorously.

---

## 2. Prerequisites

### 2.1 Linear Algebra (Essential)

**Vectors and matrices**: You need comfort with matrix-vector multiplication, transposes, and inverses.

**Positive semi-definite (PSD) matrices**: A symmetric matrix K in R^{n x n} is PSD if and only if:

```
v^T K v >= 0    for all v in R^n
```

Equivalently, all eigenvalues of K are non-negative. PSD matrices are the multivariate generalization of "non-negative numbers." They appear everywhere in this document because:
- Covariance matrices are always PSD
- Kernel matrices (Gram matrices) must be PSD
- The GP posterior covariance is PSD

**Eigendecomposition**: For a symmetric matrix K = U Lambda U^T where U is orthogonal and Lambda is diagonal with eigenvalues. This matters because:
- Kernel properties are often characterized by their spectrum (eigenvalues)
- Computational shortcuts exploit eigendecompositions
- Regret bounds depend on eigenvalue decay rates

**Matrix inversion**: The GP posterior requires inverting an n x n matrix. The Woodbury identity and Cholesky decomposition are important computational tools:

```
Cholesky: K = L L^T  (L lower triangular)
Solve K^{-1} y  via  L z = y  then  L^T x = z
Cost: O(n^3) for decomposition, O(n^2) for each solve
```

**Block matrix algebra**: For the joint Gaussian conditioning derivation, you need the partitioned matrix inverse formula:

```
[A  B]^{-1}    involves    A - B D^{-1} C    (Schur complement)
[C  D]
```

### 2.2 Probability and Statistics (Essential)

**Multivariate Gaussian distribution**: A random vector x in R^n follows N(mu, Sigma) with density:

```
p(x) = (2 pi)^{-n/2} |Sigma|^{-1/2} exp(-1/2 (x - mu)^T Sigma^{-1} (x - mu))
```

**Conditional Gaussians** (the single most important derivation for GPs): If

```
[x_1]     [mu_1]   [Sigma_11  Sigma_12]
[x_2] ~ N([mu_2] , [Sigma_21  Sigma_22])
```

then the conditional distribution is:

```
x_1 | x_2 ~ N(mu_1 + Sigma_12 Sigma_22^{-1} (x_2 - mu_2),
              Sigma_11 - Sigma_12 Sigma_22^{-1} Sigma_21)
```

This single formula is the **entire engine** behind Gaussian Process regression. Every GP posterior formula is an instance of this.

**Bayes' theorem**: Prior * Likelihood = Posterior (up to normalization). Thompson Sampling and GP inference are Bayesian.

**Expectation, variance, covariance**: Cov(X, Y) = E[XY] - E[X]E[Y]. Covariance measures linear co-variation.

### 2.3 Optimization (Helpful)

**Convex optimization basics**: UCB-style algorithms maximize an acquisition function. Understanding when this is tractable matters.

**Gradient-based methods**: Kernel hyperparameter optimization (length-scales, etc.) typically uses gradient descent on the marginal likelihood.

### 2.4 Basic Bandit Theory (Helpful)

**Exploration-exploitation tradeoff**: The fundamental tension -- exploit what you know (pull the best arm) vs. explore what you don't (try uncertain arms).

**Regret**: The cumulative difference between the optimal arm's reward and what you actually received:

```
R_T = sum_{t=1}^{T} [f(x*) - f(x_t)]
```

where x* is the optimal arm and x_t is the arm pulled at time t. Sublinear regret R_T = o(T) means the algorithm learns.

**UCB principle**: Choose the arm with the highest upper confidence bound:

```
x_t = argmax_x [mu(x) + beta * sigma(x)]
```

where mu(x) is the estimated mean, sigma(x) is the uncertainty, and beta controls exploration.

---

## 3. Recommended Learning Path

### Level 0: Foundations (if needed)
1. Linear algebra (Gilbert Strang's course or 3Blue1Brown's "Essence of Linear Algebra")
2. Probability (any undergraduate text, focus on multivariate Gaussians)
3. Standard multi-armed bandits (Lattimore & Szepesvari, "Bandit Algorithms", Chapters 1-7)

### Level 1: Kernel Smoothing (simplest approach, implement first)
1. **Nadaraya-Watson estimator** (Section 9 of this document)
2. **Toroidal distance** (Section 13)
3. **Implementation**: Modify the current `bandit.py` to propagate rewards using kernel weights

This gives you 80% of the benefit with 20% of the complexity. No GP needed.

### Level 2: Kernel Methods + Gaussian Processes
1. **Kernel functions** (Section 4)
2. **Gaussian Processes** (Section 5) -- Rasmussen & Williams textbook, Chapters 1-2
3. **Periodic kernels** (Section 12)

### Level 3: GP Bandits
1. **GP-UCB** (Section 6) -- Srinivas et al. 2010
2. **Thompson Sampling with correlated priors** (Section 11)
3. **Correlated bandits formal theory** (Section 10)

### Level 4: Advanced Theory
1. **Contextual bandits** (Section 7)
2. **Lipschitz bandits** (Section 8)
3. **RKHS theory** and regret bound proofs

### Recommended "Just Build It" Path
```
Current system (independent arms)
        |
        v
Add Nadaraya-Watson kernel smoothing with toroidal distance  [Level 1, ~50 lines of code]
        |
        v
Add periodic kernel for hour-of-day and day-of-week          [Level 1.5, ~30 more lines]
        |
        v
Switch to Thompson Sampling with correlated Beta priors       [Level 2, ~100 lines]
        |
        v
Full GP-UCB or GP-TS with composite periodic kernel           [Level 3, use GPyTorch]
```

---

## 4. Part I: Kernel Methods -- The Foundation

### 4.1 What Is a Kernel?

**Intuitive explanation**: A kernel is a function that measures similarity between two points. Given two inputs x and x', the kernel k(x, x') returns a real number indicating "how similar" they are. Higher values mean more similar.

**Formal definition**: A kernel is a function k: X x X -> R that is:
1. **Symmetric**: k(x, x') = k(x', x)
2. **Positive semi-definite**: For any finite set of points {x_1, ..., x_n}, the **Gram matrix** K with entries K_{ij} = k(x_i, x_j) is positive semi-definite

The PSD condition means: for all vectors c in R^n,

```
sum_{i=1}^{n} sum_{j=1}^{n} c_i c_j k(x_i, x_j) >= 0
```

This is equivalent to saying: the Gram matrix K has all non-negative eigenvalues.

**Why PSD matters**: The PSD property guarantees that the kernel corresponds to an inner product in some (possibly infinite-dimensional) feature space. By Mercer's theorem, any continuous PSD kernel can be written as:

```
k(x, x') = <phi(x), phi(x')>_H = sum_{i=1}^{infinity} lambda_i phi_i(x) phi_i(x')
```

where phi: X -> H is a feature map into a Hilbert space H, lambda_i are non-negative eigenvalues, and phi_i are eigenfunctions.

**Connection to notification scheduling**: The kernel will encode our belief about which time slots are similar. k((Monday, 9AM), (Monday, 10AM)) should be large (adjacent hours, same day). k((Monday, 9AM), (Thursday, 3AM)) should be small.

### 4.2 Common Kernel Functions

#### 4.2.1 RBF (Radial Basis Function) / Squared Exponential / Gaussian Kernel

```
k_RBF(x, x') = sigma_f^2 * exp(-||x - x'||^2 / (2 * l^2))
```

**Parameters**:
- `sigma_f^2`: Signal variance (overall vertical scale of the function)
- `l`: Length-scale (how far apart inputs must be before the function values become uncorrelated)

**Properties**:
- Infinitely differentiable (very smooth functions)
- Stationary: k depends only on x - x', not on x and x' individually
- Isotropic: k depends only on ||x - x'||, not the direction
- k(x, x) = sigma_f^2 (self-similarity is maximal)
- k(x, x') -> 0 as ||x - x'|| -> infinity (distant points are uncorrelated)

**For notification scheduling**: If we naively use RBF with Euclidean distance over (day, hour) pairs, Monday-11PM and Tuesday-12AM would appear far apart. This is wrong -- they are 1 hour apart. We need cyclical distance (see Section 13) or periodic kernels (see Section 12).

#### 4.2.2 Matern Kernel Family

The Matern kernel generalizes the RBF to control smoothness:

```
k_Matern(x, x') = sigma_f^2 * (2^{1-nu} / Gamma(nu)) * (sqrt(2*nu) * r / l)^nu * K_nu(sqrt(2*nu) * r / l)
```

where r = ||x - x'||, K_nu is the modified Bessel function of the second kind, and nu > 0 controls smoothness.

**Special cases (half-integer nu)**: When nu = p + 1/2 for integer p, the Matern simplifies to exponential times polynomial:

**nu = 1/2** (Ornstein-Uhlenbeck process, very rough):
```
k(x, x') = sigma_f^2 * exp(-r / l)
```

**nu = 3/2** (once differentiable):
```
k(x, x') = sigma_f^2 * (1 + sqrt(3)*r/l) * exp(-sqrt(3)*r/l)
```

**nu = 5/2** (twice differentiable):
```
k(x, x') = sigma_f^2 * (1 + sqrt(5)*r/l + 5*r^2/(3*l^2)) * exp(-sqrt(5)*r/l)
```

**nu -> infinity**: Converges to the RBF kernel (infinitely smooth).

**For notification scheduling**: User engagement patterns are probably smooth but not infinitely smooth. Matern-5/2 or Matern-3/2 are more realistic than RBF. A user's responsiveness doesn't change infinitely smoothly -- there are transitions (e.g., arriving at work, going to sleep) that create moderate roughness.

#### 4.2.3 Periodic Kernel (Exponential Sine Squared)

```
k_periodic(x, x') = sigma_f^2 * exp(-2 * sin^2(pi * |x - x'| / p) / l^2)
```

**Parameters**:
- `p`: Period (the cycle length)
- `l`: Length-scale within each period
- `sigma_f^2`: Signal variance

**Properties**:
- Exactly periodic: k(x, x') = k(x + p, x' + p) = k(x, x' + p)
- Captures repeating patterns without decay
- Combined with RBF, creates "locally periodic" patterns (periodicity that can evolve)

**For notification scheduling**: This is the **key kernel** for our problem.
- Set p = 24 to capture daily periodicity in hours
- Set p = 7 to capture weekly periodicity in days
- The combination captures "every Monday at 9AM" patterns

#### 4.2.4 Linear Kernel

```
k_linear(x, x') = sigma_b^2 + sigma_v^2 * (x - c) * (x' - c)
```

Corresponds to Bayesian linear regression. Less relevant for our cyclical problem but used in LinUCB.

### 4.3 Kernel Algebra: Building Complex Kernels

A critical property: **kernels are closed under addition and multiplication**. If k_1 and k_2 are valid kernels, then:

- **Sum**: k(x, x') = k_1(x, x') + k_2(x, x') is a valid kernel
  - Interpretation: f = f_1 + f_2 where f_1 ~ GP(0, k_1) and f_2 ~ GP(0, k_2)
  - Models superposition of patterns

- **Product**: k(x, x') = k_1(x, x') * k_2(x, x') is a valid kernel
  - Interpretation: correlation is high only if BOTH kernels give high correlation
  - Models interaction of patterns

- **Scalar multiplication**: k(x, x') = c * k_1(x, x') for c > 0 is a valid kernel

**Proof that sum preserves PSD**: For any c in R^n,
```
c^T (K_1 + K_2) c = c^T K_1 c + c^T K_2 c >= 0 + 0 = 0
```
since both K_1 and K_2 are PSD.

**Proof that product preserves PSD**: The Hadamard (element-wise) product of two PSD matrices is PSD (Schur product theorem). Since (K_1 o K_2)_{ij} = k_1(x_i, x_j) * k_2(x_i, x_j), the Gram matrix of the product kernel is the Hadamard product of the individual Gram matrices.

**Locally periodic kernel** (crucial for us):
```
k_locally_periodic(x, x') = k_periodic(x, x') * k_RBF(x, x')
```

This creates periodic patterns that can change amplitude over time. The periodic kernel provides the repeating structure, and the RBF modulates it so that correlations decay with overall distance.

### 4.4 Kernels on Structured Domains

For our 2D grid of (day, hour), we can define **product kernels** over each dimension:

```
k((d, h), (d', h')) = k_day(d, d') * k_hour(h, h')
```

Or **additive kernels**:

```
k((d, h), (d', h')) = k_day(d, d') + k_hour(h, h')
```

**Product**: High similarity only when BOTH day AND hour are similar. This means Monday-9AM is similar to Tuesday-10AM only if both the day kernel and hour kernel agree.

**Additive**: Captures independent effects. A "good hour" effect (9AM is good regardless of day) plus a "good day" effect (Monday is good regardless of hour). Additive kernels allow the GP posterior to decompose into interpretable components.

### 4.5 Reproducing Kernel Hilbert Spaces (RKHS)

The RKHS H_k associated with kernel k is the function space:

```
H_k = {f : f(x) = sum_{i=1}^{N} alpha_i k(x, x_i) for some N, alpha_i, x_i}
```

completed under the inner product:

```
<f, g>_{H_k} = sum_{i,j} alpha_i beta_j k(x_i, x_j)
```

**The reproducing property**: For any f in H_k,

```
f(x) = <f, k(x, .)>_{H_k}
```

This means evaluation at a point x is a bounded linear functional, which is the defining property of an RKHS.

**RKHS norm**: ||f||_{H_k} measures the "complexity" or "roughness" of f with respect to kernel k. Functions with small RKHS norm are "smooth" relative to k. The GP-UCB regret analysis assumes f lies in the RKHS ball {f : ||f||_{H_k} <= B} for some bound B.

**For notification scheduling**: The RKHS norm of our reward function f(day, hour) being bounded means that user engagement patterns are "smooth" in the sense defined by our kernel -- nearby time slots have similar engagement rates. This is a reasonable assumption.

### References for Section 4
- Rasmussen & Williams (2006), "Gaussian Processes for Machine Learning", Chapter 4
- Scholkopf & Smola (2002), "Learning with Kernels"
- Duvenaud (2014), "Automatic Model Construction with Gaussian Processes" (PhD thesis, Chapter 2 -- the "Kernel Cookbook": https://www.cs.toronto.edu/~duvenaud/cookbook/)
- Bach (2024), "Learning Theory from First Principles", Lecture 6

---

## 5. Part II: Gaussian Processes

### 5.1 Definition

**Intuitive explanation**: A Gaussian Process is a distribution over functions. Just as a Gaussian distribution describes uncertainty over a number, a GP describes uncertainty over an entire function. Any finite collection of function values follows a multivariate Gaussian distribution.

**Formal definition**: A Gaussian Process is a collection of random variables, any finite number of which have a joint Gaussian distribution. A GP is fully specified by:

- **Mean function**: m(x) = E[f(x)]
- **Covariance function (kernel)**: k(x, x') = E[(f(x) - m(x))(f(x') - m(x'))]

We write:

```
f ~ GP(m(x), k(x, x'))
```

This means: for any finite set of points X = {x_1, ..., x_n},

```
[f(x_1), ..., f(x_n)]^T ~ N(m, K)
```

where m_i = m(x_i) and K_{ij} = k(x_i, x_j).

**For notification scheduling**: We place a GP prior over the reward function f(day, hour). Before any data, we believe f is drawn from GP(0, k) where k encodes our structural beliefs (periodicity, smoothness). After observing user responses, the GP posterior gives us updated beliefs about f at ALL 168 time slots simultaneously.

### 5.2 The GP Prior

We typically assume a zero-mean prior: m(x) = 0. This is not restrictive because:
1. The mean can be absorbed into the kernel
2. With enough data, the posterior mean adapts to the true function
3. For our notification problem, we have no prior bias toward any time slot being better

The prior is entirely characterized by the kernel k(x, x'). Before seeing any data:

```
f ~ GP(0, k)
```

At any set of test points X_*, the prior distribution is:

```
f_* ~ N(0, K_{**})
```

where (K_{**})_{ij} = k(x_*^i, x_*^j). Functions sampled from this prior are random but respect the smoothness and periodicity encoded in k.

### 5.3 GP Regression: The Posterior

**Setup**: We have observed noisy outputs y = f(X) + epsilon where epsilon ~ N(0, sigma_n^2 I) at training points X = {x_1, ..., x_n}. We want to predict f at test points X_*.

**Joint distribution**: Under the GP prior, the joint distribution of observed values y and latent function values f_* at test points is:

```
[y  ]     [K(X,X) + sigma_n^2 I    K(X, X_*) ]
[f_*] ~ N([K(X_*, X)                K(X_*, X_*)])
```

with zero mean (for simplicity).

**Derivation of the posterior**: Apply the conditional Gaussian formula from Section 2.2:

```
f_* | X, y, X_* ~ N(mu_*, Sigma_*)
```

where:

```
mu_*    = K(X_*, X) [K(X, X) + sigma_n^2 I]^{-1} y

Sigma_* = K(X_*, X_*) - K(X_*, X) [K(X, X) + sigma_n^2 I]^{-1} K(X, X_*)
```

**Step-by-step derivation**:

Let:
- K = K(X, X) + sigma_n^2 I  (n x n training covariance + noise)
- K_* = K(X, X_*)  (n x n_* cross-covariance)
- K_{**} = K(X_*, X_*)  (n_* x n_* test covariance)

The joint is:
```
[y  ]     [K       K_* ]
[f_*] ~ N([K_*^T   K_{**}])
```

Applying the conditional formula x_2 | x_1 ~ N(mu_2 + Sigma_21 Sigma_11^{-1} (x_1 - mu_1), Sigma_22 - Sigma_21 Sigma_11^{-1} Sigma_12):

```
f_* | y ~ N(K_*^T K^{-1} y,  K_{**} - K_*^T K^{-1} K_*)
```

**Interpretation**:
- **Posterior mean** mu_*(x) = K_*^T K^{-1} y: A linear combination of observed outputs y, weighted by the kernel similarities between test and training points, adjusted by the inverse of the training covariance.
- **Posterior variance** is reduced from the prior variance K_{**} by the term K_*^T K^{-1} K_*, which represents how much information the training points provide about the test points.
- Variance is **zero** at training points (in the noiseless case) and approaches the prior variance far from training points.

**For notification scheduling**: After observing that the user responded (reward=1) at Monday-9AM and did not respond (reward=0) at Wednesday-3AM:
- The posterior mean at Monday-10AM will be high (close to Monday-9AM, correlated by kernel)
- The posterior mean at Tuesday-9AM will be moderately high (same hour, different day)
- The posterior mean at Saturday-3AM will be low (close to Wednesday-3AM's bad result)
- The posterior variance at Friday-6PM will still be high (no nearby observations)

**This is exactly the "reward propagation" we want.**

### 5.4 Marginal Likelihood and Hyperparameter Optimization

The kernel has hyperparameters theta = {sigma_f, l, p, sigma_n, ...}. We optimize them by maximizing the **log marginal likelihood**:

```
log p(y | X, theta) = -1/2 y^T K^{-1} y - 1/2 log |K| - n/2 log(2*pi)
```

where K = K(X, X) + sigma_n^2 I depends on theta.

**Three terms**:
1. `-1/2 y^T K^{-1} y`: Data fit term. Prefers models that explain the data.
2. `-1/2 log |K|`: Complexity penalty. Prefers simpler models (prevents overfitting).
3. `-n/2 log(2*pi)`: Normalization constant.

**Gradient**:

```
d/d(theta_j) log p(y | X, theta) = 1/2 tr((alpha alpha^T - K^{-1}) dK/d(theta_j))
```

where alpha = K^{-1} y. This gradient can be computed and used with any gradient-based optimizer.

**For notification scheduling**: Hyperparameter optimization will automatically learn:
- How quickly engagement decays with hour difference (hour length-scale)
- How much day-of-week matters (day length-scale)
- How noisy user responses are (sigma_n)
- Whether patterns are truly periodic (period parameters)

### 5.5 Computational Complexity

- **Training (computing the posterior)**: O(n^3) for matrix inversion (or Cholesky decomposition)
- **Prediction (at n_* new points)**: O(n^2 * n_*) for mean, O(n^2 * n_*) for variance
- **Storage**: O(n^2) for the kernel matrix

For our problem: n is the number of observed user responses (grows over time). With n=168 (one observation per slot), the 168 x 168 matrix is trivial. Even with n=10,000 observations, modern machines handle it. For very large n, approximations exist (sparse GPs, random Fourier features).

### References for Section 5
- Rasmussen & Williams (2006), "Gaussian Processes for Machine Learning", Chapters 2-5 (free online: http://gaussianprocess.org/gpml/)
- Chuong B. Do (2007), Stanford CS229 GP notes
- Distill.pub (2019), "A Visual Exploration of Gaussian Processes" (https://distill.pub/2019/visual-exploration-gaussian-processes/)

---

## 6. Part III: GP-UCB (Gaussian Process Upper Confidence Bound) Bandits

### 6.1 Problem Setting

We want to sequentially optimize an unknown function f: X -> R under noisy observations. At each round t:
1. Choose a point x_t in X
2. Observe y_t = f(x_t) + epsilon_t where epsilon_t ~ N(0, sigma_n^2)
3. Goal: minimize cumulative regret R_T = sum_{t=1}^T [f(x*) - f(x_t)]

**Assumption**: f is drawn from a GP, or more generally, f has bounded RKHS norm: ||f||_{H_k} <= B.

**For notification scheduling**: X = {(d, h) : d in {0,...,6}, h in {0,...,23}} with |X| = 168. f(d, h) = probability of user engagement at day d, hour h. We observe noisy binary rewards (responded or not).

### 6.2 The GP-UCB Algorithm

**Algorithm** (Srinivas et al., 2010):

```
Input: Domain X, kernel k, noise variance sigma_n^2, confidence schedule {beta_t}
Initialize: D_0 = {} (empty dataset)

For t = 1, 2, 3, ...:
    1. Compute GP posterior given D_{t-1}:
       mu_{t-1}(x) = k(x, X_{1:t-1})^T [K_{t-1} + sigma_n^2 I]^{-1} y_{1:t-1}
       sigma_{t-1}^2(x) = k(x, x) - k(x, X_{1:t-1})^T [K_{t-1} + sigma_n^2 I]^{-1} k(x, X_{1:t-1})

    2. Select arm with highest UCB:
       x_t = argmax_{x in X}  [mu_{t-1}(x) + sqrt(beta_t) * sigma_{t-1}(x)]

    3. Observe y_t = f(x_t) + epsilon_t

    4. Update dataset: D_t = D_{t-1} union {(x_t, y_t)}
```

### 6.3 The Acquisition Function

The UCB acquisition function at time t is:

```
alpha_t(x) = mu_{t-1}(x) + sqrt(beta_t) * sigma_{t-1}(x)
```

**Decomposition**:
- `mu_{t-1}(x)`: Exploitation term. Favors points with high predicted reward.
- `sqrt(beta_t) * sigma_{t-1}(x)`: Exploration term. Favors points with high uncertainty.
- `beta_t`: Controls the exploration-exploitation tradeoff. Larger beta_t means more exploration.

**Why UCB works with GPs**: The GP posterior provides calibrated uncertainty. Points that have been observed have low sigma (well-known). Points far from observations (in kernel space) have high sigma (uncertain). The kernel propagates information: observing one point reduces uncertainty at similar points.

### 6.4 The Beta Parameter

**For finite domains** (our case, |X| = 168):

Srinivas et al. (2010), Theorem 1:

```
beta_t = 2 * log(|X| * t^2 * pi^2 / (6 * delta))
```

where delta in (0, 1) is the confidence parameter (typically 0.1).

For our problem with |X| = 168 and delta = 0.1:

```
beta_t = 2 * log(168 * t^2 * pi^2 / 0.6)
       = 2 * log(168 * t^2 * 16.45)
       = 2 * log(2763.6 * t^2)
```

At t = 100: beta_100 = 2 * log(2763.6 * 10000) = 2 * log(27,636,000) ~ 2 * 17.13 ~ 34.3

In practice, beta_t is often set to a smaller constant (e.g., beta = 2) because the theoretical values are conservative.

### 6.5 Information Gain and Regret Bounds

**Maximum Information Gain**: The key quantity in GP-UCB regret analysis:

```
gamma_T = max_{A subset X, |A|=T}  (1/2) * log |I + sigma_n^{-2} K_A|
```

where K_A is the kernel matrix over the selected points A. This measures the maximum amount of information (in nats) about f that can be obtained from T observations.

**Cumulative regret bound** (Theorem 1, Srinivas et al. 2010):

With probability >= 1 - delta:

```
R_T <= sqrt(C_1 * T * beta_T * gamma_T)
```

where C_1 = 8 / log(1 + sigma_n^{-2}).

**Information gain for common kernels**:

| Kernel | gamma_T | Regret R_T |
|--------|---------|------------|
| Linear | O(d * log T) | O(sqrt(d * T * log T)) |
| RBF/SE | O((log T)^{d+1}) | O(sqrt(T * (log T)^{d+1})) |
| Matern (nu) | O(T^{d(d+2nu)/(2nu+d(d+2nu))} * (log T)) | Depends on nu, d |

For our problem (d=2, finite domain of 168 points):
- gamma_T <= T * log(168) in the worst case (but much less for smooth kernels)
- The RBF kernel gives gamma_T = O((log T)^3) which yields near-logarithmic regret

**Why information gain matters**: It captures the effective dimensionality of the problem. If the kernel has rapidly decaying eigenvalues (very smooth functions), gamma_T is small, and the regret is low. The kernel structure buys us faster learning.

### 6.6 Comparison to Standard UCB1

Standard UCB1 treats all 168 arms independently. Its regret:

```
R_T^{UCB1} = O(sqrt(168 * T * log T))
```

GP-UCB with a smooth kernel:

```
R_T^{GP-UCB} = O(sqrt(T * (log T)^3))   (for RBF kernel, d=2)
```

The improvement: GP-UCB's regret does not depend (polynomially) on the number of arms. It depends on the "effective dimensionality" determined by the kernel. With 168 arms but strong correlations, the effective dimensionality might be ~5-10, giving enormous speedup.

### 6.7 GP-UCB for Notification Scheduling

At each interaction opportunity:

1. Maintain GP posterior over all 168 slots using all past observations
2. Compute UCB for each slot: mu(d, h) + sqrt(beta_t) * sigma(d, h)
3. Schedule notification at the slot with highest UCB
4. After observing user response, update the GP posterior

The kernel automatically propagates the observation to all correlated slots. Observing a reward at Monday-9AM will:
- Increase mu at Tuesday-9AM (correlated by hour)
- Increase mu at Monday-10AM (correlated by time proximity)
- Decrease sigma at all nearby slots (reduced uncertainty)

### References for Section 6
- Srinivas, Krause, Kakade, Seeger (2010), "Gaussian Process Optimization in the Bandit Setting: No Regret and Experimental Design" (https://arxiv.org/abs/0912.3995)
- Srinivas et al. (2012), IEEE Transactions on Information Theory version (https://ieeexplore.ieee.org/document/6138914/)
- Chowdhury & Gopalan (2017), "On Kernelized Multi-armed Bandits"

---

## 7. Part IV: Contextual Bandits

### 7.1 What Are Contextual Bandits?

**Intuitive explanation**: In standard MAB, you always face the same set of arms. In contextual bandits, each round comes with a **context** (side information) that affects which arm is best. The optimal arm depends on the context.

**Formal definition**: At each round t:
1. Nature reveals context c_t in C
2. Agent chooses arm a_t in A
3. Agent observes reward r_t = f(c_t, a_t) + epsilon_t

The reward function f(c, a) depends on both the context and the arm.

**For notification scheduling**: Context could include:
- Day of week (already in our arm definition)
- User's recent activity level
- Weather, holidays, calendar events
- Time since last notification
- User's current location

### 7.2 Linear Contextual Bandits (LinUCB)

**Assumption**: The expected reward is linear in the feature vector:

```
E[r_t | c_t, a_t] = theta_a^T * phi(c_t, a_t)
```

where phi(c, a) in R^d is a feature vector and theta_a in R^d is the unknown parameter for arm a.

**The LinUCB Algorithm** (Li et al., 2010):

```
For each arm a, maintain:
  A_a = I_d + sum_{t: a_t = a} phi_t phi_t^T   (d x d design matrix)
  b_a = sum_{t: a_t = a} r_t * phi_t            (d x 1 reward-weighted features)

At each round t with context c_t:
  For each arm a:
    theta_hat_a = A_a^{-1} b_a
    alpha_t(a) = theta_hat_a^T phi(c_t, a) + alpha * sqrt(phi(c_t, a)^T A_a^{-1} phi(c_t, a))
  Choose a_t = argmax_a alpha_t(a)
```

**The UCB term**: `sqrt(phi^T A^{-1} phi)` is the posterior standard deviation under a Gaussian prior on theta.

**Regret bound**: O(d * sqrt(T * log T)) -- depends on feature dimension d, not number of arms.

### 7.3 Kernel Contextual Bandits (KernelUCB)

**Motivation**: What if the reward is a nonlinear function of context and arm?

**KernelUCB** (Valko et al., 2013): Replaces the linear model with a kernel model. The algorithm works in the RKHS associated with kernel k:

```
At round t:
  For each arm a:
    mu_t(c_t, a) = k_t^T (K_t + lambda I)^{-1} r_{1:t-1}
    sigma_t^2(c_t, a) = k(c_t a, c_t a) - k_t^T (K_t + lambda I)^{-1} k_t
    UCB_t(a) = mu_t(c_t, a) + beta_t * sigma_t(c_t, a)
  Choose a_t = argmax_a UCB_t(a)
```

where k_t is the vector of kernel evaluations between the current context-arm pair and all previous ones.

**Key insight**: When the kernel is just the dot product k(x, x') = x^T x', KernelUCB reduces exactly to LinUCB. KernelUCB is a **nonlinear generalization** of LinUCB.

**Regret bound**: O(sqrt(T * d_tilde)) where d_tilde is the effective dimension of the data in the feature space.

### 7.4 Relationship to Our Problem

Our current 168-arm formulation is a **non-contextual** bandit: the arm (day, hour) is the "context" baked into the arm definition. But we could reformulate it as contextual:

**Reformulation**: A single arm ("send notification"), with context c_t = (day_of_week, hour, other_features). The reward depends on when we send, captured through the context.

**Advantages of contextual formulation**:
- Can incorporate additional features (user activity, weather, etc.)
- LinUCB with cyclical features (sin/cos encoding of hour/day) gives a simple parametric model with built-in periodicity
- Can handle non-stationary environments by including time as a feature

**Simple cyclical feature encoding for LinUCB**:
```
phi(day, hour) = [sin(2*pi*hour/24), cos(2*pi*hour/24),
                  sin(2*pi*day/7),   cos(2*pi*day/7),
                  sin(2*pi*hour/12), cos(2*pi*hour/12),  // second harmonic
                  sin(2*pi*day/3.5), cos(2*pi*day/3.5),  // second harmonic
                  1]                                       // bias term
```

This 9-dimensional feature vector captures the fundamental periodicity. LinUCB with these features learns in O(9 * sqrt(T)) instead of O(sqrt(168 * T)).

### References for Section 7
- Li, Chu, Langford, Schapire (2010), "A Contextual-Bandit Approach to Personalized News Article Recommendation" (https://arxiv.org/abs/1003.0146)
- Valko, Korda, Munos, Flaounas, Cristianini (2013), "Finite-Time Analysis of Kernelised Contextual Bandits"
- Chu, Li, Reyzin, Schapire (2011), "Contextual Bandits with Linear Payoff Functions"

---

## 8. Part V: Lipschitz Bandits

### 8.1 The Smoothness Assumption

**Intuitive explanation**: Lipschitz bandits assume that the reward function doesn't change too quickly. If two arms are "close" (in some metric), their rewards must be similar. This is a weaker assumption than the GP assumption -- we don't need a probabilistic model, just a bound on how fast the function can change.

**Formal definition**: A function f: X -> R is L-Lipschitz with respect to metric d if:

```
|f(x) - f(x')| <= L * d(x, x')    for all x, x' in X
```

**For notification scheduling**: If f is the true engagement function and d is our toroidal distance on the (day, hour) grid, then L-Lipschitz means: the engagement at Monday-9AM and Monday-10AM can differ by at most L * 1 (since they are 1 hour apart).

### 8.2 Continuum-Armed Bandits with Lipschitz Rewards

**Setting**: The arm space is continuous, typically X = [0, 1]^d, and f is Lipschitz.

**Naive uniform discretization**: Divide [0, 1]^d into a grid with spacing epsilon. This gives (1/epsilon)^d arms. Apply standard UCB to these arms. With epsilon = T^{-1/(d+2)}, the regret is:

```
R_T = O(T^{(d+1)/(d+2)})
```

For d=2 (our day-hour grid): R_T = O(T^{3/4}).

### 8.3 The Zooming Algorithm (Kleinberg, Slivkins, Upfal, 2008-2019)

**Key insight**: Don't waste time exploring regions that are clearly suboptimal. Focus resolution (zoom in) near the optimum.

**Zooming dimension**: Instead of the ambient dimension d, the regret depends on the **zooming dimension** d_z, which measures the effective complexity near the optimum. Often d_z << d.

**Algorithm sketch**:
1. Maintain a set of "active balls" covering the arm space
2. Each ball has a center (arm), radius, and confidence interval for the reward
3. At each step, select the arm with the highest UCB among active balls
4. When an arm's confidence interval tightens sufficiently, split its ball into smaller balls near it
5. Prune balls that are provably suboptimal

**Regret bound**:

```
R_T = O(T^{(d_z + 1)/(d_z + 2)} * (log T)^{1/(d_z + 2)})
```

**Zooming continuity**: The algorithm exploits a weaker condition than global Lipschitz. It needs smoothness only near the optimal arm. If the reward function is smooth near the peak but rough elsewhere, zooming still works well.

### 8.4 Higher-Order Smoothness

If f has M Lipschitz derivatives (Holder smoothness of order alpha = M + beta for some beta in (0, 1]):

```
R_T = O(T^{(d + alpha)/(2*alpha + d)})
```

For alpha = 2 (twice differentiable), d = 2: R_T = O(T^{4/6}) = O(T^{2/3}).

The smoother the function, the faster we learn. This motivates using smooth kernels in GP-based approaches.

### 8.5 Connection to GP Bandits

Lipschitz bandits and GP bandits make different assumptions:
- **Lipschitz**: Worst-case smoothness bound. No probabilistic model. Robust but conservative.
- **GP**: Probabilistic model. Stronger assumptions but faster learning if assumptions hold.

The RKHS perspective bridges them: if f has bounded RKHS norm ||f||_{H_k} <= B for a Matern-nu kernel, then f is Holder smooth with order nu - d/2 (when nu > d/2). So GP assumptions imply Lipschitz-type smoothness.

### 8.6 Application to Notification Scheduling

The Lipschitz framework tells us: if engagement changes smoothly across time slots, we don't need to try all 168 independently. The zooming algorithm would naturally:
1. Start by coarsely exploring (morning vs. evening, weekday vs. weekend)
2. Zoom into promising regions (9-11AM on weekdays)
3. Fine-tune within those regions

However, for our problem, the GP approach is more natural because:
- We have a natural periodicity structure that Lipschitz methods don't exploit
- Our domain is small (168 points) and discrete
- The GP gives posterior uncertainty for principled exploration

### References for Section 8
- Kleinberg, Slivkins, Upfal (2019), "Bandits and Experts in Metric Spaces" (JACM)
- Bubeck, Munos, Stoltz, Szepesvari (2008), "Online Optimization in X-Armed Bandits" (NeurIPS)
- Kleinberg (2005), "Nearly Tight Bounds for the Continuum-Armed Bandit Problem"
- Grant, Leslie, Sherlock (2020), "On Thompson Sampling for Smoother-than-Lipschitz Bandits" (AISTATS)

---

## 9. Part VI: Kernel Smoothing / Nadaraya-Watson Estimator

### 9.1 Intuitive Explanation

**The idea**: Instead of maintaining independent Q-values for each arm, compute a smoothed estimate by taking a weighted average of all observed rewards, where the weights depend on the distance between arms. Arms that are "close" get higher weights.

This is the **simplest practical approach** to reward propagation and can be added to the current system with minimal code changes.

### 9.2 Mathematical Formulation

The **Nadaraya-Watson estimator** for the reward function at point x is:

```
f_hat(x) = sum_{i=1}^{n} W_i(x) * y_i
```

where the weights are:

```
W_i(x) = K_h(x, x_i) / sum_{j=1}^{n} K_h(x, x_j)
```

and K_h is a kernel function with bandwidth h:

```
K_h(x, x_i) = K(d(x, x_i) / h)
```

Here:
- d(x, x_i) is the distance between points x and x_i
- K is a kernel (weighting) function
- h is the bandwidth (controls smoothing width)
- y_i is the observed reward at point x_i

**The weights always sum to 1**: sum_i W_i(x) = 1. So f_hat(x) is a proper weighted average.

### 9.3 Common Kernel (Weighting) Functions

Note: These are "smoothing kernels" in the statistical sense (density functions used as weights), not the same as GP covariance kernels (though they are related).

**Gaussian (most common)**:
```
K(u) = (1/sqrt(2*pi)) * exp(-u^2 / 2)
```

**Epanechnikov (optimal in MSE sense)**:
```
K(u) = (3/4)(1 - u^2)  if |u| <= 1, else 0
```

**Uniform (box kernel)**:
```
K(u) = 1/2  if |u| <= 1, else 0
```

**Tricube**:
```
K(u) = (70/81)(1 - |u|^3)^3  if |u| <= 1, else 0
```

### 9.4 Bandwidth Selection

The bandwidth h controls the bias-variance tradeoff:
- **Small h**: Low bias (captures local features), high variance (noisy)
- **Large h**: High bias (oversmooths), low variance (stable)

**Silverman's rule of thumb** (for 1D):
```
h = 1.06 * sigma_hat * n^{-1/5}
```

where sigma_hat is the estimated standard deviation of the data.

**Cross-validation**: Leave one observation out, predict it from the rest, minimize prediction error:
```
h_CV = argmin_h sum_{i=1}^{n} (y_i - f_hat_{-i}(x_i))^2
```

where f_hat_{-i} is the estimate leaving out observation i.

For our problem with 168 discrete slots, the bandwidth h_hour and h_day should be tuned. Good starting values:
- h_hour = 2-3 (reward propagates 2-3 hours in each direction)
- h_day = 1-2 (reward propagates 1-2 days in each direction)

### 9.5 Nadaraya-Watson on the Notification Grid

For our 2D grid with toroidal distance (Section 13), the estimator becomes:

```
f_hat(d, h) = sum_{i=1}^{n} W_i(d, h) * y_i
```

where:

```
W_i(d, h) = K(d_hour(h, h_i) / bw_hour) * K(d_day(d, d_i) / bw_day) / Z(d, h)
```

and Z(d, h) is the normalizing constant ensuring weights sum to 1.

Here:
- d_hour(h, h_i) is the circular distance between hours (wrap-around at 24)
- d_day(d, d_i) is the circular distance between days (wrap-around at 7)
- bw_hour, bw_day are bandwidths for each dimension

### 9.6 Integration with the Current Bandit

**Approach 1: Smoothed Q-values** (simplest modification)

After updating arm (d, h) with reward r, propagate to all arms:

```python
def update_arm_with_propagation(day, hour, reward, bw_hour=2.0, bw_day=1.5):
    for d in range(7):
        for h in range(24):
            # Toroidal distance
            delta_h = min(abs(h - hour), 24 - abs(h - hour))
            delta_d = min(abs(d - day), 7 - abs(d - day))

            # Gaussian kernel weight
            weight = exp(-0.5 * (delta_h/bw_hour)**2 - 0.5 * (delta_d/bw_day)**2)

            if weight > 0.01:  # Skip negligible updates
                # Update with discounted reward
                update_arm(d, h, reward * weight)
```

**Approach 2: Smoothed readout** (alternative)

Keep the raw Q-values independent but smooth them when reading:

```python
def get_smoothed_q(day, hour, state, bw_hour=2.0, bw_day=1.5):
    numerator = 0.0
    denominator = 0.0
    for d in range(7):
        for h in range(24):
            key = f"{d}_{h}"
            q = state["q_values"].get(key, 0.0)
            delta_h = min(abs(h - hour), 24 - abs(h - hour))
            delta_d = min(abs(d - day), 7 - abs(d - day))
            weight = exp(-0.5 * (delta_h/bw_hour)**2 - 0.5 * (delta_d/bw_day)**2)
            numerator += weight * q
            denominator += weight
    return numerator / denominator if denominator > 0 else 0.0
```

Approach 2 is a true Nadaraya-Watson estimator and is generally preferable because it doesn't compound smoothing effects across multiple updates.

### 9.7 Properties and Limitations

**Advantages**:
- Extremely simple to implement (~20 lines of code)
- No matrix inversions, no GP machinery
- Computationally O(168) per prediction (trivial)
- Easy to understand and debug

**Limitations**:
- No principled uncertainty quantification (no sigma for exploration)
- Bandwidth must be hand-tuned or cross-validated
- No principled way to handle heterogeneous smoothness (e.g., sharper transitions at bedtime vs. gradual changes during daytime)
- Not a full probabilistic model

**For our "just get it working" goal**: Start here. This gives most of the reward propagation benefit with minimal complexity.

### References for Section 9
- Nadaraya (1964), "On Estimating Regression"
- Watson (1964), "Smooth Regression Analysis"
- Wasserman (2006), "All of Nonparametric Statistics", Chapter 5
- Wikipedia: Kernel regression (https://en.wikipedia.org/wiki/Kernel_regression)

---

## 10. Part VII: Correlated Bandits -- Formal Theory

### 10.1 Formal Definition

**Standard MAB**: K arms with independent reward distributions. Pulling arm i reveals information only about arm i.

**Correlated MAB**: K arms with a joint reward distribution. The rewards r_1, r_2, ..., r_K are correlated random variables drawn from a joint distribution P(r_1, ..., r_K). Pulling arm i and observing r_i updates beliefs about ALL arms r_j for j != i.

**Formal model** (Gupta, Granmo, Agrawala, 2021): There exists a latent random variable Z such that:

```
r_i = g_i(Z) + epsilon_i
```

where g_i are known (or partially known) functions and epsilon_i is independent noise. Observing r_i provides information about Z, which in turn provides information about all other arms.

### 10.2 Pseudo-Rewards and Correlation Structure

**Pseudo-rewards** (Gupta et al.): For arms i and j, define the pseudo-reward:

```
tilde{mu}_j(i, r_i) = E[r_j | r_i]
```

This is the expected reward of arm j given that we observed reward r_i from arm i. The pseudo-reward captures how information propagates.

**Correlation matrix**: Define C_{ij} = Corr(r_i, r_j). When C is not the identity matrix, we have correlated arms. The further C is from I, the more information propagates across arms.

**For notification scheduling**: The correlation structure is:
- C((Mon 9AM), (Mon 10AM)) is high (adjacent hours)
- C((Mon 9AM), (Tue 9AM)) is moderate (same hour, next day)
- C((Mon 9AM), (Fri 3AM)) is low (dissimilar slot)
- We encode this through the kernel: C_{ij} ~ k(x_i, x_j) / sqrt(k(x_i, x_i) * k(x_j, x_j))

### 10.3 The C-UCB Algorithm (Correlated UCB)

**Algorithm** (Gupta et al., 2021):

```
Initialize: For each arm i, set UCB_i = infinity
For t = 1, 2, ...:
    Pull arm i_t = argmax_i UCB_i
    Observe reward r_t

    For each arm j:
        Update UCB_j using pseudo-reward tilde{mu}_j(i_t, r_t)
```

**Key result**: Arms that are "non-competitive" (provably suboptimal given the correlation structure) are pulled only O(1) times, compared to O(log T) for standard UCB. This is because a single observation of a competitive arm can rule out a non-competitive arm through correlation.

**Regret improvement**: If there are M << K non-competitive arms, C-UCB saves O(M * log T) regret compared to standard UCB.

**For notification scheduling**: If Monday-9AM is optimal, then C-UCB can rule out Monday-3AM after very few observations (strong negative correlation with optimal). It focuses exploration on truly ambiguous arms like Monday-8AM and Monday-10AM.

### 10.4 Gaussian Correlated Bandits

When the joint reward distribution is multivariate Gaussian:

```
[r_1, ..., r_K]^T ~ N(mu, Sigma)
```

the posterior after observing a subset of arms is exactly a conditional Gaussian (same as GP regression). After observing arms in set S with values r_S:

```
mu_j | r_S = mu_j + Sigma_{j,S} Sigma_{S,S}^{-1} (r_S - mu_S)
sigma_j^2 | r_S = Sigma_{jj} - Sigma_{j,S} Sigma_{S,S}^{-1} Sigma_{S,j}
```

This is identical to GP regression with a finite domain. The Gaussian correlated bandit is exactly the GP-UCB setting restricted to finite arms.

### 10.5 Beyond Gaussian: Copula-Based Approaches

For non-Gaussian correlations (e.g., binary rewards), copulas can model the dependence structure:

```
P(r_1 <= x_1, ..., r_K <= x_K) = C(F_1(x_1), ..., F_K(x_K))
```

where F_i are marginal CDFs and C is the copula function. The Gaussian copula uses the multivariate Gaussian CDF:

```
C(u_1, ..., u_K) = Phi_K(Phi^{-1}(u_1), ..., Phi^{-1}(u_K); Sigma)
```

For binary rewards (notification: responded or not), this allows modeling correlation even though individual rewards are Bernoulli.

### References for Section 10
- Gupta, Granmo, Agrawala (2021), "Multi-Armed Bandits with Correlated Arms" (IEEE Trans. Info. Theory, https://arxiv.org/abs/1911.03959)
- Pandey, Agarwal, Chakrabarti, Josifovski (2007), "Bandits for Taxonomies"
- Mersereau, Rusmevichientong, Tsitsiklis (2009), "A Structured Multiarmed Bandit Problem and the Greedy Policy"

---

## 11. Part VIII: Thompson Sampling with Correlated Priors

### 11.1 Standard Thompson Sampling (Review)

**For independent arms**: Each arm i has unknown mean mu_i. Maintain independent Beta or Gaussian priors:

```
mu_i ~ Beta(alpha_i, beta_i)  [for Bernoulli rewards]
mu_i ~ N(m_i, s_i^2)          [for Gaussian rewards]
```

**Algorithm**:
```
For t = 1, 2, ...:
    For each arm i, sample theta_i ~ P(mu_i | history)
    Pull arm i_t = argmax_i theta_i
    Observe reward, update P(mu_{i_t} | history)
```

When arms are independent, updating arm i_t does not change the posterior of any other arm.

### 11.2 Correlated Thompson Sampling with Gaussian Priors

**Setup**: Instead of independent priors, place a **joint Gaussian prior** over all arm means:

```
[mu_1, ..., mu_K]^T ~ N(m_0, Sigma_0)
```

where Sigma_0 encodes prior correlations (derived from the kernel).

**Algorithm** (Correlated Thompson Sampling):

```
Initialize: m = m_0, Sigma = Sigma_0

For t = 1, 2, ...:
    1. Sample theta ~ N(m, Sigma)
    2. Pull arm i_t = argmax_i theta_i
    3. Observe reward y_t = mu_{i_t} + epsilon_t, epsilon_t ~ N(0, sigma_n^2)
    4. Update posterior:
       e_{i_t} = unit vector with 1 at position i_t

       Kalman gain:
       k_t = Sigma e_{i_t} / (e_{i_t}^T Sigma e_{i_t} + sigma_n^2)

       Mean update:
       m <- m + k_t * (y_t - e_{i_t}^T m)

       Covariance update:
       Sigma <- Sigma - k_t * e_{i_t}^T Sigma
```

**Derivation of the update**: This is the standard Bayesian update for a multivariate Gaussian. Observing arm i_t with reward y_t is equivalent to conditioning on y_t = e_{i_t}^T mu + epsilon_t where mu is the vector of arm means.

Using the conditional Gaussian formula:
```
mu | y_t ~ N(m + Sigma e_{i_t} (e_{i_t}^T Sigma e_{i_t} + sigma_n^2)^{-1} (y_t - e_{i_t}^T m),
            Sigma - Sigma e_{i_t} (e_{i_t}^T Sigma e_{i_t} + sigma_n^2)^{-1} e_{i_t}^T Sigma)
```

This simplifies to the Kalman filter update above.

**The magic of correlated priors**: When we observe arm i_t:
- The mean of EVERY arm j is updated: m_j <- m_j + (Sigma_{j, i_t} / (Sigma_{i_t, i_t} + sigma_n^2)) * (y_t - m_{i_t})
- The update magnitude for arm j is proportional to Sigma_{j, i_t}, the prior covariance between arms j and i_t
- Arms correlated with i_t get larger updates
- The covariance between ALL pairs of arms is updated (reduced)

### 11.3 Setting Up the Prior Covariance

For our 168-arm notification problem, the prior covariance matrix Sigma_0 is 168 x 168. We set:

```
(Sigma_0)_{ij} = k(x_i, x_j)
```

where k is our composite periodic kernel (defined in Section 14) and x_i = (day_i, hour_i) is the (day, hour) pair for arm i.

**Example** with a simple product kernel:

```
k((d, h), (d', h')) = sigma_f^2 * exp(-2*sin^2(pi*|h-h'|/24) / l_h^2)
                                * exp(-2*sin^2(pi*|d-d'|/7) / l_d^2)
```

This gives:
- (Sigma_0)[(Mon,9AM), (Mon,10AM)] ~ sigma_f^2 * 0.95 * 1.0 = 0.95 * sigma_f^2 (very correlated)
- (Sigma_0)[(Mon,9AM), (Tue,9AM)] ~ sigma_f^2 * 1.0 * 0.90 = 0.90 * sigma_f^2 (correlated)
- (Sigma_0)[(Mon,9AM), (Fri,3AM)] ~ sigma_f^2 * 0.05 * 0.20 = 0.01 * sigma_f^2 (almost independent)

### 11.4 GP Thompson Sampling

**Full GP-TS**: Instead of maintaining a fixed-dimensional Gaussian posterior, maintain a full GP posterior and sample entire functions from it.

```
For t = 1, 2, ...:
    1. Sample f_t ~ GP(mu_{t-1}, k_{t-1})  [sample function from GP posterior]
    2. Pull x_t = argmax_x f_t(x)
    3. Observe y_t = f(x_t) + epsilon_t
    4. Update GP posterior: (mu_t, k_t)
```

**Sampling from a GP posterior** (at the 168 grid points):
1. Compute posterior mean vector mu in R^{168}
2. Compute posterior covariance matrix Sigma in R^{168 x 168}
3. Compute Cholesky: Sigma = L L^T
4. Sample z ~ N(0, I_{168})
5. Return f = mu + L z

This is computationally cheap for 168 points: Cholesky is O(168^3) ~ 4.7 million operations, negligible on modern hardware.

**Regret bound for GP-TS** (Chowdhury & Gopalan, 2017):

```
R_T = O(sqrt(T * gamma_T * log T))
```

Same order as GP-UCB. Thompson Sampling has the advantage of being naturally randomized, which can help in adversarial settings.

### 11.5 Correlated Thompson Sampling with Beta Priors (for Binary Rewards)

Since notification responses are binary (responded / didn't respond), Beta priors are more natural.

**Challenge**: Beta distributions don't support multivariate correlation directly.

**Solution 1: Probit approximation**: Map Beta to Gaussian, add correlations in the Gaussian space, map back.

**Solution 2: Direct kernel-weighted pseudo-updates** (practical approach):

```
For each arm i, maintain Beta(alpha_i, beta_i).

When arm i_t is pulled and reward y_t is observed:
  For each arm j:
    weight_j = k(x_j, x_{i_t})  / k(x_{i_t}, x_{i_t})   [kernel similarity]
    pseudo_count = weight_j * effective_sample_size

    if y_t = 1:
      alpha_j += pseudo_count
    else:
      beta_j += pseudo_count
```

This is a heuristic but effective approach. The kernel weight determines how much of the observation to "share" with each arm. The `effective_sample_size` parameter (typically 0.5-2.0) controls the propagation strength.

**Solution 3: Gaussian approximation to Beta**: When alpha_i and beta_i are both large enough (>5), Beta(alpha, beta) is well-approximated by N(alpha/(alpha+beta), alpha*beta/((alpha+beta)^2*(alpha+beta+1))). Use the correlated Gaussian Thompson Sampling above with this approximation.

### 11.6 Practical Algorithm for Notification Scheduling

Combining the above ideas, here is a complete practical algorithm:

```
INITIALIZE:
  For each slot (d, h):
    alpha[d][h] = 1  (prior success count)
    beta[d][h] = 1   (prior failure count)
  Compute kernel matrix K[168 x 168] with periodic kernel

EACH ROUND:
  1. For each slot (d, h):
       Sample theta[d][h] ~ Beta(alpha[d][h], beta[d][h])
  2. Select slot (d*, h*) = argmax theta[d][h]
     (restrict to future slots if scheduling constraints exist)
  3. Send notification at (d*, h*)
  4. Observe reward y in {0, 1}
  5. For each slot (d, h):
       w = K[(d,h), (d*,h*)] / K[(d*,h*), (d*,h*)]
       effective = w * propagation_strength
       if y = 1:
         alpha[d][h] += effective
       else:
         beta[d][h] += effective
```

### References for Section 11
- Thompson (1933), "On the Likelihood that One Unknown Probability Exceeds Another"
- Russo & Van Roy (2014), "Learning to Optimize via Posterior Sampling" (Mathematics of Operations Research)
- Russo, Van Roy, Kazerouni, Osband, Wen (2018), "A Tutorial on Thompson Sampling" (https://web.stanford.edu/~bvr/pubs/TS_Tutorial.pdf)
- Chowdhury & Gopalan (2017), "On Kernelized Multi-armed Bandits"
- Kandasamy, Schneider, Poczos (2018), "Parallelised Bayesian Optimisation via Thompson Sampling"

---

## 12. Part IX: Periodic Kernels for Time Structure

### 12.1 The Need for Periodicity

User engagement has inherent cyclical structure:
- **Daily cycle (period = 24 hours)**: People sleep at night, work during the day
- **Weekly cycle (period = 7 days)**: Weekday vs. weekend patterns differ
- **Interactions**: "Monday 9AM" is different from "Saturday 9AM" despite same hour

Standard stationary kernels (RBF, Matern) don't capture periodicity -- they only see raw distance. Hour 23 and hour 0 are 23 apart in raw distance but only 1 apart in reality.

### 12.2 The Standard Periodic Kernel

**Exponential Sine Squared (ExpSineSquared)**:

```
k_periodic(x, x') = sigma_f^2 * exp(-2 * sin^2(pi * |x - x'| / p) / l^2)
```

**Derivation intuition**: The sin^2 term creates a distance metric that wraps around:
- When |x - x'| = 0: sin^2(0) = 0, k = sigma_f^2 (maximum similarity)
- When |x - x'| = p/2: sin^2(pi/2) = 1, k = sigma_f^2 * exp(-2/l^2) (minimum similarity)
- When |x - x'| = p: sin^2(pi) = 0, k = sigma_f^2 (back to maximum -- periodic!)

**Construction via feature map**: The periodic kernel can be derived by mapping x to a circle:

```
phi(x) = [cos(2*pi*x/p), sin(2*pi*x/p)]
```

and applying an RBF kernel in this 2D feature space. The distance on the circle naturally wraps around.

More precisely, the squared distance in the feature space is:

```
||phi(x) - phi(x')||^2 = 2 - 2*cos(2*pi*(x-x')/p) = 4*sin^2(pi*(x-x')/p)
```

Substituting into the RBF kernel:

```
k(x, x') = exp(-||phi(x) - phi(x')||^2 / (2*l^2))
          = exp(-4*sin^2(pi*(x-x')/p) / (2*l^2))
          = exp(-2*sin^2(pi*(x-x')/p) / l^2)
```

This is exactly the periodic kernel.

### 12.3 Locally Periodic Kernel

Pure periodic kernels assume exact repetition forever. In practice, patterns drift. The **locally periodic kernel** modulates periodicity with overall decay:

```
k_locally_periodic(x, x') = k_periodic(x, x') * k_RBF(x, x')
                           = sigma_f^2 * exp(-2*sin^2(pi*|x-x'|/p) / l_p^2) * exp(-|x-x'|^2 / (2*l_se^2))
```

**Effect**: Nearby periods are highly correlated, distant periods gradually decorrelate. For our problem, this means "this Monday's 9AM" correlates with "next Monday's 9AM" but less with "Monday 9AM three months from now."

### 12.4 Composite Kernel for the (Day, Hour) Grid

We need to capture:
1. Within-day periodicity (hour cycle, period 24)
2. Within-week periodicity (day cycle, period 7)
3. Interaction between day and hour

**Approach A: Product of periodic kernels**

```
k((d,h), (d',h')) = k_hour(h, h') * k_day(d, d')
```

where:
```
k_hour(h, h') = sigma_h^2 * exp(-2*sin^2(pi*|h-h'|/24) / l_h^2)
k_day(d, d')  = sigma_d^2 * exp(-2*sin^2(pi*|d-d'|/7) / l_d^2)
```

**Properties**: High correlation requires BOTH hour AND day to be similar. Captures "Monday 9AM is like Tuesday 9AM" and "Monday 9AM is like Monday 10AM."

**Approach B: Sum of periodic kernels (additive model)**

```
k((d,h), (d',h')) = k_hour(h, h') + k_day(d, d')
```

**Properties**: Captures independent "hour effect" and "day effect." Monday 9AM is similar to any-day 9AM (hour match) and also similar to Monday-any-hour (day match). Allows the GP to decompose the reward into f(d,h) = f_hour(h) + f_day(d).

**Approach C: Sum-of-products with interaction**

```
k((d,h), (d',h')) = k_hour(h, h') + k_day(d, d') + k_hour(h, h') * k_day(d, d')
```

This captures both main effects and their interaction.

**Approach D: Multiple harmonics**

```
k_hour(h, h') = sigma_1^2 * exp(-2*sin^2(pi*|h-h'|/24) / l_1^2)
              + sigma_2^2 * exp(-2*sin^2(pi*|h-h'|/12) / l_2^2)
              + sigma_3^2 * exp(-2*sin^2(pi*|h-h'|/8) / l_3^2)
```

Multiple harmonics capture non-sinusoidal patterns: "sharp peak at 9AM, gradual decline through afternoon, sharp drop at 11PM." The 24-hour fundamental captures the basic day/night cycle, the 12-hour harmonic captures AM/PM differences, the 8-hour harmonic captures finer structure.

### 12.5 Kernel Matrix Visualization

For the product-of-periodic kernel on our 168-arm grid, the 168x168 kernel matrix has a beautiful block structure:

```
Ordering arms as: (Mon,0), (Mon,1), ..., (Mon,23), (Tue,0), ..., (Sun,23)

The matrix looks like:
[B_11  B_12  ...  B_17]
[B_21  B_22  ...  B_27]
[...   ...   ...  ... ]
[B_71  B_72  ...  B_77]

where each B_{ij} is a 24x24 block:
(B_{ij})_{hh'} = k_hour(h, h') * k_day(i-1, j-1)
```

The diagonal blocks B_{ii} are all identical (same-day correlations). The off-diagonal blocks B_{ij} are scaled by k_day(i-1, j-1) (day-to-day correlation factor).

### 12.6 Weekday/Weekend Structure

An important refinement: engagement patterns differ more between weekdays and weekends than between adjacent weekdays. We can encode this with a custom day kernel:

```
k_day(d, d') = sigma_w^2 * [weekday(d) == weekday(d')] + sigma_d^2 * exp(-2*sin^2(pi*|d-d'|/7) / l_d^2)
```

where the first term is a delta kernel that gives extra correlation between weekdays (Mon-Fri) and separately between weekend days (Sat-Sun), and the second term is the standard periodic day kernel.

Alternatively, use an ARD (Automatic Relevance Determination) kernel with a 7-dimensional one-hot encoding of the day, letting the GP learn which days are similar.

### References for Section 12
- MacKay (1998), "Introduction to Gaussian Processes" (periodic kernels section)
- Duvenaud (2014), "Automatic Model Construction with Gaussian Processes" (Chapter 2, Kernel Cookbook)
- Roberts, Osborne, Ebden, Sherlock, and Sherlock (2013), "Gaussian Processes for Time-Series Modelling"
- Rasmussen & Williams (2006), "Gaussian Processes for Machine Learning", Section 4.2.3

---

## 13. Part X: Toroidal Distance Metrics

### 13.1 The Problem with Euclidean Distance

On our (day, hour) grid, Euclidean distance gives:
- d((Mon, 0AM), (Mon, 23PM)) = 23 hours
- d((Sunday, 12PM), (Monday, 12PM)) = 6 days

But in reality:
- Monday 0AM and Monday 11PM are 1 hour apart (wrapping around midnight)
- Sunday 12PM and Monday 12PM are 1 day apart (wrapping around the week)

We need a **wrap-around** (toroidal) distance.

### 13.2 Circular (Wrap-Around) Distance

For a single cyclical dimension with period P:

```
d_circular(a, b; P) = min(|a - b|, P - |a - b|)
```

Equivalently:

```
d_circular(a, b; P) = P/2 - |P/2 - |a - b||
```

Or using the modular arithmetic approach:

```
d_circular(a, b; P) = min((a - b) mod P, (b - a) mod P)
```

**For hours** (P = 24):
- d_circular(23, 0; 24) = min(23, 1) = 1  (correct!)
- d_circular(9, 15; 24) = min(6, 18) = 6

**For days** (P = 7):
- d_circular(6, 0; 7) = min(6, 1) = 1  (Sunday to Monday = 1 day, correct!)
- d_circular(0, 3; 7) = min(3, 4) = 3  (Monday to Thursday = 3 days)

### 13.3 Toroidal Distance (2D Wrap-Around)

Our grid wraps in BOTH dimensions simultaneously, forming a **torus**. The toroidal distance is:

```
d_torus((d1, h1), (d2, h2)) = sqrt(d_circular(d1, d2; 7)^2 + d_circular(h1, h2; 24)^2)
```

But since the scales of hours and days are different, we should normalize:

```
d_torus_normalized((d1, h1), (d2, h2)) = sqrt((d_circular(d1, d2; 7) / s_d)^2 + (d_circular(h1, h2; 24) / s_h)^2)
```

where s_d and s_h are scale factors (or equivalently, use different bandwidths/length-scales per dimension in the kernel).

### 13.4 Toroidal Geometry

**Visualization**: A torus is a donut shape. Imagine:
1. Take our 7 x 24 grid (a rectangle)
2. Roll it into a cylinder by connecting the left and right edges (hour wraps: hour 23 connects to hour 0)
3. Bend the cylinder into a donut by connecting the top and bottom (day wraps: Sunday connects to Monday)

Every point on the torus has neighbors in all directions with no boundary effects.

**Formal topology**: The torus T^2 = S^1 x S^1 is the product of two circles. Our domain is the discrete torus Z_7 x Z_24, the product of cyclic groups.

### 13.5 Using Toroidal Distance in Kernels

**Option A: Periodic kernel (recommended)**

The periodic kernel inherently handles wrap-around through the sin^2 term:

```
sin^2(pi * d_circular(h, h'; 24) / 24) = sin^2(pi * |h - h'| / 24)
```

This is the same whether we compute |h - h'| directly or use the circular distance, because sin^2(pi * x / 24) = sin^2(pi * (24 - x) / 24) by the identity sin(pi - theta) = sin(theta).

So periodic kernels **automatically handle** the toroidal structure! No explicit toroidal distance needed.

**Option B: RBF with toroidal distance**

If using a non-periodic kernel like RBF, substitute the toroidal distance:

```
k_RBF_torus((d,h), (d',h')) = sigma_f^2 * exp(-d_torus^2 / (2 * l^2))
```

Note: This is a valid kernel because the circular distance is a proper metric on the circle, and the RBF kernel applied to any proper metric yields a valid kernel.

**Option C: Matern with toroidal distance**

Similarly for Matern kernels. However, care is needed: the Matern kernel is only guaranteed to be PSD for the Euclidean metric. For circular metrics, one should verify PSD-ness or use the periodic kernel construction instead.

### 13.6 Implementation

```python
def circular_distance(a, b, period):
    """Compute wrap-around distance between a and b with given period."""
    diff = abs(a - b)
    return min(diff, period - diff)

def toroidal_distance(d1, h1, d2, h2, scale_d=1.0, scale_h=1.0):
    """Compute distance on the (day, hour) torus."""
    dd = circular_distance(d1, d2, 7) / scale_d
    dh = circular_distance(h1, h2, 24) / scale_h
    return math.sqrt(dd**2 + dh**2)

def periodic_kernel_2d(d1, h1, d2, h2, sigma_f=1.0, l_h=3.0, l_d=1.5):
    """Product of periodic kernels for day and hour."""
    k_h = math.exp(-2 * math.sin(math.pi * abs(h1 - h2) / 24)**2 / l_h**2)
    k_d = math.exp(-2 * math.sin(math.pi * abs(d1 - d2) / 7)**2 / l_d**2)
    return sigma_f**2 * k_h * k_d
```

### References for Section 13
- Demofox Blog (2017), "Calculating the Distance Between Points in Wrap Around (Toroidal) Space" (https://blog.demofox.org/2017/10/01/calculating-the-distance-between-points-in-wrap-around-toroidal-space/)
- Papadimitriou, Raghavan, Tamaki, Vempala (2003), "Latent Semantic Indexing: A Probabilistic Analysis" (discusses toroidal metrics)
- Any molecular dynamics textbook (periodic boundary conditions / minimum image convention)

---

## 14. Part XI: Putting It All Together -- The Notification Scheduling Kernel

### 14.1 The Full Composite Kernel

For the 168-arm notification bandit, we propose:

```
k((d,h), (d',h')) = k_hour(h, h') * k_day(d, d') + k_hour_only(h, h') + k_day_only(d, d') + sigma_n^2 * delta((d,h), (d',h'))
```

where:

**Hour component** (periodic, with harmonics):
```
k_hour(h, h') = sigma_h1^2 * exp(-2*sin^2(pi*|h-h'|/24) / l_h1^2)
              + sigma_h2^2 * exp(-2*sin^2(pi*|h-h'|/12) / l_h2^2)
```

**Day component** (periodic + weekday/weekend):
```
k_day(d, d') = sigma_d^2 * exp(-2*sin^2(pi*|d-d'|/7) / l_d^2)
             + sigma_wd^2 * 1[same_type(d, d')]
```
where same_type returns 1 if both are weekdays or both are weekend days.

**Independent hour effect**: k_hour_only(h, h') captures "9AM is always good" regardless of day.

**Independent day effect**: k_day_only(d, d') captures "Monday is always good" regardless of hour.

**Noise**: sigma_n^2 * delta is the observation noise (diagonal term).

### 14.2 Hyperparameter Initialization

Reasonable starting values for notification scheduling:

| Parameter | Value | Interpretation |
|-----------|-------|----------------|
| sigma_h1 | 0.5 | Amplitude of 24-hour cycle |
| l_h1 | 3.0 | ~3 hour correlation length |
| sigma_h2 | 0.3 | Amplitude of 12-hour harmonic |
| l_h2 | 2.0 | Correlation length for AM/PM pattern |
| sigma_d | 0.3 | Amplitude of weekly cycle |
| l_d | 1.5 | ~1.5 day correlation length |
| sigma_wd | 0.2 | Weekday/weekend similarity boost |
| sigma_n | 0.3 | Observation noise |

These can be optimized using marginal likelihood (Section 5.4) as data accumulates.

### 14.3 The 168 x 168 Kernel Matrix

**Construction**: For arms indexed as i = 24*d + h for d in {0,...,6}, h in {0,...,23}:

```python
import numpy as np

def build_kernel_matrix(sigma_h1=0.5, l_h1=3.0, sigma_h2=0.3, l_h2=2.0,
                        sigma_d=0.3, l_d=1.5, sigma_wd=0.2, sigma_n=0.3):
    K = np.zeros((168, 168))
    for i in range(168):
        d_i, h_i = divmod(i, 24)
        for j in range(168):
            d_j, h_j = divmod(j, 24)

            # Hour kernels
            k_h1 = sigma_h1**2 * np.exp(-2 * np.sin(np.pi * abs(h_i - h_j) / 24)**2 / l_h1**2)
            k_h2 = sigma_h2**2 * np.exp(-2 * np.sin(np.pi * abs(h_i - h_j) / 12)**2 / l_h2**2)
            k_hour = k_h1 + k_h2

            # Day kernel
            k_d = sigma_d**2 * np.exp(-2 * np.sin(np.pi * abs(d_i - d_j) / 7)**2 / l_d**2)
            same_type = float((d_i < 5) == (d_j < 5))  # both weekday or both weekend
            k_day = k_d + sigma_wd**2 * same_type

            # Composite: product + additive terms
            K[i, j] = k_hour * k_day + k_hour + k_day

            # Noise on diagonal
            if i == j:
                K[i, j] += sigma_n**2

    return K
```

This matrix is 168 x 168 = 28,224 entries. It fits in memory trivially and can be inverted in microseconds.

### 14.4 Complete GP-UCB System

```python
class NotificationGPUCB:
    def __init__(self, kernel_params):
        self.n_arms = 168
        self.K_prior = build_kernel_matrix(**kernel_params)
        self.observations = []  # list of (arm_index, reward)
        self.sigma_n = kernel_params.get('sigma_n', 0.3)

    def get_posterior(self):
        """Compute GP posterior mean and variance for all 168 arms."""
        if not self.observations:
            return np.zeros(self.n_arms), np.diag(self.K_prior)

        obs_idx = [o[0] for o in self.observations]
        obs_y = np.array([o[1] for o in self.observations])

        # K_train: kernel between observed arms
        K_train = self.K_prior[np.ix_(obs_idx, obs_idx)] + self.sigma_n**2 * np.eye(len(obs_idx))

        # K_cross: kernel between all arms and observed arms
        K_cross = self.K_prior[:, obs_idx]

        # Cholesky solve
        L = np.linalg.cholesky(K_train)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, obs_y))
        V = np.linalg.solve(L, K_cross.T)

        mu = K_cross @ alpha
        var = np.diag(self.K_prior) - np.sum(V**2, axis=0)
        var = np.maximum(var, 1e-10)  # numerical stability

        return mu, var

    def select_arm(self, beta=2.0):
        """Select arm with highest UCB."""
        mu, var = self.get_posterior()
        ucb = mu + np.sqrt(beta) * np.sqrt(var)
        return np.argmax(ucb)

    def update(self, arm_index, reward):
        """Record observation."""
        self.observations.append((arm_index, reward))

    def arm_to_day_hour(self, arm_index):
        """Convert linear index to (day, hour)."""
        return divmod(arm_index, 24)
```

### 14.5 Complete Thompson Sampling System

```python
class NotificationTS:
    def __init__(self, kernel_params):
        self.n_arms = 168
        self.K_prior = build_kernel_matrix(**kernel_params)
        self.sigma_n = kernel_params.get('sigma_n', 0.3)
        self.mu = np.zeros(self.n_arms)
        self.Sigma = self.K_prior.copy()

    def select_arm(self):
        """Sample from posterior and select best arm."""
        L = np.linalg.cholesky(self.Sigma + 1e-6 * np.eye(self.n_arms))
        sample = self.mu + L @ np.random.randn(self.n_arms)
        return np.argmax(sample)

    def update(self, arm_index, reward):
        """Bayesian update (Kalman filter style)."""
        # Kalman gain
        s = self.Sigma[:, arm_index]  # column of Sigma
        s_ii = self.Sigma[arm_index, arm_index] + self.sigma_n**2
        k = s / s_ii

        # Update mean
        innovation = reward - self.mu[arm_index]
        self.mu = self.mu + k * innovation

        # Update covariance (rank-1 downdate)
        self.Sigma = self.Sigma - np.outer(k, s)

        # Ensure symmetry and PSD
        self.Sigma = (self.Sigma + self.Sigma.T) / 2
```

---

## 15. Part XII: Intermediate Approaches (Simpler than Full GP)

Here is a progression from simplest to most sophisticated, with a description of each level.

### Level 0: Fully Independent Arms (Current System)

**What it is**: The current `bandit.py`. Each arm maintains its own Q-value. No information sharing.

**Complexity**: O(1) per update, O(168) per recommendation.

**Pros**: Simple, no assumptions about correlation.
**Cons**: Slow convergence, wastes observations.

### Level 1: Nadaraya-Watson Kernel Smoothing

**What it is**: Section 9 above. Smooth Q-values using a kernel over the toroidal grid.

**Complexity**: O(168) per update or readout.

**What to implement**: Add the `get_smoothed_q()` function from Section 9.6 to `bandit.py`. Use periodic distance.

**Pros**: Very simple. Immediate reward propagation. No matrix operations.
**Cons**: No uncertainty quantification. Can't do principled exploration.

### Level 2: Exponential Smoothing with Neighbor Propagation

**What it is**: When an arm is updated, also update its k nearest neighbors with geometrically decaying weights.

```python
def update_with_neighbors(day, hour, reward, decay_factor=0.5, max_hops=3):
    """Update arm and propagate to neighbors."""
    update_arm(day, hour, reward)
    for hop in range(1, max_hops + 1):
        weight = decay_factor ** hop
        if weight < 0.01:
            break
        # Hour neighbors (wrapping)
        for dh in [-hop, hop]:
            neighbor_hour = (hour + dh) % 24
            update_arm(day, neighbor_hour, reward * weight)
        # Day neighbors (wrapping)
        for dd in [-hop, hop]:
            neighbor_day = (day + dd) % 7
            update_arm(neighbor_day, hour, reward * weight)
```

**Complexity**: O(max_hops) per update.

**Pros**: Trivial to implement. Direct reward propagation.
**Cons**: No principled bandwidth selection. Only propagates along axes (not diagonally).

### Level 3: Correlated Beta Thompson Sampling

**What it is**: Section 11.5. Maintain Beta(alpha, beta) for each arm. Propagate pseudo-counts using kernel weights.

**Complexity**: O(168) per update (to propagate to all arms).

**Pros**: Principled exploration (Thompson Sampling). Reward propagation. Handles binary rewards naturally.
**Cons**: Heuristic propagation (not exact Bayesian). Bandwidth/propagation_strength must be tuned.

### Level 4: Gaussian Thompson Sampling with Correlated Prior

**What it is**: Section 11.2. Maintain a 168-dimensional Gaussian posterior. Full Kalman-filter updates.

**Complexity**: O(168^2) per update (rank-1 covariance update). O(168^3) for Cholesky at selection time.

**Pros**: Exact Bayesian inference (under Gaussian model). Full correlation exploitation. Principled exploration.
**Cons**: Gaussian model for binary rewards (approximation). 168x168 matrix operations (still fast).

### Level 5: Full GP-UCB / GP-TS

**What it is**: Sections 6, 11.4, 14.4-14.5. Full GP with kernel hyperparameter optimization.

**Complexity**: O(n^3) where n is number of observations (not 168; can grow larger). O(168^2) for prediction.

**Pros**: Most principled. Adapts hyperparameters. Best theoretical guarantees.
**Cons**: Most complex. Requires GP library (GPyTorch, GPflow). Hyperparameter optimization adds computational cost.

### Recommendation for ProactiveClaw

**Start with Level 2 or Level 3**. Here is why:

1. Level 2 (neighbor propagation) can be added to the existing `bandit.py` in ~20 lines of code. It gives immediate improvement.

2. Level 3 (correlated Beta TS) is the best balance of sophistication and simplicity. It gives:
   - Natural exploration (Thompson Sampling)
   - Natural handling of binary rewards (Beta distribution)
   - Reward propagation through kernel weights
   - Modest implementation effort (~100 lines)

3. Level 5 (full GP) is only needed if you want:
   - Automatic hyperparameter learning
   - Rigorous theoretical guarantees
   - Non-stationary modeling (evolving user patterns)
   - Integration with a rich feature set (contextual bandits)

---

## 16. Complete Reference List

### Foundational Texts
1. Rasmussen & Williams (2006), "Gaussian Processes for Machine Learning" -- free at http://gaussianprocess.org/gpml/
2. Lattimore & Szepesvari (2020), "Bandit Algorithms" -- free at https://tor-lattimore.com/downloads/book/book.pdf
3. Scholkopf & Smola (2002), "Learning with Kernels"
4. Bishop (2006), "Pattern Recognition and Machine Learning", Chapter 6 (Kernels) and Chapter 3.3 (Bayesian Linear Regression)

### Key Papers -- GP Bandits
5. Srinivas, Krause, Kakade, Seeger (2010), "Gaussian Process Optimization in the Bandit Setting: No Regret and Experimental Design", ICML -- https://arxiv.org/abs/0912.3995
6. Srinivas, Krause, Kakade, Seeger (2012), "Information-Theoretic Regret Bounds for Gaussian Process Optimization in the Bandit Setting", IEEE Trans. Info. Theory -- https://ieeexplore.ieee.org/document/6138914/
7. Chowdhury & Gopalan (2017), "On Kernelized Multi-armed Bandits", ICML
8. Desautels, Krause, Burdick (2014), "Parallelizing Exploration-Exploitation Tradeoffs in Gaussian Process Bandit Optimization", JMLR -- https://jmlr.org/papers/volume15/desautels14a/desautels14a.pdf

### Key Papers -- Correlated Bandits
9. Gupta, Granmo, Agrawala (2021), "Multi-Armed Bandits with Correlated Arms", IEEE Trans. Info. Theory -- https://arxiv.org/abs/1911.03959
10. Pandey, Agarwal, Chakrabarti, Josifovski (2007), "Bandits for Taxonomies: A Model-based Approach", SDM

### Key Papers -- Thompson Sampling
11. Russo, Van Roy, Kazerouni, Osband, Wen (2018), "A Tutorial on Thompson Sampling", Foundations and Trends in ML -- https://web.stanford.edu/~bvr/pubs/TS_Tutorial.pdf
12. Kandasamy, Schneider, Poczos (2018), "Parallelised Bayesian Optimisation via Thompson Sampling"

### Key Papers -- Contextual Bandits
13. Li, Chu, Langford, Schapire (2010), "A Contextual-Bandit Approach to Personalized News Article Recommendation" -- https://arxiv.org/abs/1003.0146
14. Valko, Korda, Munos, Flaounas, Cristianini (2013), "Finite-Time Analysis of Kernelised Contextual Bandits" -- https://arxiv.org/pdf/1309.6869
15. Chu, Li, Reyzin, Schapire (2011), "Contextual Bandits with Linear Payoff Functions"

### Key Papers -- Lipschitz Bandits
16. Kleinberg, Slivkins, Upfal (2019), "Bandits and Experts in Metric Spaces", JACM
17. Bubeck, Munos, Stoltz, Szepesvari (2011), "X-Armed Bandits", JMLR
18. Janz, Burt, Gonzalez (2020), "Bandit Optimisation of Functions in the Matern Kernel RKHS", AISTATS -- https://arxiv.org/abs/2001.10396

### Key Papers -- Kernel Methods and Periodic Kernels
19. Duvenaud (2014), "Automatic Model Construction with Gaussian Processes", PhD thesis -- https://www.cs.toronto.edu/~duvenaud/cookbook/
20. MacKay (1998), "Introduction to Gaussian Processes"
21. Roberts, Osborne, Ebden, Sherlock, and Sherlock (2013), "Gaussian Processes for Time-Series Modelling"

### Kernel Smoothing
22. Nadaraya (1964), "On Estimating Regression", Theory of Probability and Its Applications
23. Watson (1964), "Smooth Regression Analysis", Sankhya
24. Wasserman (2006), "All of Nonparametric Statistics", Springer

### Online Resources
25. Distill.pub (2019), "A Visual Exploration of Gaussian Processes" -- https://distill.pub/2019/visual-exploration-gaussian-processes/
26. Duvenaud, "The Kernel Cookbook" -- https://www.cs.toronto.edu/~duvenaud/cookbook/
27. Roelants (2020), "Gaussian processes: exploring kernels" -- https://peterroelants.github.io/posts/gaussian-process-kernels/

---

## Appendix A: Notation Reference

| Symbol | Meaning |
|--------|---------|
| x, x' | Input points (e.g., (day, hour) pairs) |
| f(x) | True (unknown) reward function |
| y | Noisy observation: y = f(x) + epsilon |
| k(x, x') | Kernel / covariance function |
| K | Gram matrix: K_{ij} = k(x_i, x_j) |
| mu(x) | GP posterior mean at x |
| sigma^2(x) | GP posterior variance at x |
| sigma_n^2 | Observation noise variance |
| sigma_f^2 | Signal variance (kernel amplitude) |
| l | Length-scale parameter |
| p | Period parameter |
| nu | Matern smoothness parameter |
| beta_t | UCB exploration parameter at time t |
| gamma_T | Maximum information gain after T rounds |
| R_T | Cumulative regret after T rounds |
| H_k | RKHS associated with kernel k |
| N(mu, Sigma) | Multivariate Gaussian distribution |
| GP(m, k) | Gaussian Process with mean m, covariance k |
| Beta(alpha, beta) | Beta distribution with parameters alpha, beta |
| d_circular(a, b; P) | Circular distance with period P |

## Appendix B: Quick-Start Implementation Guide

To add reward propagation to the current `bandit.py` with minimal changes:

**Step 1**: Add toroidal distance function (2 lines)
**Step 2**: Add kernel weight function (3 lines)
**Step 3**: Modify `update_arm` to propagate to neighbors using kernel weights (10 lines)
**Step 4**: Modify `get_recommendations` to use smoothed Q-values (5 lines)

Total: approximately 20 lines of additional code for Level 2 improvement.

For Level 3 (correlated Beta TS): replace the Q-value system entirely with Beta distributions and kernel-weighted pseudo-count propagation. This is a larger refactor (~100 lines) but gives principled exploration.

For Level 5 (full GP): use GPyTorch or scikit-learn's GaussianProcessRegressor with a custom periodic kernel. The kernel composition from Section 14 can be built using scikit-learn's kernel algebra (ExpSineSquared * ExpSineSquared + ExpSineSquared + ExpSineSquared).
