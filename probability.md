```
\text {Cost}(z_1, ... z_ K) = \sum _{i=1}^{n} \min _{j=1,...,K} \left\|  x^{(i)} - z_ j \right\| ^2
```

## Probability Models and Axioms

- Sample space ${\displaystyle \Omega}$ should be: mutually exclusive, collectively exhaustive, with the right granularity
- ${\displaystyle P(A)>= 0}$
- ${\displaystyle P(\Omega)=1}$
- ${\displaystyle P(A ∩ \Omega)= {P(A)}}$
- ${\displaystyle P(A) + P(A^c) = 1}$
- if $A \subseteq B$ then $P(A) \leq P(B)$
- ${\displaystyle P(A ∪ B) = P(A) + P(B) - P(A ∩ B)}$
- ${\displaystyle P(A ∩ B^c) = P(A) - P(A ∩ B)}$
- ${\displaystyle S ∩ (T ∪ U) = (S ∩ T) ∪ (S ∩ U)}$
- ${\displaystyle S ∪ (T ∩ U) = (S ∪ T) ∩ (S ∪ U)}$
- Bonferroni Inequality: ${\displaystyle P(A_1 ∩ A_2) \geq P(A_1) + P(A_2)}$
- Interpretation: - frequencies, description of beliefs, betting preferences
- [De Morgan Laws](https://brilliant.org/wiki/de-morgans-laws/)
- Countable vs uncountable sets

## Useful Formulas

Infinite Sum Law with $a$ positive and $\leq 1$, i.e. geometric series

$${\displaystyle \sum_{k=0}^{\infty}a^{k}=\frac{1}{1-a}}$$

Starting from one:

$${\displaystyle \sum_{k=1}^{\infty}a^{k}=\frac{a}{1-a}}$$

Starting from $j$:

$${\displaystyle \sum_{k=j}^{\infty}2^{-k}=2^{-j+1}}$$

Infinite Sum Law $b$:

$${\displaystyle 0 + 1 + ... n = \frac{n(n+1)}{2}}$$

[Sequence](https://en.wikipedia.org/wiki/Geometric_series#Formula) / Sequence Convergence

## Conditioning Rules

### Multiplication Rules

$${\displaystyle P(A ∩ B) = {P(B)P(A | B)} = {P(A)P(B | A)} }$$

$${\displaystyle P(A ∩ B ∩ C) = {P(A)P(B | A)P(C | A ∩ B)}}$$

### Total Probability Theorem

i.e. probability of an event $A$ is the sum of the probabilities of that event happening under every possible scenario $B_n$ times the probability of that scenario happening (the same applies for **expectations**):

$${\displaystyle P(A)=\sum_{n}P(A\mid B_{n})\Pr(B_{n}),}$$

$${\displaystyle E[X]=\sum_{n}P(A_n)E[X\mid A_{n}]} $$

$${\displaystyle E[X]=\sum_{y}p_Y(y)E[X\mid Y = y]}$$

### Bayes Rules

$${\displaystyle P(A\mid B)={\frac {P(B\mid A)P(A)}{P(B)}}}$$

$${\displaystyle P(A\mid B)={\frac {P(B ∩ A)}{P(B)}}}$$

### Independence

$${\displaystyle P(A | B) = P(A)}$$

$${\displaystyle P(A ∩ B)=P(A)P(B)}$$

- $A$ and $B$ independent ⇒ $A$ and $B^c$ independent ⇒ $B^c$ and $A$ independent ⇒ $B^c$ and $A^c$ independent
- Independent events $≠$ Disjoint events!
- Independence **does not imply** conditional independence
- **Independence intuitive definition:** info of some event does not change probabilities of the other event.

### Total probability theorem

$${\displaystyle \Pr(A)=\sum _{n}\Pr(A\cap B_{n})}$$

$${\displaystyle \Pr(A)=\sum _{n}\Pr(A\mid B_{n})\Pr(B_{n})}$$

## Counting

- **Counting Principle**: product for $i$ that goes from $1$ to $n$
- **Permutations**: number of ways of ordering $n$ elements: $n!$
- **Number of all possbile subsets** of size ${n} \rightarrow 2^n$
- **Combinations**: number of $k$ elements subsets of a given $n$ elements set<br>
  $${\displaystyle \binom{n}{k} = {\frac {n!}{(n - k)!k!}}}$$
- **Binomial probabilities**: probability of obtaining $k$ heads in $n$ tosses is:<br>
  $$P(X=k) = \binom{n}{k}p^k(1-p)^{n-k}$$
- Partitions
- Convention: $0! = 1$

## Discrete Random Variables

### Definitions

#### Expected value

$${\displaystyle E(X) = \sum_{x}{p_{X}(x)x}}$$

<span style="color:blue">
  <em>Properties of expectation</em>
</span>

<br>

<span style="color:blue">
  <em>Expectation rule</em>
</span>

<br>

<span style="color:blue">
  <em>Total probability theorem vs Total expectation theorem</em>
</span>

#### Variance

$$\operatorname {Var} (X)=\operatorname {E} \left[(X-\mu )^{2}\right]$$

$$\operatorname {Var} (X)=\operatorname {E}[X^2] - \operatorname{E}[X]^2$$

<span style="color:blue">
  <em>Total Variance Theorem</em>
</span>

#### Covariance

Covariance is **bilinear**, i.e.

$$\textsf{Cov}(aX + bY, Z) = a\textsf{Cov}(X,Z) + b\textsf{Cov}(Y,Z)$$

or, stated in another way:

$$\textsf{Cov}(X_1 + X_2, Y ) = \textsf{Cov}(X_1, Y ) + \textsf{Cov}(X_2, Y)$$

### Distributions

#### Bernoulli Distribution

- **PMF**: Taking value $1$ with probability $p$ and taking the value $0$ with probability $1-p$

- **Expected value**: $p$

- **Variance**: $p(1-p)$

#### Uniform Distribution

- **PMF**: Integers from $a$ to $b$ (with $a \leq b$) with probability $p(x) = \frac{1}{b - a +1}$

- **Expected value**: ${\displaystyle (b-a)/2}$

- **Variance**: ${\displaystyle (b - a)(b - a +2)/12}$

#### Binomial Distribution

- **PMF**: number of successes with probability $p$ over $n$ events. It is a sequence of Bernoulli rvs

$$
{\displaystyle {\binom {n}{k}}p^{k}q^{n-k}}
$$



- **Expected value**: $np$

- **Variance**: $np(1-p)$

#### [Poisson](https://en.wikipedia.org/wiki/Poisson_distribution) Distribution

- **PMF**: Infinitely many independent tosses of a coin; number of tosses until first "success"

  $${\displaystyle {\frac {\lambda ^{k}e^{-\lambda }}{k!}}}$$

- **Expected value**: $\lambda$

- **Variance**: $\lambda$

#### Geometric Distribution

- **PMF**: Infinitely many independent tosses of a coin; number of tosses until first "success"

$${\displaystyle p_{X}(k) = (1-p)^{k-1}p}$$

- **Expected value**: $1/p$

- **Variance**: $(1 - p)/p^2$

- **Characteristics**: memoryless

### Property

For $X$ being a random variable taking as values integer greater than zero, we have that:

$${\displaystyle E(x) = \sum_{k=1}^{\infty}{kp_{X}(k)} = \sum_{k=1}^{\infty}{P(X\geq{k})}}$$

## Continuous Random Variables

### Definitions

#### Expected value

$${\displaystyle E(X) = \int_{x}{xf_{X}(x)dx}}$$

#### Variance

$${\displaystyle \operatorname {Var}(X)= \int_{x}{(x-\mu)^2f(x)dx}}$$

#### Cumulative Distribution Function (**CDF**)

$${\displaystyle F_{X}(x)=\operatorname {P} (X\leq x)}$$

$$F_{X}(x)=\int_ {-\infty }^{x}f_{X}(t)\,dt$$

- CDF is non descreasing

### Distributions

#### [Exponential](https://en.wikipedia.org/wiki/Exponential_distribution) Distribution

- **PDF**: this random variable is the continuous analogous of the geometric distribution, and its pdf is:

  $$f(x;\lambda) = \lambda e^{-\lambda x}$$

  for $x \geq 0$, and $0$ otherwise.

- **Expected value**: $\lambda$

- **Variance**: $\lambda$

#### [Gaussian](https://en.wikipedia.org/wiki/Normal_distribution) Distribution

- **PDF**:

$${\displaystyle f(x) = \frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}}$$

- **Expected value**: $\mu$

- **Variance**: $\sigma^2$

- Nice analytical properties, model noise, central limit theorem

#### [Poisson](https://en.wikipedia.org/wiki/Exponential_distribution) Distribution

- **PDF**: this random variable is the continuous analogous of the geometric distribution, and its pdf is:

  $$f(x;\lambda) = \lambda e^{-\lambda x}$$

  for $x \geq 0$, and $0$ otherwise.

- **Expected value**: $\lambda$

- **Variance**: $\lambda$

### Conditioning

<span style="color:blue">
  <em>Todo</em>
</span>

### Independence

<span style="color:blue">
  <em>Todo</em>
</span>

## Further Topics on Rvs

- A function of a random variable is random variable itself.

- **Derived Distributions**: Given $X$, I want to get the distribution of $Y = g(X)$

  $${\displaystyle p_Y(y) = P(g(x) = y) = \sum_{x: g(x)=y}{p_{X}(x)}}$$

  i.e. we sum the probabilities of all those $x$ for which $g(x)$ yields $y$

$${\displaystyle Y = aX + b: p_Y(y) = p_X \big( \frac{y - b}{a}\big)}$$

- **notation**: random variable $X$, numerical value $x$

- **notation**: $p_{X}(x) = P(X=x) = P({w \in \Omega \hspace{0.2cm} s.t. \hspace{0.2cm} X(w) = x})$ total probability of all outcomes for which numerical value of the rv is $x$, probability of $X$ of taking value $x$.

## Bayesian inference

- Unknown $\Theta$, treated like an rv, and not a constant (like in frequentist approach)
- Where does prior distribution of $\Theta$ come from:

  - symmetry (uniform)
  - known range
  - earlier studies
  - subjective

- Beta distribution

  $${\displaystyle f(x;\alpha ,\beta ) = c \cdot x^{\alpha -1}(1-x)^{\beta -1}}$$

## Inequalities, Convergence and WLLN

- **Markov Inequality**: given a **positive rv**, finite first moment ($E(X) < ∞$); it states that:

  $${\displaystyle \operatorname {P} (X\geq a)\leq {\frac {\operatorname {E} (X)}{a}}.}$$

  i.e. the probability of a rv of being greater than the constant $a$ should be less or equal to the expected value divided by $a$. _The intuition is that, if this doesn't hold, the expected value couldn't have the value it takes._ Here the [**video**](https://www.youtube.com/watch?v=uh-v7LchsxU). Useful only on the upper bound. Remembering that the rv should be positive greatly helps in the intuitive explanation of the inequality.

- **Chebyshev Inequality**: If I also know the variance of the rv, and I know that it is not infinite, I can get a better bound compared to Markov Inequality.

  $$\Pr(|X-\mu |\geq k\sigma )\leq {\frac {1}{k^{2}}}$$

  The proof trasforms the left argument by squaring $(x - \mu)$ and $k$, then simply applies Markov Inequality.

- $X_1, ...., X_n$ i.i.d. then
  $$
  \displaystyle{M_n = \frac{X_1 + ... + X_n}{n}} \rightarrow E[X] \\
  \displaystyle{E[M_n] = \frac{E[X_1 + ... + X_n]}{n}} = \frac{n\mu^2}{n} = \mu\\
  \displaystyle{ \operatorname{Var}[M_n] = \frac{\operatorname{Var}(X_1 + ... + X_n)}{n^2}} = \frac{n\sigma^2}{n^2} = \frac{\sigma^2}{n}\\
  $$
  

  Appling Chebyshev, we get:

  $$
  \displaystyle{P(|M_n - \mu| \geq \epsilon) \leq \frac{\operatorname{Var}(M_n)}{\epsilon^2}= \frac{\sigma^2}{n\epsilon^2} = \frac{\sigma^2}{n}}
  $$
  

  which goes to zero for a fixed $\epsilon$ and for $n$ that goes to infinity

- Convergence in probability

  1. conjecture of limit
  2. compute the probability of being $\epsilon$ away from conjectured limit

- Chernoff bound > better bound

- Hoeffding's Inequality:

  $$
  \displaystyle \displaystyle \mathbf{P}\left(\left|\overline{X}_ n-\mathbb E[X]\right|\geq \epsilon \right) \leq 2 \exp \left(-\frac{2n\epsilon ^2}{(b-a)^2}\right)\qquad \text {for all }\epsilon >0
  $$
  
- **Central Limit theorem**: given iid rvs:

  - X1,...,Xn are i.i.d.;

  - $\mathbb {E}[X_1] = \mu < \infty ,$ and $\text {Var}(X_1) = \sigma ^2 < \infty$

  then

  $${\displaystyle Z_n = \frac{X_1 + ... + X_n - \mu n}{\sigma\sqrt n}}$$

  $${\displaystyle \lim _{n\to \infty }\Pr(Z_n \leq z) = \Phi (z)}$$

  Stated in a different way:

  $$\sqrt{n}\left( \overline{X} _n - \mu \right) = \sqrt{n}\left( \left(\frac{1}{n} \sum_ {i = 1}^ n X_ i\right) - \mu \right) \xrightarrow [n \to \infty ]{(d)} Z,$$

  where $Z$ is a normal random variable with mean $0$ and variance $\sigma^2$.

  c) Dependence will not make a difference because the definition of convergence in probability involves probabilities of the form $P(|Yn−a|≥ϵ)$. These probabilities are completely determined by the marginal distributions of the random variables Yn , and these marginal distributions are the same as for the sequence Xn.

![AA](Capture.PNG)

"Consistent estimator" is defined at 3:21 of the previous lecture. In the context of statistics, the adjective "consistent" means that the estimator $Θ^n$ converges in probability towards $θ$ when $n→+∞$ for every possible value of the parameter $θ$.

Consistent estimator: converges to real one as n goes inf Unbiased: expectation is equal to the real value

C.I.

Use upper bound use sample means estimate use sample variance estimate

## Bernoulli and Poisson Processes

![AA](Capture5.PNG)

![AA](Capture6.PNG)

![AA](Capture7.PNG)

Poisson process foresees arrivals over time in an **uncoordinated** manner. The poisson distribution is:

obtained taking the limit of the binomial process having $n$ that goes to infinity and $p$ that goes to zero so that $np$ stays constant.

$$\displaystyle{P(N=n)={\frac {\lambda ^{n}}{n!}}e^{-\lambda }}$$

The probability of an arrival over a time period of time $\tau$ is equal to:

$$\tau \lambda$$

and, over an interval of lenght $\tau$:

$$\displaystyle{P(N_\tau=n)={\frac {(\lambda \tau) ^{n}}{n!}}e^{-(\lambda \tau)}}$$

$$\displaystyle{E[N_\tau]= \lambda \tau}$$

$$\displaystyle{Var[N_\tau]= \lambda \tau}$$

- **Time until first arrival**: the CDF and PDF of the time until the first arrival are respectively:

  $$\displaystyle{F(x)=1 - e^{-\lambda \tau}}$$

  $$\displaystyle{f(x)=\lambda e^{-\lambda \tau}}$$

  The last one is an Exponential distribution, which is memoryless. The time until the $k$th arrival is instead (**Erlang Distribution**):

  $$f_{Y_k}={\lambda^{k}y^{{k-1}}e^{{-\lambda y}} \over (k-1)!}$$

  $$E_{Y_k}= \frac{k}{\lambda}$$

  $$Var(Y_k)=\frac{k}{\lambda^2}$$

Also interarrival times are exponential, so $Y_k$

The sum of two independent Poisson processes (**merged process**) is Poisson and the parameter is the sum of the parameters.

In a merged process, the probability that a particular arrival came from the first stream is equal to:

$$\frac{\lambda_1}{(\lambda_2 + \lambda_1)}$$

where $\lambda_1$ and $\lambda_2$ are the parameters of the first and of the second stream respectively.

- Random Incidence

- The **Pascal random variable** is an extension of the geometric random variable. It describes the number of trials until the kth success, which is why it is sometimes called the "kth-order interarrival time for a Bernoulli process."

- for an exponential distribution:

  $$E[X^2] = 2 / \lambda^2$$

# Statistics

Trinity:

- Estimation
- Confidence intervals
- Hypoteshis testing

More on Central Limit Theorem

- **Law of Large Numbers (LLN)**

- **PDF**

  $${\displaystyle \varphi (x)={\frac {1}{\sqrt {2\pi }}}e^{-{\frac {1}{2}}x^{2}}}$$

  $${\displaystyle \varphi (x)={\frac {1}{\sigma \sqrt {2\pi }}}e^{-\frac{(x - \mu)^2}{2 \sigma^2}}}$$

- **Invariant Under Affine Transformation**: Given a Gaussian $X\sim \mathcal{N}(\mu, \sigma^2)$, $aX + b$ is still a Gaussian with mean $a\mu + b$ and variance $a^2\sigma^2$.

- **Standardization**:

  $$
  Z = \frac{X - \mu}{\sigma} \sim \mathcal{N}(0, 1)%%
  $$
  
- **Identifiability**: Recall that the parameter $θ$ is identifiable if the map $θ↦P_θ$ is injective. Here, the notation $θ↦P_θ$ denotes a function that takes as input $θ∈Θ$ and outputs a probability distribution $P_θ$. In other words, if $θ≠θ^′$ (and both in $Θ$), then $P_θ≠P_{θ^′}$

- **Symmetry**: if $X\sim \mathcal{N}(\mu, \sigma^2)$ then $-X\sim \mathcal{N}(\mu, \sigma^2)$

- **XXX**: Let $X_1,...,X_n$ be i.i.d. random variables with mean $μ$ and variance $σ^2$. Denote the sample mean by $\overline{X}_ n = (X_1 + ... + X_n)/n$. Assume that $n$ is large enough that the central limit theorem (clt) holds. The following random variable $Z$ has approximate distribution $X\sim \mathcal{N}(0, 1)$:

$$\displaystyle Z =\frac{\overline{X}_ n-\mu }{\sqrt{\sigma ^2/n}}$$

$$\displaystyle Z =\sqrt{n}\frac{\overline{X}_ n-\mu }{\sigma }$$

- The **quantile** of order $1−α$ of a variable $X$, denoted by $q_α$ (specific to a particular $X$), is the number such that $P(X≤q_α)=1−α$.

- **Convergence**: from Strong to Weak we have:

  - almost sure convergence
  - convergence in probability
  - convergence in distribution

- Convergence in probability does not imply convergence in expectation, e.g. given $T_n$ as

  $$\displaystyle \displaystyle \mathbf{P}(T_ n=0) = 1 - \frac{1}{n}$$

  $$\displaystyle \displaystyle \mathbf{P}(T_ n=2^n) = \frac{1}{n}$$

  Then

  $$\displaystyle \displaystyle \mathbf{P}\left(\left|T_ n-0\right|>\epsilon \right)=\frac{1}{n}\longrightarrow 0.$$

  Which implies convergence in probability. But:

  $$\displaystyle E[T_n] = \frac{2^ n}{n}\longrightarrow \infty$$

- Slusky Theorem: you can sum or multiply a rv that converges in distribution with a rv that converges in probability only if the one which converges in probability is a constant

- def of an **estimator**: statitics that does not depend on my unknown parameter

- **Sample Space**: where my observations lay; **Parameter Space**: where my parameters lay

- Note that the sample space of X is not unique. For example, if X∼Ber(p) , then both {0,1} and R can serve as a sample space. However, in general, we associate a random variable with its smallest possible sample space (which would be {0,1} if X∼Ber(p) ).

- parametric models: parameters have finite dimensions, nonparametric models: parameters have infinite dimensions,

A model is **well-specified** if $\exists \theta s.t. P = P_\theta$, i.e. that for some $\theta$, the distribution of th unknown is the right distribution.

Linear regression model: $(X_1, Y_1),\ldots , (X_ n,Y _n) \in \mathbb {R}^ d \times \mathbb {R}$ are i.i.d from the linear regression model $Y_ i=\beta ^\top X _i + \varepsilon_ i, \quad \varepsilon _i \stackrel{iid}{\sim } \mathcal{N}(0,1)$ for an unknown $β∈R^d$ and $X_ i \sim \mathcal{N} _d(0,I_ d)$ independent of εi .

- **Jensen's Inequality**: make an example

- **Bias of an extimator**: given the estimator of $\theta$, i.e. $\hat{\theta_n}$, the bias of the estimator is defined as:

  $$\mathbb E[\hat{\theta }_ n] - \theta .$$

- **Variance of an estimator**

- **Quadratic risk** is defined as:

  $$\mathbb E[(\hat{\theta }_ n - \theta )^2].$$

  Minimizing that means minimizing both variance and bias of the estimator. Indeed, expanding the formula we get that the quadratic risk is equal to the bias suquared plus the bias.

## Confidence Intervals

it's random, depend on data, but they do not depend on $\theta$. Defined as "probability that Confidence interval contains $\theta$".

**Example**

You have the estimator $\bar{R}_n$ as defined above. You want the probability of the distance between this estimator and the true parameter $p$ be more then $x$ with low probability (probability $α$), i.e.:

$$P(|\bar{R}_n-p|\ge x)=\alpha$$

So you first normalize the variable:

$$P(\frac{\sqrt{n} |\bar{R}_n-p|}{\sqrt{p(1-p)}}\ge \frac{\sqrt{n}x}{\sqrt{p(1-p)}})=\alpha$$

but notice that the random variable converges to a standard normal, so:

$$P(|Z|\ge \frac{\sqrt{n}x}{\sqrt{p(1-p)}})=\alpha$$

that follows because of convergence in distribution that is guaranteed by the Central Limit Theorem.

$$P\Bigg( \frac{\sqrt{n} |\bar R_n - p| }{\sqrt{p(1-p)}} \ge \frac{\sqrt{n} x }{\sqrt{p(1-p)}} \Bigg) \approx P\Bigg( |Z| \ge \frac{\sqrt{n} x }{\sqrt{p(1-p)}} \Bigg)$$

You will have this:

$$2*P(Z\ge \frac{\sqrt{n}x}{\sqrt{p(1-p)}})=\alpha$$

or using the complement:

$$2*\left(1-P(Z\lt \frac{\sqrt{n}x}{\sqrt{p(1-p)}}) \right)=\alpha$$

using the CDF:

$$2*\left(1-\Phi(\frac{\sqrt{n}x}{\sqrt{p(1-p)}}) \right)=\alpha$$

That is equivalent to:

$$\Phi(\frac{\sqrt{n}x}{\sqrt{p(1-p)}}) =1-\alpha/2$$

Inverting the CDF

$$x=\frac{\sqrt{p(1-p)}\Phi^{-1}(1-\alpha/2)}{\sqrt{n}}$$

but notice that the inverse is just the percentile of $α/2$, so:

$$x=\frac{\sqrt{p(1-p)}q_{\alpha/2}}{\sqrt{n}}$$

three solutions then:

- Conservative bound
- Solving Quadratic Equation for $p$
- Plug-in

### Asymptotic Confidence Interval

(Slide 19/61 of Chapter 2) - An interval $I$ (possibly random) whose boundaries do not depend on $θ$ (the unknown parameter) and such that

$$\lim_{n \to \infty} \mathbf{P}_\theta[\theta \in \mathcal{I}] \ge 1-\alpha, \forall \theta \in \Theta$$

is called an asymptotic confidence interval of level $1−α$. Note that the above does not necessarily mean that the interval must depend upon $n$. Nor does it mean that the interval is required to be random.

### Asymptotic Variance of an Estimator

The asymptotic variance of an estimator $\widehat{θ}$ for a parameter $θ$ is defined as $V(\widehat{\theta})$, if

$$\sqrt{n}(\widehat{\theta } - \theta ) \xrightarrow [n \to \infty ]{\mathrm{(D)}} \mathcal{N}(0, V(\widehat{\theta }))$$

### Delta Method

Roughly, if there is a sequence of random variables $X_n$ satisfying

$${{\sqrt {n}}[X_{n}-\theta ]\,{\xrightarrow {D}}\,{\mathcal {N}}(0,\sigma ^{2})},$$

where $θ$ and $σ^2$ are finite valued constants and ${\displaystyle {\xrightarrow {D}}}$ denotes convergence in distribution, then

$${\displaystyle {{\sqrt {n}}[g(X_{n})-g(\theta )]\,{\xrightarrow {D}}\,{\mathcal {N}}(0,\sigma ^{2}\cdot [g'(\theta )]^{2})}}$$

$$\displaystyle \mathcal{N}\left(0,\left(g'\left(\frac{1}{\lambda }\right)\right)^2 \left(\frac{1}{\lambda ^2}\right)\right)$$

Warning (example on the ): It's very important that we apply g′ to the value 1/λ , and not λ . We start with a consistent estimator, namely X¯¯¯¯n , whose limit is E[X]=1/λ , and the Delta method asks us to apply g′ to the limit of that consistent estimator. Be careful about this, as it is a common source of errors. The Delta method states that continuously differentiable function applied to an asymptotically normal sequence of random variables is again asymptotically normal.

- Two Sample test, e.g. difference between two averages
- One Sample test, e.g. difference vis a vis a benchmark

### Hypothesis Testing

- $H_0$: $\theta \in \Theta_0$
- $H_1$: $\theta \in \Theta_1$

Always looking for evidence against $H_0$, never evidence in favor of it. Remark: Regardless of the data, our conclusion will never be to accept the null. On observing the data, we will either reject the null in favor of the alternative OR we will fail to reject the null. In the latter case, we are not claiming that the null is true, rather we are stating that the data does not provide us with enough evidence to refute the null hypothesis.

- **Test Statistics**:

- **Type 1 and Type 2 Error**: In this example, let's say that the jury makes a type 1 error if the suspect satisfies H0 while the jury rules in favor of H1 . Let's say the jury makes a type 2 error if the suspect satisfies H1 while the jury rules in favor of H0.

  - **Type 1**: [...]

  - **Type 2**: where $Pθ(ψ_n=0)$ is the probability of the event $ψ_n=0$ under the probability distribution $P_θ$ when $θ∈Θ_1$ , i.e. the probability of not rejecting $H_0$ when $H_1$ is true

- **Power of a test**: is defined as

  $$\pi _{\psi_ n} = \inf _{\theta \in \Theta _1} (1 - \beta_ {\psi _ n}(\theta))$$

  _That_ is, the power of a test is the lowest possible value that one minus the type 2 error can take, given that $\theta$ belongs to $\Theta_1$, i.e. given that $H_0$ should be refused.

- A test $ψ$ has level $α$ if

  $$\displaystyle \alpha \geq \alpha _{\psi }(\theta )\qquad \text {for all }\, \theta \in \Theta _0$$

  _where_ $\alpha _{\psi }=\mathbf{P}_\theta (\psi =1)$ is the type 1 error. We will often use the word "level" to mean the "smallest" such level, i.e. the least upper bound of the type 1 error, defined as follows:

  $$\displaystyle \text {sup}_{\theta \in \Theta _0} \alpha_ {\psi }(\theta )$$

- Determine the smallest threshold $C$ such that the test $ψ_{n,C}$ has level $α$, i.e. the type 1 error should be equal to $\alpha$ given that $\theta \in \Theta_0$

### Total Variation Distance

It is defined as (for discrete and continuous rvs respectively):

$$\text {TV}(\mathbf{P}_{\theta }, \mathbf{P}_{\theta '})={\max _{A \subset E}}\, \big |\mathbf{P}_{\theta }(A)-\mathbf{P}_{\theta '}(A)\big |\,$$

$$\text {TV}(\mathbf{P}, \mathbf{Q}) = \frac{1}{2} \, \sum _{x \in E} |f(x) - g(x)|.$$

$$\text {TV}(\mathbf{P}, \mathbf{Q}) = \frac{1}{2} \, {\color{blue}{\int }} _{x \in E} |f(x) - g(x)|$$

Informally_, this is the largest possible difference between the probabilities that the two probability distributions can assign to the same event. Note that the distance between two distributions only depends on the distributions themselves and not their relation to each other (the joint distribution). This is why assuming $X$ and $Y$ are independent (or not) does not affect the total variation distance.

- $d(P,Q)=d(Q,P)$ (symmetric)
- $d(P,Q)≥0$ (nonnegative)
- $d(P,Q)=0⟺P=Q$ (definite)
- $d(P,V)≤d(P,Q)+d(Q,V)$ (triangle inequality)

In the above, $P=Q$ means P(A)=Q(A) for A⊂E , where E is the common sample space of P and Q .

### Kullback-Leibler (KL) Divergence

Also known as relative entropy. Let $\mathbf{P}$ and $\mathbf{Q}$ be discrete probability distributions with pmfs $p$ and $q$ respectively. Let's also assume P and Q have a common sample space E. Then the KL divergence (also known as relative entropy ) between $P$ and $\mathbf{Q}$ is defined by

$$
\text {KL}(\mathbf{P}, \mathbf{Q}) = \sum _{x \in E} p(x) \ln \left( \frac{p(x)}{q(x)} \right),
$$


where the sum is only over the support of $\mathbf{P}$.

$$
\text {KL}(\mathbf{P}, \mathbf{Q}) = {\color{blue}{\int }} _{x \in E} p(x) \ln \left( \frac{p(x)}{q(x)} \right) dx
$$


Then, _I_ know

### Maximum Likelihood

Minimizing KL divergence means maximizing the likelihood
$$
L _n(x_1, \ldots , x_ n, \theta ) = \prod _{i = 1}^ n p_\theta (x_ i)
$$
i.e. maximizing the probability of observing sample $x_1,...,x_n$ given a certain $\theta$. It is crucial that we interpret the likelihood $L_n$ as a function of $θ$. That is, $L_n$ varies as $θ$ ranges over the parameter space $Θ$.

Likelihood for a **Bernoulli**:

$$L _n(x_1, \ldots , x_ n, p) = p^{\sum _{i = 1}^n x_i} (1 - p)^{n - \sum_ {i = 1}^n x_i}$$

Likelihood for a **Gaussian**:

$$L _n(x_1, \ldots , x_ n, (\mu , \sigma ^2)) = \prod _{i =1}^ n \frac{1}{\sqrt{2 \pi } \sigma } \exp \left(-\frac{1}{2 \sigma ^2} (x_ i - \mu )^2\right) =$$

$$\frac{1}{(\sigma \sqrt{2 \pi })^ n} \exp \left(-\frac{1}{2 \sigma ^2} \sum _{i = 1}^ n (x_ i - \mu )^2\right).$$

Likelihood for a **Poisson**:

$$L _n(x_1, \ldots , x_ n, \lambda ) = \prod _{i = 1}^ n e^{-\lambda } \frac{\lambda ^{x_ i}}{{x _i}!} = e^{-n \lambda } \frac{\lambda ^{\sum_ {i = 1}^ n x _i}}{x_1 ! \cdots x_ n !}.$$

Likelihood for a **Exponential**:

$$L _n(x_1, \ldots , x_ n, \lambda ) = \lambda^n e^{-\lambda \sum _{i = 1}^ n x_ i} $$

Likelihood for a **Uniform**:

Note that $\displaystyle \max _{x} f(x)$ _is_ the maximum value of the function, which is different from $\text {argmax}f(x)$, the value of the argument $x$ at which the function is maximum.

A function $g:I→R$ is **concave** (or concave down), where $I$ is an interval, if for all pairs of real numbers $x_1<x_2∈I$

$$\displaystyle \displaystyle g(tx_1+(1-t)x_2)\geq tg(x_1)+(1-t)g(x_2)\qquad \text {for all } \, 0 < t < 1.$$

A function $g:I→R$ is **convex** (or concave down), where $I$ is an interval, if for all pairs of real numbers $x_1<x_2∈I$

$$\displaystyle \displaystyle g(tx_1+(1-t)x_2)\leq tg(x_1)+(1-t)g(x_2)\qquad \text {for all } \, 0 < t < 1.$$

- concave if and only if $g′′(x)≤0$ for all $x∈I$;
- convex if and only if $g′′(x) \geq 0$ for all $x∈I$;

$$\displaystyle \displaystyle \sqrt{n} \left(\mathbf{g}(\mathbf{T} _n) - \mathbf{g}(\vec{\theta }) \right) \xrightarrow [n\to \infty ]{(d)} \nabla \mathbf{g}(\vec{\theta })^ T\mathbf{T}\, \sim \, \displaystyle \mathcal{N}\left(0, \nabla \mathbf{g}(\vec{\theta })^ T \Sigma_ {\mathbf{X}} \nabla \mathbf{g}(\vec{\theta })\right)\qquad (\mathbf{T}\sim \mathcal{N}(\mathbf{0},\Sigma _\mathbf{X})).$$

**Reciprocal Rule for derivative**

**Matrix Determinant**

**Random Variables Independence**

$${\displaystyle f_{X,Y}(x,y)=f_{X}(x)\cdot f_{Y}(y)}$$

### Fisher Information

Let $θ∈Θ⊂R_d$ and let $(E,{P_θ})$ be a statistical model. Let $f_θ(x)$ be the pdf of the distribution $P_θ$. Then, the Fisher information of the statistical model is:

$$I(θ)=Cov(∇ℓ(θ))=−E[Hℓ(θ)]$$

where $ℓ(θ)=lnfθ(X)$. The definition when the distribution has a pmf $p_θ(x)$ is also the same, with the expectation taken with respect to the pmf. It is also defined as minus the expectation of the second derivative of the log-likelihood.

$$\displaystyle \mathcal{I}(\theta) = -\mathbb E[\ell ^{\prime \prime }(\theta)]$$

### Asymptotic Normality of the MLE

Only applicable if:

1. The parameter is identifiable.
2. For all $\theta \rightarrow \Theta$, the support of $P_\Theta$ does not depend on $\theta$;
3. $\theta^\ast$ is not on the boundary of $\theta$;
4. $I(\theta)$ is invertible

Consider the statistical model $({Ber(θ)}∈(0,1))$. Let $ℓ(θ)$ denote the log-likelihood of one observation of this model. You observe samples $X_1,...,X_n∼Ber(θ∗)$ and construct the MLE $\widehat{\theta }_ n^{\text {MLE}}$ for $θ^∗$. By the theorem for the convergence of the MLE (you are allowed to assume that all necessary conditions for this theorem hold), this implies that:

$$\sqrt{n}(\widehat{\theta }_ n^{\text {MLE}} - \theta ^*) \xrightarrow [n \to \infty ]{(d)} \mathcal{N}(0, \sigma ^2)$$

for some constant $σ^2$ that depends on $θ^∗$.

Let* $X_1,...,X_n∼iid P_{θ^∗}$ for some true parameter $θ^∗∈R_d$. We construct the associated statistical model and the maximum likelihood estimator $\hatθ_{n}^{MLE}$ for $θ^∗$. Recall that, under some technical conditions,

$$\sqrt{n}(\widehat{\theta }_ n^{MLE} - \theta) \xrightarrow [n \to \infty ]{(d)} \mathcal{N}(0, \mathcal{I}(\theta)^{-1})$$

where $I(θ^∗)$ denotes the Fisher information. That is, the MLE $\hatθ_{n}^{MLE}$ is asymptotically normal with asymptotic covariance matrix $\mathcal{I}(θ^∗)^{−1}$. Standardizing the statement of asymptotic normality above we get:

$$\sqrt{n} \mathcal{I}(\theta)^{{\color{blue}{a}} } (\widehat{\theta } _n^{MLE} - \theta)\xrightarrow [n \to \infty ]{(d)} \mathcal{N}(0, I_{d \times d})$$

### Method of Moments

$$\widehat{\theta } _n^{\text {MM}} := \psi ^{-1}\left( \frac{1}{n} \sum_ {i = 1}^ n X _i, \frac{1}{n} \sum_ {i = 1}^ n X _i^2, \ldots , \frac{1}{n} \sum_ {i = 1}^ n X_i^ d \right)$$

### M-Estimation

Let $X_1,...,X_n$ be i.i.d. with some unknown distribution $P$ and an associated parameter $μ^∗$. We make no modeling assumption that $P$ is from any particular family of distributions.

An M-estimator $\hat{μ}$ of the parameter $μ^∗$ is the **argmin of an estimator of a function $Q(μ)$** of the parameter which satisfies the following:

- $Q(μ)=E[ρ(X,μ)]$ for some function ρ:E×M→R, where M is the set of all possible values of the unknown true parameter $μ^∗$;

- $Q(μ)$ attains a unique minimum at $μ=μ^∗$, in M. That is, $\, \displaystyle \text {argmin}_{\mu \in \mathcal{M}}\mathcal{Q}(\mu ) \, =\, \mu ^*$.

In* general, the goal is to find the loss function ρ such $Q(μ)=E[ρ(X,μ)]$ has the properties stated above.

Note that the function $ρ(X,μ)$ is in particular a function of the random variable $X$, and the expectation in $E[ρ(X,μ)]$ is to be taken against the true distribution $P$ of $X$, with associated parameter value $μ^∗$.

Because $Q(μ)$ is an expectation, we can construct a (consistent) estimator of $Q(μ)$ by replacing the expectation in its definition by the sample mean.

For this example, you gain nothing. But notice that this is a generalization of MLE, so you could use just use KL and get MLE, if you use the squared difference, you get the sample mean. But you can use other distances (divergences) and get more interesting estimators.

#### Cauchy Distribution

$$f_ m(x) = \frac{1}{\pi } \frac{1}{1 + (x - m)^2}.$$

where $m$ is location parameter; $m$ is also the median, but the mean is not identifiable.

#### Huber's Loss

$$h_\delta (x) = \begin{cases} \frac{x^2}{2} \quad \text {if} \, \, \left| x \right| \le \delta \ \delta ( \left| x \right| - \delta /2 ) \quad \text {if} \, \, \left| x \right| > \delta \end{cases}$$

#### Parametric Hypothesis Testing, Finite Sample Sizes, and Chi-Squared and Student's T Distributions

Test hypotheses when the i.i.d. data samples have a Gaussian distribution.

Recognize when you cannot assume the test statistic to be Gaussian (in the small sample sizes regime).

#### Chi-Squared Distribution

The $χ^2_d$ distribution with $d$ degrees of freedom is given by the distribution of:

$$Z_1^2 + Z_2^2 + \cdots + Z_ d^2,$$

where $Z_1, \ldots , Z_ d \stackrel{iid}{\sim } \mathcal{N}(0,1)$; $E[X]= d$; $VAR[X]= 2d$

Let $Z∼N(0,I_{d×d})$ denote a random vector whose components are standard Gaussians: $Z(1),...,Z(d)∼N(0,1)$. Which one of the following random variables has a chi-squared distribution with d degrees of freedom?

$${\displaystyle \left|{\boldsymbol {x}}\right|_{2}:={\sqrt {x_{1}^{2}+\cdots +x_{n}^{2}}}.}$$

Example: $X_1, \ldots , X_ n \stackrel{iid}{\sim } \mathcal{N}(0, \sigma ^2)$ and let:

$$V _n = \frac{1}{n} \sum_ {i = 1}^ n X_ i^2$$

define $A$ so that $AV_{n}∼χ2$.

$$\frac{n}{\sigma ^2} V_n = \sum_ {i = 1}^ n \frac{X _i^2}{\sigma ^2}= \sum_ {i = 1}^ n \left(\frac{X_ i}{\sigma }\right)^2,$$

#### Cochran's Theorem

Cochran's theorem states that if $X_1,...,X_n \stackrel{iid}{\sim } \mathcal{N} (μ,σ^2)$, then the sample variance

$$S _n := \frac{1}{n} \left(\sum_ {i = 1}^ n X _i^2\right) - (\overline{X}_ n)^2$$

satisfies:

- $\overline{X}_ n$ is independent of $S_n$, and
- $\frac{nS_n}{σ^2}∼χ^{2}_{n−1}$ .

Relates the sample variance and sample mean when the data samples are i.i.d. Gaussian.

#### Student's T Distribution

The definition of the student's T distribution with $n−1$ degrees of freedom is that it is given by the distribution of $\frac{Z}{\sqrt{V/(n−1)}}$ where $Z∼N(0,1)$, $V∼χ^2_{n−1}$ and $Z$ and $V$ are independent (by Cochrane). Since we are dividing by $V$, a $χ^2$ random variable, then $T_n$ will not have the same distribution as $N(0,1)$ for all $n≥2$.

Now consider the test statistics:

$$T_{n} := \sqrt{n} \left( \frac{\overline{X}_ n - \mu }{\sqrt{\frac{1}{n- 1} \sum _{i = 1}^ n (X_ i - \overline{X}_ n)^2} } \right).$$

Student's T test for non-asymptotic level $α$ is **only applicable** if $X$ is a Gaussian random variable; in that case only the test statistic follow a Student's T distribution for a finite number of samples $n$.

The denominator represents the unbiased estimator (see the $n-1$) for the sample standard deviation. It $T_n$ follows a student t distribution (it is a key assumption that the data is Gaussian. Otherwise, the test statistic $T_n$ will not necessarily follow the student's T distribution and, hence, may not even be pivotal). The Student's t distribution converges to the standard normal with increasing sample sizes.

Advantages:

- Non Asymptotic
- Can be run on small samples
- Can also use it for large sample Sizes

Drawbacks:

- It assumes Gaussian samples

#### Wald's test

Starting from the fact that (exploiting the normality of the MLE):

$$\sqrt{n} \mathcal{I}(\theta)^{{{\frac{1}{2}}} } (\widehat{\theta } _n^{MLE} - \theta)\xrightarrow [n \to \infty ]{(d)} \mathcal{N}(0, I_{d \times d})$$

The Wald test, by taking a quadratic form, reduces the test problem to a one-sided one-dimensional comparison. The test is of the form:

$$\left| \sqrt{n}\, \mathcal{I}(\mathbf{0})^{1/2}(\widehat{\theta } _n^{MLE}- \mathbf{0}) \right| ^2 \xrightarrow [n\to \infty ]{(d)} \chi ^2_ k\,$$

where $k$ is the number of parameters. This is done **under the NULL hypothesis**.

_Remark_: Real matrices satisfying $\, \mathbf{M}^{T}=\mathbf{M}^{-1}\,$ (or equivalently $\, \mathbf{M}\mathbf{M}^ T=\mathbf{M}^ T\mathbf{M}=\mathbf{1}_{d\times d},\,$) are called orthogonal matrices. In general, in $d$ dimensions and for any orthogonal matrix $\, \mathbf{M},\,$, $\, \mathbf{MZ},\,$ is also a standard multivariate Gaussian vector if $\mathbf{Z}$ is a standard multivariate Gaussian.

#### Likelihood ratio test

$$\psi _C = \mathbf{1}\left( \frac{L_ n(x_1, \ldots , x_ n; \theta _1 )}{L_ n(x_1, \ldots , x_ n; \theta _0 )} > C \right).$$

Perform the likelihood ratio test for a family of hypothesis testing questions. Use an asymptotically normal estimator to test implicit hypotheses involving an unknown parameter, i.e.:

$$\psi _C = \mathbf{1}\left( \frac{L_ n(x_1, \ldots , x_ n; \widehat{\theta _n}^{MLE})}{L_ n(x_1, \ldots , x_ n; \theta _0 )} > C \right).$$

which becomes (taking log and multiplying by 2):

$$T _n = 2 \left( \ell_ n(\widehat{\theta _n}^{MLE}) - \ell_ n(\widehat{\theta _ n}^{c}) \right)$$

with:

$$T _n \xrightarrow [n \to \infty ]{(d)} \chi_ {d-r}^2$$

#### Multinomial Distribution

[here (part 4)](https://courses.edx.org/courses/course-v1:MITx+18.6501x+3T2019/courseware/unit4/u04s05_hypotesting/1?activate_block_id=block-v1%3AMITx%2B18.6501x%2B3T2019%2Btype%40vertical%2Bblock%40u04s05_hypotesting-tab1) - copy it

#### Goodness of fit

Suppose you observe iid samples $X_1, \ldots , X_ n \sim P$ from some **unknown** distribution $\mathbf{P}$. Let $F$ denote a family of known distributions (e.g, F it be the family of normal distributions ${ \mathcal{N}(\mu , \sigma ^2) }_{\mu \in \mathbb {R}, \sigma ^2 > 0}$). In the topic of goodness of fit testing, our goal is to answer the question "Does $\mathbf{P}$ belong to the family $F$?".

#### Chi Square Test

**for Multinomial distributions** Let $\widehat{\mathbf{p}}$ denote the MLE for a categorical statistical model $( { a_1, \ldots , a_ K } , { \mathbf{P}_{\mathbf{p}} }_ {\mathbf{p} \in \Delta _ K})$ . Let p∗ denote the true parameter. Then n−−√(pˆ−p∗) is asymptotically normal and

$$n \sum_{i = 1}^K \frac{(\widehat{p_ i} - p_i^\ast)^2}{p_i^\ast} \xrightarrow [n \to \infty ]{(d)} \chi_{K -1}^2$$

This test derives from the Wald's test, and $K-1$ degrees of freedom derive from the dependency between the parameters in the multinomial.

<span style="color:blue">The test is asymptotic (e.g. only applies for $n \rightarrow \infty$)</span>

y

#### Empirical CDF

The empirical cumulative distribution function , also called the empirical cdf, is the random function:

$$\displaystyle t \mapsto \frac{1}{n} \sum _{i = 1}^ n \mathbf{1}(X_ i \leq t).$$

_Law of Large Numbers_ $\mapsto$ uniformly $\mapsto$ **Glivenko Cantelli**

_Central Limit Theorem_ $\mapsto$ uniformly $\mapsto$ **Donsker's Theorem**

**Donsker's Theorem** states that, looking at the worse possible $t$:

$$\sqrt{n} \sup _{t \in \mathbb {R}} |F_ n(t) - F(t)| \xrightarrow [n \to \infty ]{(d)} \sup_{0 \leq x \leq 1} |\mathbb {B}(x)|,$$

where $\mathbb {B}(x)$ is a [brownian bridge](https://en.wikipedia.org/wiki/Brownian_bridge).

#### Kolmogrov-Smirnoff Test

The Kolmogorov-Smirnov test statistic is defined as (also called $L_\infty$ distance)

$${T _n = \sup_{t \in \mathbb {R}} \sqrt{n} \bigg| F_n(t) - F^0(t) \bigg|}$$

and the Kolmogorov-Smirnov test is

$$\displaystyle \displaystyle \mathbf{1}(T _n>q_\alpha )\qquad \text {where } q_\alpha =q_\alpha (\sup_{t \in [0,1]}\left| \mathbb {B}(t) \right|).$$

By definition, $T_n$ is a pivotal statistic under $H_0$.

[**Pivotal distribution** is a distribution which does not depend on unknown parameters, so you can compute quantiles.]

Here, $q_\alpha =q_\alpha (\sup_{t \in [0,1]}\left| \mathbb {B}(t) \right|)\,$ is the $(1−α)$ quantile of the supremum of the Brownian bridge as in Donsker's Theorem. It can be computed explicitly as follows:

$$\displaystyle Tn = \sqrt{n}\sup _{t \in \mathbb {R}} \bigg| F_ n(t) - F^0(t) \bigg|$$

$\sqrt{n}\max_{i=1,\ldots ,n}\left{ \max \left(\left| \frac{i-1}{n}-F^0(X_{(i)}) \right|,\left| \frac{i}{n}-F^0(X_{(i)}) \right| \right) \right}$

[INVERSE TRANSFORM SAMPLING](https://en.wikipedia.org/wiki/Inverse_transform_sampling)

where X(i) is the order statistic , and represents the i(th) smallest value of the sample. For example, X(1) is the smallest and X(n) is the greatest of a sample of size n .

```python

def T(n):
    SIMSIZE=1000000
    i = np.arange(1,n+1)[:,np.newaxis]
    u = np.random.uniform(size=(n,SIMSIZE))
    u.sort(axis=0)
    return np.max( np.maximum(np.abs((i-1)/n - u), np.abs(i/n - u)), axis=0 )
```

#### Other Goodness of Fit Tests

- Cramer - Von Mises ($L_2$)
- [...]

#### Kolmogrov-Lilliefors Test

Till now, $F$ was completely deterministic, with given parameters. But what if we want to test wether a distribution is e.g. Gaussian, no matter the parameters?

First, I substitute mean and variance with sample ones. But this makes $F$ artificially more stick to the data. But if we plug in estimators for $μ$ and $σ^2$ (and not their true values), the Donker's convergence no longer holds.

The **Kolmogorov-Smirnov** test was designed to test if the data has a specific distribution; it is not useful for deciding whether or not the true distribution $\, \mathbf{P}\,$ lies in a given family of distributions. In this case we use the **Kolmogrov-Lilliefors** test.

#### QQ plots

[Here](https://stats.stackexchange.com/questions/101274/how-to-interpret-a-qq-plot) and [here](https://seankross.com/2016/02/29/A-Q-Q-Plot-Dissection-Kit.html)

### Bayesian Statistics

In the Bayesian set-up, **we do not even assume that there exists a true parameter**, or at least we model it as a random variable to represent our uncertainty. we use the data to update our prior belief about a parameter and transform it into a posterior belief, which is reflected by a posterior distribution. In this framework, **we model the true parameter as a random variable** and update its distribution as we receive more data.

Usually a **beta distribution** is used to represent the distribution of the prior, especially if we are dealing with a Bernoulli random variable. Recall that the Beta distribution in $x$ is defined as the distribution with support $[0,1]$ and pdf:

$$C(\alpha , \beta ) x^{\alpha -1}(1-x)^{\beta -1}$$

where $α$ and $β$ are parameters that satisfy $α>0$, $β>0$. Here, $C(α,β)$ is a normalization constant that does not depend on $x$ (simulate a beta distribution [**here**](http://eurekastatistics.com/beta-distribution-pdf-grapher/)).

After observing the available sample $X_1, . . . ,X_n$ we can update our belief about $p$ by taking its distribution conditionally on the data, which is called the posterior distribution.

We start with the conditional likelihood:

$$L(X_1, \ldots , X_ n | \theta )$$

**Note**: this is exactly the same likelihood we used in the $MLE$ approach.

We want to get:

$$\pi (\theta |X_1, \ldots , X_ n)$$

In this way:

$$\displaystyle \pi (\theta |X_1,\dots ,X_ n)=\frac{L_n(X_1,\dots ,X_ n|\theta )\pi (\theta )}{\int_\Theta L_n(X_1,\dots ,X_ n|t)\pi (t)\; dt}$$

which is equal to say:

$$\displaystyle \pi (\theta |X_1,\dots ,X_ n)\propto L_n(X_1,\dots ,X_ n|\theta )\pi (\theta )$$

Since the denominator is independent of \theta

In the bernoulli case:

$$p _n(X_1,\dots ,X_ n|\theta )=\theta ^{\sum _{i=1}^ n X_ i}(1-\theta )^{n-\sum _{i=1}^ n X_ i}.$$

$$\displaystyle \pi (\theta |X_1,\dots ,X_ n) \propto p _n(X_1,\dots ,X_ n|\theta )\pi (\theta )$$

$$\displaystyle ... \propto \theta ^{a-1}(1-\theta )^{b-1}\theta ^{\sum _{i=1}^ n X_ i}(1-\theta )^{n-\sum _{i=1}^ n X_ i}$$

$$\displaystyle ... \propto \theta ^{a+\sum _{i=1}^ n X_ i -1}(1-\theta )^{b+n-\sum _{i=1}^ n X_ i -1}.$$

According to Bayes' rule, the posterior distribution (up to a constant of proportionality) is computed by multiplying the prior and posterior distributions taken as a function of the parameter. As a result, we need the full distribution for $π(\theta)$ as well as the likelihood function $L(X_1,X_2,...,X_n|\theta)$.

#### Uniformative priors

When I don't have any prior knowledge:

- if $\Theta$ is bounded: **uniform**
- if $\Theta$ is unbounded: **improper prior**, we set formally equal to $1$, which integrates to $\infty$

For both of them, $\pi(\theta) = 1$

#### Jeffrey's prior

"Invariant under rescaling of the parameter". It's the _non-informative_ prior equal to:

$$\pi_{J}(\theta ) \propto \sqrt{\text {det}I(\theta )},$$

Where $det$ is the determinant (trying to fit everyone in one number) and $I$ is the Fisher Information. So more weight is given when $I(θ)$ is high. The Fisher information is also the reciprocal of the MLE variance, so when the Fisher information is high, the MLE variance is low and thus the MLE has less uncertainty. Combining, we get that the Jeffreys prior gives more weight to values of θ whose MLE estimate has less uncertainty.

Continuing from the above reasoning, when the MLE estimate has less uncertainty and we are able to estimate it more precisely. This corresponds to the data giving more information about the parameter when the Jeffreys prior yields larger values.

Again, Jeffreys prior gives more weight to regions with high Fisher information . By the given interpretation for the Fisher information, this means that at these areas, a small change to θ will influence the data relatively more, or in other words, potential outcomes are more sensitive to slight changes in θ .

#### Conjugate Priors

One side concept introduced in the second Bayesian lecture is the conjugate prior. Simply put, a prior distribution π(θ) is called conjugate to the data model, given by the likelihood function L(Xi|θ) , if the posterior distribution π(θ|X1,X2,...,Xn) is part of the same distribution family as the prior.

> The way I understand this is seeing the Fisher information (and its square root, that is, Jeffrey's prior) as a function of (\theta), i.e. the parameter we are estimating. This parameter lives in a certain space, and some parts of that space may be "flat", in the sense that if you choose a point or another in that region the model will "fit" the data similarly. On the other hand, there are regions for which small changes to (\theta) make big differences to how well the model fits the data. These later regions have high Fisher information.

### Linear Regression

Given a joint probability distribution P for the random pair $(X,Y)$, the regression function of Y with respect to X is defined as

$$\nu (x) = \mathbb E[Y | X = x] = \sum_{\Omega_Y} y \cdot \mathbf{P}(Y = y \; |\; X = x)$$

which tells us the average value of $Y$ given the knowledge that $X=x$. In the case of continuous distributions where we can compute the conditional density $f(y|x)$, the expression on the right hand side is replaced with an integral:

$$\mathbb E[Y | X = x] = \int_{\Omega_Y} y f(y | x) dy$$

In Linear Regression , we will work with the assumption that the regression function<br>
$ν(x):=E[Y|X=x]$ is linear, so that

$$ν(x)=a+bx$$

we will be studying the Least Squares Estimator. It is an estimator $(\hat{a}, \hat{b})$ so that $\hat{Y}=\hat{a}+\hat{b}X$ is "close" (in some distance metric) to the actual $Y$ as often as possible. Assume $Var(X)≠0$. The **theoretical linear (least squares) regression** of $Y$ on $X$ prescribes that we find a pair of real numbers $a$ and $b$ that **minimize** $E[(Y−a−bX)^2]$, over all possible choices of the pair $(a,b)$; the $a$ and $b$ that minimize the squared error are:

$$a = \mathbb E[Y] - \frac{\textsf{Cov}(X,Y)}{\textsf{Var}(X)} \mathbb E[X], \qquad b = \frac{\textsf{Cov}(X,Y)}{\textsf{Var}(X)}$$

In **empirical linear regression**, we are given a collection of points ${(x_i, y_i) }_{i=1}^{n}$. The goal is to fit a linear model $Y=a+bX+ε$ by computing the Least Squares Estimator, which minimizes the loss function

$$\frac{1}{n} \sum_{i=1}^ n (y_i - (a + bx_i))^2.$$

Using the same technique as in the problems on theoretical linear regression, one obtains the solution

$$\hat{a} = \overline{y} - \frac{\overline{xy} - \overline{x}\cdot \overline{y}}{\overline{x^2} - \overline{x}^2} \overline{x} \qquad \hat{b} = \frac{\overline{xy} - \overline{x}\cdot \overline{y}}{\overline{x^2} - \overline{x}^2}.$$

In this particular case, this is precisely what one obtains by taking the least squares solution for the theoretical linear regression problem and replacing each term with their empirical counterparts according to the plug-in principle.

> The **rank** of a matrix is defined as (a) the maximum number of linearly independent column vectors in the matrix or (b) the maximum number of linearly independent row vectors in the matrix. Both definitions are equivalent. For an r x c matrix, If r is less than c, then the maximum rank of the matrix is r.

The model is homoscedastic if $ε_1,...,ε_n$ are i.i.d.

#### Multivariate Case

analytic computation of the LSE (which is also MLE) yields:

$$\hat{{\boldsymbol \beta }} = (\mathbb {X}^ T \mathbb {X})^{-1} \mathbb {X}^ T \mathbf Y.$$

And it is distributed:

$$\hat{{\boldsymbol \beta }} \sim \mathcal{N}(\beta , \sigma ^2 (\mathbb {X}^ T \mathbb {X})^{-1}).$$

$Y$ is distributed:

$$Y∼N(X^{T}β,σ^2I_{n})$$

And, for instance, in the one dimensional case, if we assume that $ε∼N(0,σ^2I_{1000})$ for some fixed $σ^2$, so that $Y∼N(Xβ,σ^2I_{1000})$. The quadratic risk of $\hat{β}$ and the prediction error are respectively:

$$\mathbb E[| \hat{{\boldsymbol \beta }} - {\boldsymbol \beta }|_2^2 = \sigma^2 \mathrm{tr}((\mathbb {X}^ T\mathbb {X})^{-1})$$

$$\mathbb E[ | \mathbf Y- \mathbb {X}\hat{{\boldsymbol \beta }} |_2^2 ] = \sigma^2(n-p)$$

$tr$ means_ "trace" and it is the sum of all the diagonal entries.

Doing inference regarding regression means producing **non-asymptotic** estimates

<span style="color:blue">RECITATION 23 FOR SPECTRAL THEOREM, ORTHOGONAL AND ORTHONORMAL VECTORS</span>

; this will be the basis of PCA; given a _symmetric square matrix_ $A$, it can be decomposed in:

$$A = V \Lambda V^T$$

Let $A$ be a square $n×n$ matrix with n linearly independent eigenvectors qi (where i = 1, ..., n). Then A can be factorized as

$${\displaystyle \mathbf {A} =\mathbf {Q} \mathbf {\Lambda } \mathbf {Q} ^{-1}}$$

where $Q$ is the square $n×n$ matrix whose $i^{th}$ column is the eigenvector $q_i$ of $A$, and $Λ$ is the diagonal matrix whose diagonal elements are the corresponding eigenvalues, $Λ_{ii} = λ_i$.

where $\Lambda$ is a diagonal matrix and $V$ is a collection of orthogonal vectors

**Projection matrixes**, then eigenvalues are $+1$ or $-1$

Consider the model Y|X∼N(XTβ,1) , where X is a p -dimensional random variable. Here, β is a fixed constant. Indicate whether the following statements are true, or false.

- $E[Y|X]$ is a constant random variable.
- The expected value of $Y$, $E[Y]$ is a constant random variable, if we assume that each Xi has mean μ. Indeed, $\mathbb {E}[Y]=\mathbb {E}[\mathbb {E}[Y|\mathbf X]]=\mathbb {E}[\mathbf X^ T\beta ]=\sum_{i=1}^ p \beta_i\mu$, which is constant, using the law of iterated expectations.
- If $X_i$'s are iid Gaussian, then the conditional mean, $E[Y|X]$ is a Gaussian random variable. Indeed, $\mathbf X^ T\beta = \sum_{i=1}^ p X_ i\beta_i$ is a sum of iid Gaussian random variables, and is itself a Gaussian random variable.

### Generalized Linear models

$Y | X=x \mathcal{}$ some family distribution, these family is the family for which the conditions of the asymptotic normality of the central limit theorems uphold, and the other ones won't be part of this family (bernoulli, exponential, Poisson).

As it turns out, this comes at a cost: finding the Maximum Likelihood Estimator becomes more difficult (in general). We relax the assumption that $μ$ is linear. Instead, we assume that $g∘μ$ is linear, for some function $g$:

$$g(\mu (\mathbf x)) = \mathbf x^T \beta.$$

The function $g$ is assumed to be known, and is referred to as the **link function**. Through an appropriate choice of the link function, which depends on the model, we should be able to compute an estimator $\hat{β}$, usually the MLE.

#### Exponential Families

Recall that the one-parameter canonical exponential family have pdf/pmf parametrized by $θ$ of the form

$$\displaystyle \displaystyle f_\theta (y) = \exp \left( \frac{y \theta - b(\theta )}{\phi } + c(y,\phi ) \right)$$

where $b$ and $c$ are known functions, and $ϕ$ is a known number referred to as the dispersion parameter. The function $b(θ)$ is also known as the log-partition function.

Note that b(θ) does not depend on y and c(y,ϕ) does not depend on θ .

**Fisher info refresh and derivation** <https://courses.edx.org/courses/course-v1:MITx+18.6501x+3T2019/courseware/unit3/u03s04_methodestimation/3>

**Link function** goes from $\mu$ to $X^T\beta$, not the other way around

The canonical exponential families, parametrized by $θ$, with the log-partition function $b(θ)$ having the property that $b′(θ)=μ$. Recall that in GLMs, the point of the link function is to assume $g(μ(x))=x^Tβ$, where $μ$ is the regression function: the mean of $Y$ given $X=x$, $E[Y|X=x]$.

Based on the properties of the log-partition function $b$, we derived previously that $b′(θ)=μ$, so we have the identity $g(μ)=(b′)^{−1}(μ)$.

The assumptions of a distribution for Y and a link function g(μ(x)) relate Y and X=x through the following equation:

$$\displaystyle g(\mu (\mathbf{x})=\mathbb {E}[Y | \mathbf{X}=\mathbf{x}]) = \mathbf{x}^ T {\boldsymbol \beta }.$$

**Poisson Case**: the function $b(θ)=e^θ$ for the Poisson exponential family. Further, $ϕ=1$ for the Poisson exponential family. Therefore, the log-likelihood function

$$\displaystyle \ell _n(\mathbf Y,\mathbb {X},{\boldsymbol \beta }) = \sum_ i \frac{Y _i h(X_ i^ T {\boldsymbol \beta }) - b(h(X_ i^ T {\boldsymbol \beta }))}{\phi } + c$$

becomes

$$\displaystyle \ell _n(\mathbf Y,\mathbb {X},{\boldsymbol \beta }) = \sum_ i \left(Y _i X_ i^ T {\boldsymbol \beta }- e^{X_ i^ T {\boldsymbol \beta }}\right) + c.$$

[**Chain Rule Reminder**: $(f\circ g)'=(f'\circ g)\cdot g'.$]

### PCA

Given the vector of dimension $1 \times d$

$$\mathbf X_i = (X_i^{(1)}, ..., X_i^{(d)})^T, \space\space\space i = 1,..., n$$

Let $X_i,i=1,...,n$ be iid data points in $\mathbb {R}^d$. As presented in the lecture and given in the slides, the empirical covariance matrix is:

$$\displaystyle S \triangleq \frac{1}{n} \sum_{i=1}^{n} \left(\mathbf X_i \mathbf X_i^ T \right) - \overline{\mathbf X}~ \overline{\mathbf X}^T,$$

where empirical mean $\overline{\mathbf X}$ is:

$$\displaystyle \overline{\mathbf X} \triangleq \frac{1}{n} \sum_{i=1}^{n} \mathbf X_i = (\overline{\mathbf X}^{(1)}, ..., \overline{\mathbf X}^{(d)})^T = \frac{1}{n} \mathbb X^T \mathbb{1}$$

#### TODO

**Write Fisher info as second derivative, draw beta distribution in Bernoulli case and show with simulations how for close to 0 and 1 the asymptotic variance is lower; compute inverse of a function**



# Machine Learning

### Points and Vectors

#### Norm of a vector

Norm: Answer the question how big is a vector

- Norm of a vector: ${\displaystyle \left\|{\boldsymbol {x}}\right\|_{2}:={\sqrt {x_{1}^{2}+\cdots +x_{n}^{2}}}.}$
- NumPy: `numpy.linalg.norm(x)`

If it is not specified, it is assumed to be the 2-norm, i.e. the Euclidian distance. It is also known as "length".

#### Dot product of vector

Aka "scalar product" or "inner product". It has a relationship on how vectors are arranged relative to each other

- Algebraic definition: $x⋅y≡x′y:=∑_{i=1}^nx_i∗y_i$
- Geometric definition: $x⋅y:=∥x∥∗∥y∥∗\cos(θ)$ (where $θ$ is the angle between the two vectors)
- NumPy: `np.dot(x,y)`

Note that using the two definitions and the `arccos`, the inverse function for the cosine, you can retrieve the angle between two functions as `angle_x_y = acos(dot(x,y)/(norm(x)*norm(y)))`.

> **The dot product of two orthogonal vectors is zero. The dot product of the two column matrices that represent them is zero**.

#### Unit Vector

A unit vector is a vector with length $1$. The length of a vector is also called its norm. Given any vector $x$, the unit vector pointing in the same direction as x is defined as:
$$
\frac{x}{||x||}
$$
Where $||x||$ is the norm/length of the vector $x$

#### Projection

[Vector Projection](https://medium.com/linear-algebra-basics/scalar-projection-vector-projection-5076d89ed8a8)

Matrix rank!

### Planes

An (hyper)plane in n dimensions is a $n−1$ dimensional subspace defined by a linear relation. For example, if $n=3$ hyperplanes span $2$ dimensions (and they are just called "planes"). As hyperplanes separate the space into two sides, we can use (hyper)planes to set boundaries in classification problems, i.e. to discriminate all points on one side of the plane vs all the point on the other side.

- *Normal* of a plane: any vector perpendicular to the plane.
- *Offset of the plane with the origin*: the distance of the plan with the origin, that is the specific normal between the origin and the plane

For any point in the original space, it is part of the plane only if its vector is perpendicular to the normal, that is x is part of the plane if (x−offset)⋅normal=0.

For example, in two dimensions, let's consider the "plane" ("line" in 1 d) between the points (4,0) and (0,4).

Here the offset is (2,2) and one possible normal is (4,4).

Let's consider the point a = (1,3). As (a−offset)⋅normal=([13]−[22])⋅[44]=0, *a* is part of the plane.

Let's consider the point b = (1,4). As (b−offset)⋅normal=([14]−[22])⋅[44]=4, *b* is not part of the plane.

Removing the offset creates a vector relative to a point on the plane, and then the dot product is used to check if that vector is orthogonal to the normal departing the plane from the same point.

Note that the equation (x−offset)⋅normal can be rewritten equivalently as (x⋅normal)−(offset⋅normal), where the second term will be a constant of the plane.

### Loss Function, Gradient Descent, and Chain Rule

**Loss Function**

The loss function, aka the *cost function* or the *race function*, is some way for us to value how far is our model from the data that we have. We first define an "error" or "Loss". For example in Linear Regression the "error" is the Euclidean distance between the predicted and the observed value:

$$
L(x,y;Θ)=∑_{n=1}^i|\hat{y}−y|=∑_{n=1}^i|θ_1x+θ_2−y|
$$
The objective is to minimize the loss function by changing the parameter theta. How?

**Gradient Descent**

The most common iterative algorithm to find the minimum of a function is the gradient descent. We compute the loss function with a set of initial parameter(s), we compute the gradient of the function (the derivative concerning the various parameters), and we move our parameter(s) of a small delta against the direction of the gradient at each step:

$$
\hat{θ}_{s+1}=\hat{θ}_s−γ∇L(x,y;θ)
$$
The γ parameter is known as the *learning rate*.

- too small learning rate: we may converge very slowly or end up trapped in small local minima;
- too high learning rate: we may diverge instead of converge to the minimum

**Chain rule**

How to compute the gradient for complex functions.

$$
\frac{∂y}{∂x}=\frac{∂y}{∂z}∗ \frac{∂z}{∂x}
$$

### Cross Validation

Let's get some terminology straight, generally when we say **a model** we refer to a particular method for describing how some input data relates to what we are trying to predict. We don't generally refer to particular instances of that method as different models. So you might say 'I have a linear regression model' but you wouldn't call two different sets of the trained coefficients different models. At least not in the context of model selection.

So, when you do K-fold cross validation, you are testing how well your model is able to get trained by some data and then predict data it hasn't seen. We use cross validation for this because if you train using all the data you have, you have none left for testing. You could do this once, say by using 80% of the data to train and 20% to test, but what if the 20% you happened to pick to test happens to contain a bunch of points that are particularly easy (or particularly hard) to predict? We will not have come up with the best estimate possible of the models ability to learn and predict.

We want to use all of the data. So to continue the above example of an 80/20 split, we would do 5-fold cross validation by training the model 5 times on 80% of the data and testing on 20%. We ensure that each data point ends up in the 20% test set exactly once. We've therefore used every data point we have to contribute to an understanding of how well our model performs the task of learning from some data and predicting some new data.

##### But the purpose of cross-validation is not to come up with our final model. We don't use these 5 instances of our trained model to do any real prediction. For that we want to use all the data we have to come up with the best model possible. The purpose of cross-validation is model checking, not model building.

Now, say we have two models, say a linear regression model and a neural network. How can we say which model is better? We can do K-fold cross-validation and see which one proves better at predicting the test set points. But once we have used cross-validation to select the better performing model, we train that model (whether it be the linear regression or the neural network) on all the data. We don't use the actual model instances we trained during cross-validation for our final predictive model.

Note that there is a technique called bootstrap aggregation (usually shortened to 'bagging') that does in a way use model instances produced in a way similar to cross-validation to build up an ensemble model, but that is an advanced technique beyond the scope of your question here.



**Bagging** is a simple ensembling technique in which we build many *independent* predictors/models/learners and combine them using some model averaging techniques. (e.g. weighted average, majority vote or normal average)

**Boosting** is an ensemble technique in which the predictors are not made independently, but sequentially.

XGboost

aaa

### Random Forest

Final decision is based on **majority vote**, i.e. the predicted classification will be the classification foreseen by most of the trees in the model.

### Gradient Boosting

*Gradient boosting is a machine learning technique for regression and classification problems, which produces a prediction model in the form of an ensemble of weak prediction models, typically decision trees.*

it is a SEQUENTIAL (BOOSTING) technique. 

https://shirinsplayground.netlify.com/2018/11/ml_basics_gbm/

https://www.youtube.com/watch?v=jxuNLH5dXCs&t=420s

### Entropy

The definition of entropy is:
$$
E = \sum_{i=1}^{c}-p_i\log_2p_i
$$
Where $p_i$ is simply the frequentist probability of an element/class $i$ in our data.

### Regularization



The following topics will allow you to understand part of the course material better, although not strictly required:

- Eigenvalues, eigenvectors, and spectral decomposition (linear algebra)
- Lagrange multipliers (multivariable calculus).



### Linear Classifier

Training data can be graphically depicted on a (hyper)plane. **Classifiers** are **mappings** that take **feature vectors as input** and produce **labels as output**. A common kind of classifier is the **linear classifier**, which linearly divides space(the (hyper)plane where training data lies) into two. Given a point x in the space, the classifier $h$ outputs $h(x)=1$ or $h(x)=−1$, depending on where the point x exists in among the two linearly divided spaces.

We saw in the lecture above that for a linear classifier $h, h(x;θ)=sign(θ⋅x)$, i.e. the sign of the dot product of $θ$ and $x$ (note that there are multiple parameter vectors that define the same classifier; note also that equidistant points on the same side of the classifier are classified with the same "strength").

For linearly separable data, a linear classifier can perfectly separate the data,  i.e. exists at least one classifier $h(x)$ classifies all the given points correctly.

#### Perceptron Algorithm Definition

**Perceptron** $\displaystyle \left(\big \{ (x^{(i)}, y^{(i)}), i=1,...,n\big \} , T \right):$:
  initialize $θ=0$ (vector); $\theta_0 =0$ (scalar)
    for $t=1,...,T$ do
      for $i=1,...,n$ do
        if $y^{(i)}(θ⋅x^{(i)} + \theta_0)≤0$ then
        update $θ=θ+y^{(i)}x^{(i)}$
        update $θ_0=θ_0+y^{(i)}$

When a mistake is spotted, the updated values of $θ$ and $θ_0$ provide always a better prediction. To see why, let's calculate the difference between the update value of $\theta$ times the label and the updated value times the label, i.e. $y^{(i)}(θ⋅x^{(i)} + \theta_0)≤0$​.
$$
y^{(i)}((\theta +y^{(i)} x^{(i)}) \cdot x^{(i)} + \theta _0 + y^{(i)}) - y^{(i)}(\theta \cdot x^{(i)} + \theta _0) = \\ (y^{(i)})^2 \| x^{(i)}\| ^2 + (y^{(i)})^2 =\\ (y^{(i)})^2(\| x^{(i)}\| ^2 + 1)) > 0
$$
([Dot product of a vector with itself is equal to the square ot the norm](https://proofwiki.org/wiki/Dot_Product_of_Vector_with_Itself))

#### SVM (Support Vector Machines)

At the end of this lecture, you will be able to

- understand the need for maximizing the margin
- pose linear classification as an optimization problem
- understand hinge loss, margin boundaries and regularization

Objective function = loss (how examples fit) + regularization (preference for large margin boundaries)

The **decision boundary** is the set of points x which satisfy
$$
θ⋅x+θ_0=0.
$$
The **Margin Boundary** is the set of points x which satisfy
$$
θ⋅x+θ_0=±1
$$
So, the distance from the decision boundary to the margin boundary is $\frac{1}{∣∣θ∣∣}$. As we increase $∣∣θ∣∣$, $\frac{1}{∣∣θ∣∣}$ decreases. The loss for the model is:
$$
J(\theta , \theta _0) = \frac{1}{n} \sum _{i=1}^{n} \text {Loss}_ h (y^{(i)} (\theta \cdot x^{(i)} + \theta _0 )) + \frac{\lambda }{2} \mid \mid \theta \mid \mid ^2.
$$
The first part represent the **Hinge Loss**, defined as:
$$
\max \Big( 0, 1 - (y^{(i)} (\theta \cdot x^{(i)} + \theta _0 )) \Big)
$$
It spots the points not correctly classified or within the margin boundary. The second part represent the width of the margins; Through gradient descent. In other words, we will

- Start  $θ$  at an arbitrary location: $θ←θstart$
- Update  $θ$  repeatedly with  $θ←θ−η \frac{∂J(θ,θ0)}{∂θ}$  until $θ$  does not change significantly

The training objective for the Support Vector Machine (with margin loss) can be seen as optimizing a balance between the average hinge loss over the examples and a regularization term that tries to keep the parameters small (increase the margin). This balance is set by the regularization parameter $λ>0$ (*Note* - $\theta$ is be column vector, and $\hat{y}=θ^⊤x$).

### Linear Regression

Linear regression tries to estimate a predictor $f$ which is a linear function of the feature vectors. i.e. $f(x)=∑_{i=1}^dθ_ix_i+θ_0$. In carrying out this estimation there can be two kind of mistakes:

- Structural Mistakes (Non Linear relation $\to$ $f(x)$)
- Estimation mistake (too many parameters or too low data)

In any case, the **objective** is to minimize the empirical risk $R_n$ is defined as
$$
\begin{equation} R_ n(\theta ) = \frac{1}{n} \sum _{t=1}^{n} \text {Loss}(y^{(t)} - \theta \cdot x^{(t)})\,. \end{equation}
$$

If we define the loss with the **mean square** criterion, the risk becomes:
$$
\begin{equation} R_ n(\theta ) = \frac{1}{n} \sum _{t=1}^{n} \frac{(y^{(t)} - \theta \cdot x^{(t)})^2}{2}\,. \tag{1}\end{equation}
$$
In order to minimize it that, we can use two different approaches (**Learning Algorithm**)

- **Gradient descend**

  We learned in lectures that, in general, gradient descent works by moving the parameter in the opposite direction of the slope/gradient. This is accomplished in the update by subtracting the slope/gradient multiplied by the learning rate, $η$ from the current $θ$. The update is 
  $$
  θ_{new}=θ_{old}−η∗∇θ
  $$
  in order to *nudge* the parameter down the error hill; $∇θ$ is the gradient, defined as  $∇θ=−(y_t−θ∗x_t)∗x_t$. So the update becomes:
  $$
  θ_{new}=θ_{old}+η(y_t−θ∗x_t)∗x_t
  $$

- **Closed Form Solution**:  Computing the gradient of equation $(1)$ we obtain a closed form solution for $\hat{\theta}$:
  $$
  \displaystyle  \displaystyle \nabla R_ n(\theta ) = A\theta - b (=0) \quad \text {where } \,  A = \frac{1}{n} \sum _{t=1}^{n} x^{(t)} ( x^{(t)})^ T,\,  b = \frac{1}{n} \sum _{t=1}^{n} y^{(t)} x^{(t)}.
  $$



To make the optimization more robust, we want to add a **regularization** term (**Ridge Regression**). 
$$
J_{n, \lambda } (\theta , \theta _0) = \frac{1}{n} \sum _{t=1}^{n} \frac{(y^{(t)} - \theta \cdot x^{(t)}-\theta _0)^2}{2} + \frac{\lambda }{2} \left\|  \theta  \right\| ^2
$$
The new update becomes:
$$
xθ_{new}=θ_{old}-(\lambda \theta- η(y_t−θ∗x_t)∗x_t)
$$
In this way, **we are pushed to keep $\theta$ small**; i.e. we will not see huge negative and positive coefficient (especially when we have *collinearity*)

### Nonlinear Classification

We can get more and more powerful classifiers by adding linearly independent features, $x^²$, $x^3$... This functions are linearly independent, so the original coordinates always provide something above and beyond what were in the previous ones. Note that when $x$ is already multidimensional, would result in dimensions exploding, e.g. 
$$
\mathbf{x} \in \mathbb{R^5} = \phi(\mathbf{x} \in \mathbb{R^2}) = \left[ \array{x_1\\x_2\\x_1^2\\x_2^2\\\sqrt{2}x_1x_2} \right]
$$
Once we have the new feature vector we can make non-linear classification or regression in the original data making a linear classification or regression in the new feature space:

- Classification: $h(x;θ,θ_0)=\text{sign}(θ⋅ϕ(θ)+θ_0)$
- Regression: $f(x;θ,θ_0)=θ⋅ϕ(θ)+θ_0$

More feature we add (e.g. more polynomial grades we add), better we fit the data. The key question now is **when is time to stop adding features?** We can use the validation test to test which is the polynomial form that, trained on the training set, respond better in the validation set. At the extreme, you hold out each of the training example in turn in a procedure called **leave one out cross validation**. So you take a single training sample, you remove it from the training set, retrain the method, and then test how well you would predict that particular holdout example, and do that for each training example in turn. And then you average the results.

While very powerful, this mapping could dimensionally explode quickly. Let's our original $x∈R^d$. Then a feature transformation: - quadratic (order 2 polynomial): would involve $d+≈d^2$ dimensions (the original dimensions plus all the cross products) - cubic (order 3 polynomial): would involve $d+≈d^2+≈d^3$ dimensions (the exact number of terms of a feature transformation of order $p$ of a vector of d dimensions is:
$$
\sum_{i=1}^p {d+i-1 \choose i}
$$

#### Kernels: Computational Efficiency

>  **Kernel Definition**: the kernel is an inner product of an arbitrary function of its arguments. $K(x,x^′)=⟨ϕ(x),ϕ(x^′)⟩$; an **inner product** associates each pair of vectors in the space with a scalar  quantity known as the inner product of the vectors. 

The idea is that you can take inner products between high dimensional feature vectors and evaluate that inner product very cheaply. And then, we can turn our algorithms into operating only in terms of these inner products. We define the kernel function of two feature vectors (two different data pairs) applied to a a given $ϕ$ transformation as the dot product of the transformed feature vectors of the two data:
$$
k(x,x^′;ϕ)∈R^+=ϕ(x)⋅ϕ(x^′)
$$
We can hence think of the kernel function as a kind of similarity measure, how similar the $x$ example is to the $x^′$ one. Note also that being the dot product symmetric and positive, kernel functions are in turn symmetric and positive. For example let's take $x$ and $x^′$ to be two dimensional feature vectors and the feature transformation $ϕ(x)$ defined as 
$$
ϕ(x)=[x_1,x_2,x_1^2,\sqrt2x_1x_2,x_2^2]\\
ϕ(x^′) = [x_1^\prime,x_2^\prime,{x_1^\prime}^2, \sqrt2x_1^\prime x_2^\prime,{x_2^\prime}^2]
$$
This particular ϕ transformation allows to compute the kernel function very cheaply and having very few dimensions:
$$
k(x,x′;ϕ)=ϕ(x)⋅ϕ(x′)=\\= \displaystyle {x_1}{x_1^\prime } + {x_2}{x_2^\prime } + {x_1}^2{x_1^\prime }^2 + 2{x_1}{x_1^\prime }{x_2}{x_2^\prime } + {x_2}^2{x_2^\prime }^2=\\= \displaystyle \left({x_1}{x_1^\prime } + {x_2}{x_2^\prime }\right)+ \left({x_1}{x_1^\prime } + {x_2}{x_2^\prime }\right)^2=\\= \displaystyle x \cdot x^\prime + (x \cdot x^\prime )^2
$$
Note that even if the transformed feature vectors have 5 dimensions, the kernel function return a scalar. In general, for this kind of feature transformation function $ϕ$, the kernel function evaluates as 
$$
k(x,x′;ϕ)=ϕ(x)⋅ϕ(x^′)=(1+x⋅x^′)^p
$$
where p is the order of the polynomial transformation $ϕ$. However, it is only for *some* $ϕ$ for which the evaluation of the kernel function becomes so nice! As soon we can prove that a particular kernel function can be expressed as the dot product of two particular feature transformations (for those interested the *Mercer’s theorem* stated in [these notes](https://courses.cs.washington.edu/courses/cse546/16au/slides/notes10_kernels.pdf)) the kernel function is *valid* and we don't actually need to construct the transformed feature vector (the output of $ϕ$). The task will be to turn a linear method that previously operated on $ϕ(x)$, like $\text{sign}(θ⋅ϕ(x)+θ_0)$ to an inter-classifier that only depends on those inner products, that operates in terms of kernels.

#### The Kernel Perceptron Algorithm

Let's show how we can use the kernel function in place of the feature vectors in the perceptron algorithm.

Recall that the perceptron algorithm:

```python
θ = 0                		# initialisation
for t in 1:T:
	for i in 1:n>
		if yⁱ θ⋅𝛷(xⁱ) ≦ 0   # checking if sign is the same
      	θ = θ + yⁱ𝛷(xⁱ)   	 # update θ if mistake
```

Which is the final value of the parameter $θ$ resulting from such updates ? We can write it as
$$
θ^∗=∑_{n=1}^{j}α^{(j)}y^{(j)}ϕ(x^{(j)})
$$
where $α$ is the vector of number of mistakes (and hence updates) underwent for each data pair (so $α^{(j)}$ is the (scalar) number of errors occurred with the $j$-th data pair and can also be interpreted as the relative importance of the $j$-th training example to the final predictor). When we want to make a prediction of a data pair $(x^{(i)},y^{(i)})$ using the resulting parameter value $θ^∗$ (that is the "optimal" parameter the perceptron algorithm can give us), we take an inner product with that:
$$
\text{prediction}^{(i)}=θ^∗⋅ϕ(x(i))
$$
We can rewrite the above equation as :
$$
\theta^* \cdot \phi(x^{(i)}) = [\sum_{j=1}^n \alpha^{(j)} y^{(j)} \phi(x^{(j)})] \cdot \phi(x^{(i)})\\~~=  \sum_{j=1}^n [\alpha^{(j)} y^{(j)} \phi(x^{(j)}) \cdot \phi(x^{(i)})]\\~~=\sum_{j=1}^n \alpha^{(j)} y^{(j)}k(x^{(j)},x^{(i)})
$$
But this means we can now express success or errors in terms of the $α$ vector and a valid **kernel function** (cheap to compute!). An error on the data pair $(x^{(i)},Y^{(i)})$ can then be expressed as $y^{(i)} * \sum_{j=1}^n \alpha^{(j)} y^{(j)}k(x^{(j)},x^{(i)})$. We can then base our perceptron algorithm on this check, where we start with initiating the error vector $α$ to zero, and we run through the data set checking for errors and, if found, updating the corresponding error term. In practice, our endogenous variable to minimize the errors is no longer directly theta, but became the $α$ vector, that as said implicitly gives the contribution of each data pair to the $θ$ parameter. The perceptron algorithm becomes hence the **kernel perceptron algorithm**:

**Kernel Perceptron** $\displaystyle \left(\big \{ (x^{(i)}, y^{(i)}), i=1,...,n, T \big \} \right)$ 
  initialize $α_1,...,α_n$ to some values;
  for $t=1,...,T$
    	for $i=1,...,n$
      	if (*Mistake Condition Expressed* in $α_j$)  \# checking **if** prediction is right
        Update $α_j$ appropriately  **# update $α_{j}$ if mistake**

Where the mistake condition expressed in $\alpha_j$ is $y^{(i)}\sum _{j=1}^{n} \alpha _ j y^{(j)} K(x^{j},x^{i}) \leq 0$ and the update condition is $\alpha_j = \alpha_j +1$.

#### Kernel Composition Rules

Now instead of directly constructing feature vectors by adding coordinates and then taking it in the product and seeing how it collapses into a kernel, we can construct kernels directly from simpler kernels by made of the following **kernel composition rules**:

1. $K(x,x^\prime) = 1$ is a valid kernel whose feature representation is $\phi(x) = 1$;
2. Given a function $f: \mathbb{R}^d \to \mathbb{R}$ and a valid kernel function $K(x,x^\prime)$ whose feature representation is $\phi(x)$, then $\tilde K(x,x^\prime)=f(x)K(x,x^\prime)f(x^\prime)$ is also a valid kernel whose feature representation is $\tilde \phi(x) = f(x)\phi(x)$
3. Given $K_a(x,x^\prime)$ and $K_b(x,x^\prime)$ being two valid kernels whose feature representations are respectively $\phi_a(x)$ and $\phi_b(x)$, then $K(x,x^\prime)=K_a(x,x^\prime)+K_b(x,x^\prime)$ is also a valid kernel whose feature representation is $\phi(x) = \array{\phi_a(x)\\phi_b(x)}$
4. Given $K_a(x,x^\prime)$ and $K_b(x,x^\prime)$ being two valid kernels whose feature representations are respectively $\phi_a(x) \in \mathbb{R}^A$ and $\phi_b(x) \in \mathbb{R}^B$, then $K(x,x^\prime)=K_a(x,x^\prime) * K_b(x,x^\prime)$ is also a valid kernel whose feature representation is $\phi(x) = \array{\phi_{a,1}(x)* \phi_{b,1}(x)\\phi_{a,1}(x)* \phi_{b,2}(x)\ \phi_{a,1}(x)* \phi_{b,...}(x)\ \phi_{a,1}(x)* \phi_{b,B}(x)\ \phi_{a,2}(x)* \phi_{b,1}(x)\ \phi_{a,...}(x)* \phi_{b,...}(x)\ \phi_{a,A}(x)* \phi_{b,B}(x)\}$ (see [this lecture notes](https://people.cs.umass.edu/~domke/courses/sml2011/07kernels.pdf) for a proof)

Armed with these rules we can build up pretty complex kernels starting from simpler ones.

For example let's start with the identity function as $\phi$, i.e. $\phi_a(x) = x$. Such feature function results in a kernel $K(x,x^\prime;\phi_a) = K_a(x,x^\prime) = (x \cdot x^\prime)$ (this is known as the **linear kernel**). We can now add to it a squared term to form a new kernel, that by virtue of rules (3) and (4) above is still a valid kernel:

$K(x,x^\prime) = K_a(x,x^\prime) + K_a(x,x^\prime)* K_a(x,x^\prime) = (x \cdot x^\prime) + (x \cdot x^\prime)^2$

#### The Radial Basis Kernel

We can use kernel functions, and have them in term of simply, cheap-to-evaluate functions, even when the underlying feature representation would have infinite dimensions and would be hence impossible to explicitly construct.

One example is the so called **radial basis kernel**:
$$
K(x,x^\prime) = e^{-\frac{1}{2} ||x-x^\prime||^2}
$$
It [can be proved](http://pages.cs.wisc.edu/~matthewb/pages/notes/pdf/svms/RBFKernel.pdf) that suck kernel is indeed a valid kernel and its corresponding feature representation $\phi(x) \in \mathbb{R}^\infty$, i.e. involves polynomial features up to an infinite order. The radial basis kernel look like a Gaussian (without the normalization term).

[![img](https://github.com/sylvaticus/MITx_6.86x/raw/master/Unit%2002%20-%20Nonlinear%20Classification%2C%20Linear%20regression%2C%20Collaborative%20Filtering/assets/radial_basis_kernel.png)](https://github.com/sylvaticus/MITx_6.86x/raw/master/Unit 02 - Nonlinear Classification%2C Linear regression%2C Collaborative Filtering/assets/radial_basis_kernel.png)

The above picture shows the contour lines of the radial basis kernel when we keep fixed $x$ (in 2 dimensions) and we let $x^\prime$ to move away from it: the value of the kernel then reduces in a shape that in 3-d would resemble the classical bell shape of the Gaussian curve. We could even parametrize the radial basis kernel replacing the fixed $1/2$ term with a parameter $\gamma$ that would determine the width of the bell-shaped curve (the larger the value of $\gamma$ the narrower will be the bell, i.e. small values of $\gamma$ yield wide bells).

Because the feature has infinite dimensions, the radial basis kernel has infinite expressive power and can correctly classify any training test.

The linear decision boundary in the infinite dimensional space is given by the set ${x: \sum_{j=1}n \alpha^{(j)} y^{(j)} k(x^{(j)},x) = 0 }$ and corresponds to a (possibly) non-linear boundary in the original feature vector space.

The more difficult task it is, the more iterations before this kernel perception (with the radial basis kernel) will find the separating solution, but it always will in a finite number of times. This is by contrast with the "normal" perceptron algorithm that when the set is not separable would continue to run at the infinite, changing its parameters unless it is stopped at a certain arbitrary point.

#### Other non-linear classifiers

We have seen as we can have nonlinear classifiers extending to higher dimensional space and eventually using kernel methods to collapse the calculations and **operate only *implicitly* in those high dimension spaces**. There are other ways to get nonlinear classifiers.

**Decision trees** make classification operating sequentially on the various dimensions and making first a separation on the first dimension and then, in a subsequent step, on the second dimension and so on. And you can "learn" these trees incrementally. To make these decision trees more robust, **random forest classifiers**, adds two type of randomness: 1) in randomly choosing the dimension on which to operate the cut and 2) randomly selecting the single example on which operate from the data set (with replacement) and then just average the predictions obtained from these trees.

### Recommender Systems

#### Problem definition

We keep as example across the lecture the recommendation of movies. We start with a $(n,m)$ matrix $Y$ of preferences for user $a = 1,...,n$ of movie $i = 1,...,m$. While there are many ways to store preferences, we will use a real number. The goal is to base the prediction on the prior choices of the users, considering that this $Y$ matrix could be very sparse (e.g. out of 18000 films, each individual ranked very few of them!), i.e. we want to fill these "empty spaces" of the matrix. Why not to use classification/regression based on feature vectors as learned in Lectures 1? For two reasons:

1. Deciding which feature to use or extracting them from data could be hard/infeasible
2. Often we have little data about a single users preferences, while to make a recommendation based on its own previous choices we would need lot of data.

The "trick" is then to "borrow" preferences from the other users and trying to measure how much a single user is closer to the other ones in our dataset.

#### K-Nearest Neighbor (KNN) Method

The number $K$ here means, how big should be your advisory pool on how many neighbors you want to look at. And this can be one of the hyperparameters of the algorithm. We look at the $k$ closest users that did score the element I am interested to, look at their score for it, and average their score.

$$
\hat Y_{a,i} = \frac{\sum_{b \in KNN(a,i;K)} Y_{b,i}}{K}
$$
where $KNN(a,i;K)$ is the set of K users close to user a that have a score for item $i$. But how do I define this similarity? We can use any method to define similarity between vectors, like cosine similarity ($\cos \theta = \frac{x_ a\cdot x_ b}{\left| x_ a \right| \left| x_ b \right| }$) or Euclidean distance ($\left| x_ a-x_ b \right|$).

We can make the algorithm a bit more sophisticated by weighting the neighbor scores to the level of similarity rather than just take their unweighted average:

$$
\widehat{Y}_{ai} = \displaystyle \frac{\displaystyle \sum _{b \in \text {KNN}(a)} \text {sim}(a,b) Y_{bi}}{\displaystyle \sum _{b \in \text {KNN}(a)} \text {sim}(a,b)}.
$$
where $sim(a,b)$ is some similarity measure between users $a$ and $b$. There has been many improvements that has been added to this kind of algorithm, like adjusting for the different "average" score that each user gives to the items (i.e. they compare the deviations from user's averages rather than the raw score itself). Still they are very far from today's methods. The problem of KNN is that it doesn't enable us to detect the hidden structures that is there in the data, which is that users may be similar to some pool of other users in one dimension, but similar to some other set of users in a different dimension. on top, this method depends heavily on the *choice of the similarity measure*.

#### Collaborative Filtering: the Naïve Approach

Let's start with a naïve approach where we just try to apply the same method we used in regression to this problem, i.e. minimize a function $J$ made of a distance between the observed score in the matrix and the estimated one and a regularization term.

For now, we treat each individual score **independently**, and this will be the reason for which (we will see) this method **will not work**.

So, we have our (sparse) matrix $Y$ and we want to find a dense matrix $X$ that is able to replicate at best the observed points of $Y_{a,i}$ when these are available, and fill the missing ones when $Y_{a,i} = missing$. Let's first define as $D$ the set of points for which a score in $Y$ is given: $D = {(a,i) | Y_{a,i} \neq \text{missing}}$. The $J$ function then takes any possible $X$ matrix and minimize the distance between the points in the $D$ set less a regularization parameter (we keep the individual scores to zero unless we have strong belief to move them from such state):

$$
J(X;Y,\lambda) = \frac{\sum_{(a,i) \in D} (Y_{a,i} - X_{a,i})^2}{2} + \frac{\lambda}{2}\sum_{(a,i)} X_{a,i}^2
$$
To find the optimal $X_{a,i}^* ~$ that minimize the FOC $(\partial X_{a,i} / \partial Y_{a,i}) = 0$ we have to distinguish if $(a,i)$ is in $D$ or not:

- $(a,i) \in D$: $X_{a,i}^* = \frac{Y_(a,i)}{1+\lambda}$
- $(a,i) \notin D$: $X_{a,i}^* = 0$

Clearly this result doesn't make sense: for data we already know we obtain a bad estimation (as worst as we increase lambda) and for unknown scores we are left with zeros.

#### Collaborative Filtering with Matrix Factorization

What we need to do is to actually relate scores together instead of considering them independently. The idea is then to constrain the matrix $X$ to have a lower rank, as rank captures how much independence is present between the entries of the matrix.

At one extreme, constraining the matrix to be rank 1, would means that we could factorize the matrix $X$ as just the matrix product of two single vectors, one defining a sort of general sentiment about the items for each user ($u$), and the other one ($v$) representing the average sentiment for a given item, i.e. $X=uv^T$.

But representing users and items with just a single number takes us back to the KNN problem of not being able to distinguish the possible multiple groups hidden in each user or in each item. We could then decide to divide the users and/or the items in respectively $(n,2) U$ and $(2,m) V^T$ matrices and constrain our X matrix to be a product of these two matrices (hence with rank 2 in this case): $X=UV^T$

The exact numbers $K$ of vectors to use in the user/items factorization matrices (i.e. the rank of X) is then a hyperparameter that can be selected using the validation set.

#### Alternating Minimization

Using rank 1, we can adapt the $J$ function to take the two vectors $u$ and $v$ instead of the whole $X$ matrix, and our objective becomes to found their elements that minimize such function:

$$
J(\mathbf{u},\mathbf{v}; Y, \lambda) = \frac{\sum_{a,i \in D} (Y_{a,i} - u_a * v_i)^2}{2} + \frac{\lambda}{2}\sum_a^n u_a^2 + \frac{\lambda}{2}\sum_i^m v_i^2
$$
How do we minimize $J$? We can take an **iterative approach** where we start by randomly sampling values for one of the vector and minimize for the other vector (by setting the derivatives with respect on its elements equal to zero), then fix this second vector and going minimize for the first one, etc., until the value of the function $J$ doesn't move behind a certain threshold, in an alternating minimization exercise that will guarantee us to find a local minima (but not a global one!).

Note also that when we minimise for the individual component of one of the two vectors, we obtain derivatives with respect to the individual vector elements that are independent, so the first order condition can be expressed each time in terms of a single variable.

#### Numerical example

Let's consider a value of $\lambda$ equal to $1$ and the following score dataset:

$Y = \begin{bmatrix}5 & ? & 7 \\ 1 & 2 & ?\end{bmatrix}$

and let start out minimization algorithm with $v = [2,7,8]$

L becomes:

$$
J(\mathbf{u}; \mathbf{v}, Y, \lambda) = \frac{(5-2u_1)^2+(7_8u-1)^2+(1-2u_2)^2+(2-7u_2)^2}{2}+\frac{u_1^2+u_2^2}{2}+\frac{2^2+7^2+8^2}{2}
$$
From where, setting $\partial L/\partial u_1 = 0$ and $\partial L/\partial u_2 = 0$ we can retrieve the minimizing values of $(u_1,u_2)$ as 22/23 and 8/27. We can now compute $J(\mathbf{v}; \mathbf{u}, Y, \lambda)$ with these values of $u$ to retrieve the minimizing values of $v$ and so on.

Support Vector Classification with linear kernel

```python
sklearn.svm.LinearSVC
```



### Neural Networks

- Implement a ***\*feedforward neural networks\**** from scratch to perform image classification task.

- Write down the gradient of the loss function with respect to the weight parameters using ***\*back-propagation\**** algorithm and use SGD to train neural networks.

- Understand that ***\*Recurrent Neural Networks (RNNs)\**** and ***\*long short-term memory (LSTM)\**** can be applied in modeling and generating sequences.

- Implement a ***\*Convolutional neural networks (CNNs)\**** with machine learning packages.

  

<img src="C:\Users\ilsup\AppData\Roaming\Typora\typora-user-images\image-20200320184730545.png" alt="image-20200320184730545" style="zoom:40%;" />



A neural network unit computes a non-linear weighted combination of its input:
$$
\displaystyle  f(z)\quad \text {where } z= w_0 + \sum _{i=1}^ d x_ i w_ i \\	 f(z)\quad \text {where }  z = w_0+x⋅w,
$$
where $x=[x_1,…,x_d]$ and $w=[w_1,…,w_d]^T$ and where $w_i$ are numbers called **weights**, $z$ is a number and is the weighted sum of the inputs $x_i$, and $f$ is generally a non-linear function called the **activation function **.

Recall the **hyperbolic tangent function** is defined as
$$
\displaystyle \frac{e^{z}-e^{-z}}{e^{z}+e^{-z}}=1-\frac{2}{e^{2z}+1}.
$$

<img src="C:\Users\ilsup\AppData\Roaming\Typora\typora-user-images\image-20200320184821594.png" alt="image-20200320184821594" style="zoom:40%;" />



So as I understand this question you are asking which of the following are optimized during training (the others must have been optimized before that, or will remain as they were settled):

1. The dimension of the feature representation. As from the example above would be how many fi will I consider for the hidden layer
2. The weights that control the feature representation. The wij that are used for xi to create the z
3. The hyper-parameters. Learning rate, number of hidden units/layers, etc
4. The weights for the classifier. Thinking of the output layer as the classifier it would be like the θ for this classifier. Or, the weights that would accompany the fi for the last calculation (?)

#### Back Propagation

Fill it with **[Unit 3 Neural networks (2.5 weeks)](https://courses.edx.org/courses/course-v1:MITx+6.86x+1T2020/course/#block-v1:MITx+6.86x+1T2020+type@chapter+block@unit_3) [Homework 4](https://courses.edx.org/courses/course-v1:MITx+6.86x+1T2020/course/#block-v1:MITx+6.86x+1T2020+type@sequential+block@hw4) 3. Backpropagation** once the solutions are out.

#### Recurrent Neural Network (RNN)

RNN's learn the encoding into a feature vector, unlike feed-forward networks.

The standard (unit) **softmax** function ${\displaystyle \sigma :\mathbb {R} ^{K}\to \mathbb {R} ^{K}}$ is defined by the formula:
$$
{\displaystyle \sigma (\mathbf {z} )_{i}={\frac {e^{z_{i}}}{\sum _{j=1}^{K}e^{z_{j}}}}{\text{ for }}i=1,\dotsc ,K{\text{ and }}\mathbf {z} =(z_{1},\dotsc ,z_{K})\in \mathbb {R} ^{K}}
$$

#### Convolutional Neural Networks (CNN)

- Know the differences between feed-forward and Convolutional neural networks (CNNs).
- Implement the key parts in the CNNs, including ***\*convolution\**** , ***\*max pooling\**** units.
- Determine the dimension of each channel in different layers with a given CNNs.

- **CONVOLUTION**: A small squares rolls around the image and we apply the same parameters to all the patches (**shared weights**)

  > Let's suppose that we wish to classify images of $1000×1000$ dimensions. If we wish to pass the input through a feed-forward neural network with a single hidden layer made up of $1000×1000$ hidden units each of which is fully connected to the full image, we need $10^{12}$ connections / parameters ($1'000'000 × 1'000'000$); if instead we have convolutional layer with $1$ filter of shape $11×11$ instead, we need $121$ parameters.

  Remember that convolution is defines as:
  $$
  (f * g)(t) \equiv \int _{-\infty }^{+\infty } f(\tau )g(t-\tau )d\tau
  $$
  Here is a very cool [video](https://www.youtube.com/watch?v=N-zd-T17uiE) that explains it

**(MAX) POOLING**: in order to understand if an object *is* in the picture, regardless of *where* it is, we use pooling, i.e. we take the maximum value of each patch of the feature map



### Unsupervised Learning

- Understand the definition of *clustering*
- Understand *clustering cost* with different similarity measures
- Understand the *K-means* algorithm

A *partition* of a set is a grouping of the set's elements into non-empty subsets, in such a way that **every** element is included in one and only one of the subsets. In other words, $C_1,C_2,...,C_K$ is a partition of ${1,2,...,n}$ if and only if
$$
C_1 \cup C_2 \cup ... \cup C_ K = \big \{  1, 2, ..., n \big \}\\C_ i \cap C_ j = \emptyset \quad \text {for any $i \neq j$ in $\big \{ 1, ..., k\big \} $ }
$$

The cost of each cluster is defined as the sum of the cost of each cluster 

- it can be the diameter of the cluster

- the average distance between the points 

- Or the **distance from a *representative* $z$**:
  $$
  \text{Cost}(C, z) = \sum_{i \in C} \text{dist}(x^{(i)}, z)
  $$
  and $\text{dist}$ is a certain form of distance between vectors. In our case we can use **cosine similarity** (i.e. the dot product of two vectors over the product of the two norms; it is not sensitive to the magnitude of the vectors)
  $$
  {\displaystyle {\text{similarity}}=\cos(\theta )={\mathbf {A} \cdot \mathbf {B}  \over \|\mathbf {A} \|\|\mathbf {B} \|}}
  $$
  Another way of measuring vector's similarity could have also been **Euclidean squared distance**:
  $$
  \|\mathbf{A}-\mathbf {B} \|^2
  $$
  We will use this last definition and we will define the full cost of the $k$ clusters given the $k$ representatives as:
  $$
  \text{Cost}(C_1, \dotso, C_k, z^{(1)}, \dotso, z^{(k)}) = \sum_{j = 1}^{k} \sum_{i \in C} \|x^{(i)}-z^{(i)}\|^2
  $$



#### K-means Clustering

How, given the cost definition, we can find the best partition? Here is the algorithm:

1. Randomly select $k$ representatives $z^{(1)} \dots z^{(k)}$

2. Iterate
   1. Given $z_1,\dots,z_K$ assign each data point $x^{(i)}$ to the closest $z_j$, so that

   $$
   \text {Cost}(z_1, ... z_ K) = \sum _{i=1}^{n} \min _{j=1,...,K} \left\|  x^{(i)} - z_ j \right\| ^2
   $$

   2. Given $C_1, \dots , C_K$ find the best representatives $z_1,...,z_K$ i.e. find them such that

   $$
   \displaystyle z_ j=\operatorname {argmin}_{z} \sum _{i \in C_ j} \| x^{(i)} - z \| ^2.
   $$

Given some points belonging to a cluster, how do we find the representative that minimize the squared Euclidean distance? First we compute the gradient:
$$
\nabla _{z_ j}\left(\sum _{i \in \mathbb {C}_ j} \| x^{(i)} - z_ j\| ^2\right).
$$
Obtaining:
$$
\displaystyle \sum _{i \in \mathbb {C}_ j} -2(x^{(i)} - z_ j)
$$
setting it to zero we obtain: 
$$
\displaystyle z_j = \frac{\sum _{i \in C_ j} x^{(i)}}{|C_ j|}
$$

#### K-Medoids

How, given the cost definition, we can find the best partition? Here is the algorithm:

1. Randomly select $k$ representatives $\big \{  z_1, ..., z_ K \big \}  \subseteq \big \{  x_1, ..., x_ n \big \}$

2. Iterate

   1. Given $z_1,\dots,z_K$ assign each data point $x^{(i)}$ to the closest $z_j$, so that

   $$
   \text {Cost}(z_1, ... z_ K) = \sum _{i=1}^{n} \min _{j=1,...,k} \text {dist}(x^{(i)}, z_ j)
   $$

   2. Given $C_1, \dots , C_K$ find the best representatives $z_1,...,z_K$ i.e. find them such that

   $$
   \sum _{x^{(i)} \in C_ j} \text {dist}(x^{(i)}, z_ j)
   $$



**K-means**: In step 2.1, we go through each of the $n$ $x_i$, and iterate through each of the $k$ $z_j$'s for each $x_i$ (to find the closest $z_j$). This iteration is $\mathcal{O}(nK)$. And because each $x_i$ has length $d$, the total iteration is $\mathcal{O}(ndK)$. Step 2.2 is similar, so the order of complexity remains the same.

**K-medoids**: Note that step 2.1 of the K-Medoids is the same as that of K-Means, so the time complexity is $\mathcal{O}(ndK)$. Note that step 2.2 of K-Medoids has an additional loop of iterating through the $n$ points $z_j \in {x_1,...,x_n}$ which takes $\mathcal{O}(n)$. Thus step 2.2 takes $\mathcal{O}(n^2dK)$.





- Understand the ***\*limitations\**** of the ***\*K-Means\**** algorithm
- Understand how ***\*K-Medoids\**** algorithm is different from the K-Means algorithm
- Understand the ***\*computational complexity\**** of the K-Means and the K-Medoids algorithms
- Understand the importance of choosing the right number of clusters
- Understand elements that can be supervised in unsupervised learning

