# A SIMPLE AND EFFICIENT NUMERICAL METHOD FOR PRICING DISCRETELY MONITORED EARLY-EXERCISE OPTIONS

MIN HUANG AND GUO LUO

ABSTRACT. We present a simple, fast, and accurate method for pricing a variety of discretely monitored options in the Black-Scholes framework, including autocollable structured products, single and double barrier options, and Bermudan options. The method is based on a quadrature technique, and it employs only elementary calculations and a fixed one-dimensional uniform grid. The convergence rate is  $O(1 / N^4)$  and the complexity is  $O(MN\log N)$ , where  $N$  is the number of grid points and  $M$  is the number of observation dates.

# QUANTARK IMPLEMENTATION NOTES

- The QuadratureCore implements the recursion in Eq. (3.5) using observation dates as time steps (no implicit uniform resampling).
- Product-specific adapters map barrier and one-touch contracts into boundary levels (`K^-`, `K^+`) and linear payoff coefficients (`a^-`, `b^-`, `a^+`, `b^+`, `a_M`, `b_M`).

# 1. INTRODUCTION

Exotic options are commonly traded throughout the world. Many popular exotic options are path-dependent and have early-exercise features. These options can often be priced using analytical formulas if they are continuously monitored (e.g. barrier options). In practice, however, most path-dependent exotic options are discretely monitored [4], in which case they need to be priced using numerical techniques. Due to the complicated structures of these options, traditional pricing models based on Monte-Carlo simulations and finite difference methods are often too time-consuming to be useful in practical situations. More recent pricing methods based on advanced mathematical techniques, on the other hand, tend to be more efficient (e.g. [14, 12, 10, 11]), but for many financial institutions, these methods are often too difficult to understand and to properly implement. To strike a balance between model performance and practical utility, we propose a new quadrature-based method that is much faster and more accurate than the traditional Monte-Carlo and PDE methods, yet at the same time is easy to understand and to implement. We will first give a brief review of the types of products considered, as well as the quadrature-based pricing model which is the foundation of our work. Then we will explain our method and provide numerical examples.

1.1. Autocollable Structured Products. Autocollable structured products belong to a class of exotic options with early-exercise features. Many different types of autocollable products have been created and traded in financial markets, and they have become increasingly popular in recent years. We refer to the Appendix of [9] for a description of the main features of various autocollable products.

We will consider a very common autocollable product with discrete observation dates. At each observation date there is a pre-specified barrier level. If the price of

the underlying asset is greater (less) than or equal to the barrier level (depending on the terms of the product), the option is exercised and a pre-specified fixed-rate return is paid. If the asset price is below (above) the barriers at all observation dates, the option is never exercised and the investor receives a negative return at maturity. In addition, autocollable products may have a knock-in feature. In this case, if the option is never exercised, the negative return the investor receives depends on whether the asset price at maturity reaches a pre-specified knock-in level. While the value of a continuously monitored autocollable product has a simple closed-form solution, the value of a discretely monitored autocollable product cannot be calculated easily. In the discrete case, there exist analytical solutions in terms of multiple integrals, cf. [25, 26, 9]. The numerical calculation of these integrals, however, can become prohibitive if the number of observation dates exceeds five. In practice, discretely monitored autocollable products are commonly priced using Monte-Carlo simulations. This method is straightforward, but convergence is usually slow and acceleration techniques such as variance reduction are often needed (cf. [2, 13]). Another popular method for pricing discretely monitored autocollable products is to solve the governing Black-Scholes partial differential equation (PDE) using finite difference method (cf. [9]). Assuming a second-order central difference approximation in space, the overall convergence rate of a typical finite difference based pricing method is  $O((\delta x)^2 + \delta t)$  if the explicit forward Euler method is used in time, and  $O((\delta x)^2 + (\delta t)^2)$  if the implicit Crank-Nicolson method is used. Since two-dimensional grids are needed for finite difference methods, computational complexities are at least of order  $1 / (\delta x\delta t)$ . In addition, since the payoffs of autocollable products are discontinuous (in asset price), additional care (such as smoothing of payoff functions) must be taken to ensure the accuracy of any finite difference approximations.

Remark 1.1. Some customizable products may have payoffs at maturity that are of the same type as that of European vanilla options. These products can be effectively viewed as combinations of customizable products and barrier options (see below).

1.2. Discrete Barrier Options. Barrier options are among the most popular types of exotic options. A barrier option may be activated (knock-in option) or deactivated (knock-out option) when the price of the underlying asset crosses certain barrier levels. Barrier options may be discretely or continuously monitored. A single barrier option has one barrier at each observation date, while a double barrier option has two barriers at each observation date. The final payoff of a barrier option (if it is active at maturity) may be of the same type as that of a vanilla option or that of a digital option. A special type of barrier option has a constant amount of cash as the final payoff, and such an option is called a touch option. Most barrier options have time-independent barrier levels, but options with time-dependent barrier levels have also been studied [7].

Similar to the case of autocollable structured products, there are closed-form solutions for discrete barrier options in terms of multiple integrals [23], but such solutions are often difficult to evaluate directly. In practice, discrete barrier options can be priced using Monte-Carlo simulations or standard binomial tree methods, but these methods are usually slow [5]. Other methods that have been proposed to price discrete barrier options include continuity correction approximations [4, 27], Wiener-Hopf methods [14], adaptive mesh methods [1], Hilbert transform methods [12], finite element methods [16], Fourier-cosine series expansion methods [10],

and quadrature methods [3, 6]. These methods, while useful in certain contexts, have not been as widely used as the traditional Monte-Carlo and finite difference methods, usually due to their complexity.

1.3. Bermudan Options. Bermudan options are discrete versions of American options. A Bermudan option can be exercised at any of the prescribed observation dates, and the payoff is of the same type as that of a vanilla option. Similar to discrete barrier options, Bermudan options can be priced using Monte-Carlo simulations [17], Hilbert transform methods [11], Fourier-cosine series expansion methods [10], and quadrature methods [22, 21, 6].

1.4. Overview of the Quadrature Method. Among the various methods proposed to price discretely monitored options, the quadrature method is particularly appealing because of its high efficiency and accuracy. The method has been applied to discrete barrier options and Bermudan options [3, 22, 21, 6]. The main idea is to solve for option values at each observation date via backward induction in time. The risk-neutral valuation formula is expressed as a single integral, which is then evaluated numerically to produce the option price. Specifically, let  $V$  denote the value of the option,  $S$  the value of the underlying asset,  $r$  the risk-neutral interest rate,  $t_1, \ldots, t_M$  the observation dates, and  $\mathbb{E}$  the risk-neutral expectation. If the underlying asset  $S$  does not trigger an early exercise at  $t_m$ , we have

$$
\begin{array}{l} V (t _ {m}, S) = e ^ {- r (t _ {m + 1} - t _ {m})} \mathbb {E} \left[ V (t _ {m + 1}, \cdot) | S \right] \\ = e ^ {- r (t _ {m + 1} - t _ {m})} \int_ {0} ^ {\infty} V (t _ {m + 1}, y) f (y | S) d y, \\ \end{array}
$$

where  $f(y|S)$  is a probability density function whose form depends on the model of the underlying asset. If  $S$  triggers an early exercise at  $t_m$ , on the other hand, the option price  $V(t_m, S)$  would be equal to a prescribed value. The integral above can be calculated using FFT [21, 22] or Fast Gauss Transform [6]. Since, however,  $V(t_{m+1}, y)$  is discontinuous (in  $y$ ) for autocollable products and barrier options, and non-differentiable (in  $y$ ) for Bermudan options, care must be taken to ensure the accuracy of the numerical evaluation of the integral. While several shifted or nonuniform grids have been designed in previous studies to address this difficulty [21, 6], the problem becomes particularly challenging when multiple discontinuities are present at each observation point, for instance in the case of double barrier options with time-dependent barrier levels.

1.5. Motivation of Our Work. Although a good number of advanced techniques have been proposed to improve pricing models' accuracy and efficiency, in most practical situations, simple methods that are more cost effective are usually preferred over their more sophisticated counterparts. The reason for this is twofold. First of all, when data quality is not high enough, sophisticated models are not necessarily beneficial. For instance, interest rates and volatilities are crucial components of nearly every pricing model, but they need to be estimated from available market data. If the estimated parameters contain large errors, which is not uncommon in products sold in emerging markets, any advantages gained from the use of sophisticated models may be (more than) offset by these errors, making the simpler models more attractive. Secondly, implementations of pricing models usually involve staff members from multiple business departments, and the resulting products often need active maintenance and updates. As a result, models that are

too complicated in nature may hinder effective business communications, which increases maintenance costs and operational risks. In view of these considerations, it is not difficult to see why traditional Monte-Carlo and PDE methods are still among the most popular methods in the valuation of discretely monitored options, even though their computational costs are already high enough to adversely impact their applicability in business.

In view of the practical concerns mentioned above, we propose a new quadrature method to price the aforementioned discretely monitored options in the Black-Scholes framework. The convergence rate of our method is  $O(1 / N^4)$  and the complexity is  $O(MN\log N)$ , where  $N$  is the number of grid points and  $M$  is the number of observation dates. The performance of our method is on par with previous quadrature-based methods such as the CONV method [21], but it is more straightforward, and is better suited for products with multiple discontinuities. Our method differs from other quadrature methods mainly in three aspects. First, we work with probability density functions directly instead of using characteristic functions or Toeplitz matrices. Secondly, we use only a fixed one-dimensional uniform grid to compute all integrals. Thirdly, we utilize explicit Black-Scholes formulas to improve the accuracy of the calculations. Due to these novel modifications, our method is very easy to implement, and is capable of handling sophisticated products such as double-barrier options with time-dependent barrier levels.

1.6. Organization of the Paper. The rest of the paper is organized as follows. Section 2 specifies the class of (discrete) option pricing problems that our quadrature method is intended to solve, and Section 3 presents the main recursion formula for our method. After detailing the implementation of our method in Section 4-5, we summarize the algorithm in Section 6 and then present numerical examples in Section 7. The Appendix collects a few useful theoretical results, which lay the foundation of a class of (discrete) option pricing algorithms (including the one described here) but which, to our best knowledge, do not seem to have been rigorously proved or even properly formulated in the literature. We supply the proofs here in the hope that they would be useful to interested readers.

# 2. BASIC ASSUMPTIONS

We assume that the price of the underlying asset  $S(t)$  satisfies the following stochastic differential equation in the risk-neutral measure:

$$
d S (t) = [ r (t) - q (t) ] S (t) d t + \sigma (t) S (t) d W (t), \tag {2.1}
$$

where  $r(t)$  is the risk-neutral interest rate,  $q(t)$  is the yield rate,  $\sigma(t)$  is the volatility, and  $W(t)$  is the Wiener process.

In practice, interest rates are always time dependent. Yield rates for FX products are simply foreign interest rates, and for other types of products they may be implied from futures prices. Thus yield rates are usually time dependent as well. Implied volatilities are time dependent, whereas historical volatilities can often be taken as constant.

The solution to (2.1) is

$$
S (t) = S \left(t _ {0}\right) \exp \left\{\int_ {t _ {0}} ^ {t} [ r (s) - q (s) - \frac {1}{2} \sigma^ {2} (s) ] d s + \int_ {t _ {0}} ^ {t} \sigma (s) d W (s) \right\}, \tag {2.2}
$$

where  $t_0$  is the present date. We consider a discretely monitored option with observation dates  $t_1, \ldots, t_M$ , where the last observation date  $t_M$  is the maturity date. It follows from (2.2) that each  $S(t_m)$  for  $1 \leq m \leq M$  has the lognormal distribution

$$
(2. 3) S (t _ {m}) \sim S (t _ {0}) \exp \Biggl \{\int_ {t _ {0}} ^ {t _ {m}} \bigl [ r (s) - q (s) - \frac {1}{2} \sigma^ {2} (s) \bigr ] d s + \left(\int_ {t _ {0}} ^ {t _ {m}} \sigma^ {2} (s) d s\right) ^ {1 / 2} Z \Biggr \},
$$

where  $Z$  denotes the standard normal distribution. Now define

$$
r _ {m} = \int_ {t _ {m - 1}} ^ {t _ {m}} \frac {r (s)}{\Delta t _ {m}} d s, q _ {m} = \int_ {t _ {m - 1}} ^ {t _ {m}} \frac {q (s)}{\Delta t _ {m}} d s, \sigma_ {m} ^ {2} = \int_ {t _ {m - 1}} ^ {t _ {m}} \frac {\sigma^ {2} (s)}{\Delta t _ {m}} d s,
$$

for  $1 \leq m \leq M$  where  $\Delta t_m = t_m - t_{m-1}$ , and define piecewise constant functions

$$
\tilde {r} (t) = r _ {m}, \quad \tilde {q} (t) = q _ {m}, \quad \text {a n d} \quad \tilde {\sigma} (t) = \sigma_ {m}, \qquad \text {f o r} \qquad t _ {m - 1} <   t \leq t _ {m}.
$$

For the process

$$
d \tilde {S} (t) = \left[ \tilde {r} (t) - \tilde {q} (t) \right] \tilde {S} (t) d t + \tilde {\sigma} (t) \tilde {S} (t) d W (t),
$$

since

$$
\int_ {t _ {0}} ^ {t _ {m}} r (s) d s = \sum_ {n = 1} ^ {m} \int_ {t _ {n - 1}} ^ {t _ {n}} r (s) d s = \sum_ {n = 1} ^ {m} r _ {n} \Delta t _ {n} = \int_ {t _ {0}} ^ {t _ {m}} \tilde {r} (s) d s,
$$

$$
\int_ {t _ {0}} ^ {t _ {m}} q (s) d s = \sum_ {n = 1} ^ {m} \int_ {t _ {n - 1}} ^ {t _ {n}} q (s) d s = \sum_ {n = 1} ^ {m} q _ {n} \Delta t _ {n} = \int_ {t _ {0}} ^ {t _ {m}} \tilde {q} (s) d s,
$$

and

$$
\int_ {t _ {0}} ^ {t _ {m}} \sigma^ {2} (s) d s = \sum_ {n = 1} ^ {m} \int_ {t _ {n - 1}} ^ {t _ {n}} \sigma^ {2} (s) d s = \sum_ {n = 1} ^ {m} \sigma_ {n} ^ {2} \Delta t _ {n} = \int_ {t _ {0}} ^ {t _ {m}} \tilde {\sigma} ^ {2} (s) d s,
$$

it follows that  $\tilde{S}(t_m)$  has the same distribution as  $S(t_m)$  in (2.3) for each  $m$ . Since the value of the option depends only on probability distributions of the asset price at observation dates, the option value remains the same if we replace the process  $S$  by the process  $\tilde{S}$ . In other words, we may safely assume that  $r(t)$ ,  $q(t)$ , and  $\sigma(t)$  are piecewise constant functions. Thus in what follows, we shall assume

$$
r (t) = r _ {m}, \quad q (t) = q _ {m}, \quad \text {a n d} \quad \sigma (t) = \sigma_ {m}, \qquad \text {f o r} \qquad t _ {m - 1} <   t \leq t _ {m}.
$$

Consider now a general class of discretely monitored options with barriers. Since the sum of a knock-in barrier option and a knock-out barrier option with the same observation dates and barrier levels is a vanilla option (or a digital option if the barrier options are digital), to study the pricing of these discretely monitored options, it suffices to consider a knock-out barrier option which ceases to exist when barrier levels are crossed. To this end, assume that

(A) The option has two strike prices  $K_{m}^{-}, K_{m}^{+} \in [0, \infty]$ , with  $K_{m}^{-} \leq K_{m}^{+}$ , at each observation date  $t_{m}$ ,  $m = 1, 2, \ldots, M$ .  
(B) The option is exercised if  $S \leq K_{m}^{-}$  or  $S \geq K_{m}^{+}$  at some  $t_{m}$ , and the payoffs are given by  $a_{m}^{-}S + b_{m}^{-}$  (if  $S \leq K_{m}^{-}$ ) and  $a_{m}^{+}S + b_{m}^{+}$  (if  $S \geq K_{m}^{+}$ ), respectively, for some  $a_{m}^{\pm}$ ,  $b_{m}^{\pm} \in \mathbb{R}$ .  
(C) The final payoff at maturity is

$$
V (t _ {M}, S) = a _ {M} S + b _ {M}, \qquad \text {f o r} \qquad K _ {M} ^ {-} <   S <   K _ {M} ^ {+}.
$$

These assumptions are general enough to cover a wide class of discretely monitored options, such as the ones mentioned in the introduction. For instance, common up-and-out autocollable products would have

$$
\begin{array}{l} 1 \leq m \leq M: 0 <   K _ {m} ^ {+} <   \infty , K _ {m} ^ {-} = 0, a _ {m} ^ {+} = 0, b _ {m} ^ {+} > 0; \\ m = M: a _ {M} = 0, b _ {M} <   0. \\ \end{array}
$$

Down-and-out put barrier options would have

$$
\begin{array}{l} 1 \leq m \leq M - 1: K _ {m} ^ {+} = \infty , 0 <   K _ {m} ^ {-} <   \infty , a _ {m} ^ {-} = 0, b _ {m} ^ {-} = 0; \\ m = M: K _ {M} ^ {-} = 0, 0 <   K _ {M} ^ {+} <   \infty , a _ {M} = - 1, b _ {M} = K _ {M} ^ {+}, \\ a _ {M} ^ {+} = b _ {M} ^ {+} = 0. \\ \end{array}
$$

Double barrier knock-out call options would have

$$
\begin{array}{l} 1 \leq m \leq M - 1: \quad 0 <   K _ {m} ^ {\pm} <   \infty , a _ {m} ^ {\pm} = 0, b _ {m} ^ {\pm} = 0; \\ m = M: \quad K _ {M} ^ {+} = \infty , 0 <   K _ {M} ^ {-} <   \infty , a _ {M} = 1, b _ {M} = - K _ {M} ^ {-}, \\ a _ {M} ^ {-} = b _ {M} ^ {-} = 0. \\ \end{array}
$$

Bermudan put options with strike  $K$  would have

$$
1 \leq m \leq M: K _ {m} ^ {-}: \text {t h e u n i q u e s o l u t i o n o f} K - K _ {m} ^ {-} = V (t _ {m}, K _ {m} ^ {-}),
$$

$$
K _ {m} ^ {+} = \infty , a _ {m} ^ {-} = - 1, b _ {m} ^ {-} = K;
$$

$$
m = M: \quad a _ {M} = 0, b _ {M} = 0.
$$

We will give a proof of the uniqueness of  $K_{m}^{\pm}$  for Bermudan options in the Appendix. To summarize, our basic assumptions are

(1) The underlying asset price  $S$  follows a geometric Brownian motion with piecewise constant interest rates, yield rates, and volatilities.  
(2) There are finitely many observation points, and two exercise levels (possibly  $\infty$ ) at each observation point. If  $S$  is above the upper exercise level or below the lower exercise level at any observation point, the option is exercised and the payoff is a linear function in  $S$ .  
(3) At maturity, if  $S$  is between the two exercise levels, a payoff is incurred which is also a linear function in  $S$ .

# 3. OUTLINE OF THE METHOD

Let  $V(t, S)$  denote the value of the option (as a function of asset price  $S$ ) at any time  $t$ , and let

$$
V _ {m} (S) = V \left(t _ {m}, S\right), \qquad m = 0, 1, \dots , M,
$$

denote the value of the option at the observation dates. Our goal is to find  $V_{0}(S(t_{0}))$ , and our strategy is to use backward induction in time. Since  $V_{M}(S)$  is piecewise linear in  $S$ ,  $V_{M-1}(S)$  has a simple explicit expression. For each  $m = M-1, \ldots, 1$ , we write  $V_{m-1}(S)$  as the risk-neutral expectation of  $V_{m}(S)$  for  $K_{m-1}^{-} < S < K_{m-1}^{+}$ , and as  $a_{m-1}^{\pm} S + b_{m-1}^{\pm}$  otherwise. The expectation is given by an explicit integral and is calculated numerically. The core of the quadrature method is the calculation of  $M-1$  expectation integrals step-by-step. Let

$$
\tau_ {m} = \frac {1}{2} \sigma_ {m} ^ {2} \Delta t _ {m} = \frac {1}{2} \sigma_ {m} ^ {2} (t _ {m} - t _ {m - 1}).
$$

For each  $1 \leq m \leq M - 1$ , note that  $S(t_m)$  has a lognormal distribution as in (2.3). The relevant probability density functions are known to be [19]

$$
(3. 1) \rho_ {m} (y, S) = \frac {1}{2 \sqrt {\pi \tau_ {m}} y} \exp \biggl \{- \frac {1}{4 \tau_ {m}} \Bigl (\log \frac {y}{S} - \frac {2}{\sigma_ {m} ^ {2}} \bigl [ r _ {m} - q _ {m} - \frac {1}{2} \sigma_ {m} ^ {2} \bigr ] \tau_ {m} \Bigr) ^ {2} \biggr \}.
$$

For simplicity of notations we define  $K_0^+ = \infty$  and  $K_0^- = 0$ . By the fundamental theorems of asset pricing, we have the risk-neutral pricing formula [24]:

(3.2)

$$
\begin{array}{l} V _ {m - 1} (S) = e ^ {- 2 r _ {m} \tau_ {m} / \sigma_ {m} ^ {2}} \mathbb {E} \big [ V _ {m} (\cdot) | S \big ] = e ^ {- 2 r _ {m} \tau_ {m} / \sigma_ {m} ^ {2}} \int_ {0} ^ {\infty} V _ {m} (y) \rho_ {m} (y, S) d y \\ = \frac {e ^ {- 2 r _ {m} \tau_ {m} / \sigma_ {m} ^ {2}}}{2 \sqrt {\pi \tau_ {m}}} \int_ {0} ^ {\infty} \frac {1}{y} V _ {m} (y) \exp \biggl \{- \frac {1}{4 \tau_ {m}} \Bigl (\log \frac {y}{S} - \frac {2}{\sigma_ {m} ^ {2}} \bigl [ r _ {m} - q _ {m} - \frac {1}{2} \sigma_ {m} ^ {2} \bigr ] \tau_ {m} \Bigr) ^ {2} \biggr \} d y, \\ \end{array}
$$

for  $K_{m-1}^- < S < K_{m-1}^+$  and  $1 \leq m \leq M-1$ . By Assumption (B) from Section 2, we also have

$$
V _ {m - 1} (S) = \left\{ \begin{array}{l l} a _ {m - 1} ^ {-} S + b _ {m - 1} ^ {-}, & S \leq K _ {m - 1} ^ {-} \\ a _ {m - 1} ^ {+} S + b _ {m - 1} ^ {+}, & S \geq K _ {m - 1} ^ {+} \end{array} . \right. \tag {3.3}
$$

To further study the formulas (3.2) and (3.3), we first recall some classical results on the pricing of binary options.

Lemma 3.1. Let  $K > 0$ , and let  $\chi_A$  denote the characteristic function of a set  $A$ . Consider an option with no early-exercise features.

(1) If the option has payoff  $\hat{V}_m(y) = \chi_{[K,\infty)}y$ , then  $\hat{V}_{m - 1}(S) = V_m^a (S,K,1)$ .  
(2) If  $\hat{V}_m(y) = \chi_{(0,K]}y$ , then  $\hat{V}_{m - 1}(S) = V_m^a (S,K, - 1)$ .  
(3) If  $\hat{V}_m(y) = \chi_{[K,\infty)}$ , then  $\hat{V}_{m - 1}(S) = V_m^b (S,K,1)$ .  
(4) If  $\hat{V}_m(y) = \chi_{(0,K]}$ , then  $\hat{V}_{m - 1}(S) = V_m^b (S,K, - 1)$ .

The functions  $V_{m}^{a}$  and  $V_{m}^{b}$  are defined as

$$
V _ {m} ^ {a} (S, K, \epsilon) = e ^ {- 2 q _ {m} \tau_ {m} / \sigma_ {m} ^ {2}} S N (\epsilon d _ {1}), V _ {m} ^ {b} (S, K, \epsilon) = e ^ {- 2 r _ {m} \tau_ {m} / \sigma_ {m} ^ {2}} N (\epsilon d _ {2}),
$$

where  $N$  is the cumulative normal distribution function, and

$$
d _ {1} = \frac {1}{\sqrt {2 \tau_ {m}}} \Big (\log \frac {S}{K} + \frac {2}{\sigma_ {m} ^ {2}} \big [ r _ {m} - q _ {m} + \frac {1}{2} \sigma_ {m} ^ {2} \big ] \tau_ {m} \Big), \qquad d _ {2} = d _ {1} - \sqrt {2 \tau_ {m}}.
$$

Proof. By definition  $V_{m}^{a}$  is the value of an asset-or-nothing option, and  $V_{m}^{b}$  is the value of a cash-or-nothing option. The valuation formulas are just standard results for binary options [18].

Remark 3.2. The standard Black-Scholes formulas in Lemma 3.1 ignore the possible effects of volatility smiles. If such effects need to be taken into account, one may amend the definitions of  $V_{m}^{a}$  and  $V_{m}^{b}$  (as given in the Lemma) by incorporating suitable vega-induced correction terms [15]. For instance, the value of a cash-or-nothing call option in the presence of volatility smiles would become

$$
V _ {\mathrm {s m i l e}} = V _ {\mathrm {n o s m i l e}} - \frac {\partial V _ {\mathrm {v a n i l l a}}}{\partial \sigma} \frac {\partial \sigma}{\partial K}.
$$

We can use Lemma 3.1 to obtain an explicit formula for the value of  $V_{M-1}(S)$ . By Assumption (B)-(C) from Section 2, we have

$$
V _ {M} (S) = \left\{ \begin{array}{l l} a _ {M} ^ {-} S + b _ {M} ^ {-}, & S \leq K _ {M} ^ {-} \\ a _ {M} S + b _ {M}, & K _ {M} ^ {-} <   S <   K _ {M} ^ {+}. \\ a _ {M} ^ {+} S + b _ {M} ^ {+}, & S \geq K _ {M} ^ {+} \end{array} \right. \tag {3.4}
$$

Without loss of generality we may assume  $0 < K_M^{\pm} < \infty$ , since otherwise we may choose some arbitrary  $0 < K_M^{\pm} < \infty$  and set  $a_{M}^{\pm} = a_{M}$ ,  $b_{M}^{\pm} = b_{M}$ .

Proposition 3.3. The value of the option at  $t_{M - 1}$  is given by

$$
\begin{array}{l} \tilde {V} _ {M - 1} (S) = a _ {M} ^ {-} V _ {M} ^ {a} \left(S, K _ {M} ^ {-}, - 1\right) + b _ {M} ^ {-} V _ {M} ^ {b} \left(S, K _ {M} ^ {-}, - 1\right) \\ + a _ {M} \left[ V _ {M} ^ {a} (S, K _ {M} ^ {-}, 1) - V _ {M} ^ {a} (S, K _ {M} ^ {+}, 1) \right] \\ + b _ {M} \left[ V _ {m} ^ {b} (S, K _ {M} ^ {-}, 1) - V _ {M} ^ {b} (S, K _ {M} ^ {+}, 1) \right] \\ + a _ {M} ^ {+} V _ {M} ^ {a} (S, K _ {M} ^ {+}, 1) + b _ {M} ^ {+} V _ {M} ^ {b} (S, K _ {M} ^ {+}, 1), \\ \end{array}
$$

for  $K_{M - 1}^{-} < S < K_{M - 1}^{+}$ .

Proof. Clearly, the option from  $t_{M-1}$  to  $t_M$  is equivalent to a linear combination of binary options consisting of two put options with strike  $K_M^-$ , two call options with strike  $K_M^-$  and four call options with strike  $K_M^+$ . The result then follows from Lemma 3.1.

With the aid of (3.2) and Proposition 3.3, we may write the main recursion of our quadrature method as follows.

Proposition 3.4. Let  $\tilde{V}_M = V_M$  be defined in (3.4), and  $\tilde{V}_{m - 1}$  be given by the following recursion formula:

$$
\begin{array}{l} (3. 5) \tilde {V} _ {m - 1} (S) = e ^ {- 2 r _ {m} \tau_ {m} / \sigma_ {m} ^ {2}} \int_ {K _ {m} ^ {-}} ^ {K _ {m} ^ {+}} \tilde {V} _ {m} (y) \rho_ {m} (y, S) d y + a _ {m} ^ {+} V _ {m} ^ {a} (S, K _ {m} ^ {+}, 1) \\ + b _ {m} ^ {+} V _ {m} ^ {b} (S, K _ {m} ^ {+}, 1) + a _ {m} ^ {-} V _ {m} ^ {a} (S, K _ {m} ^ {-}, - 1) + b _ {m} ^ {-} V _ {m} ^ {b} (S, K _ {m} ^ {-}, - 1), \\ \end{array}
$$

for  $1 \leq m \leq M$ . Then we have  $\tilde{V}_m(S) = V_m(S)$  for  $K_{m}^{-} < S < K_{m}^{+}$  and  $0 \leq m \leq M$ . In particular,  $\tilde{V}_0(S(t_0)) = V_0(S(t_0))$ .

Proof. This is merely the classical recursion formula for quadrature methods specialized to the Black-Scholes model. To prove the formula, we only need to show

$$
V _ {m} (S) = \tilde {V} _ {m} (S) \chi_ {\left(K _ {m} ^ {-}, K _ {m} ^ {+}\right)} + \left(a _ {m} ^ {+} S + b _ {m} ^ {+}\right) \chi_ {\left[ K _ {m} ^ {+}, \infty\right)} + \left(a _ {m} ^ {-} S + b _ {m} ^ {-}\right) \chi_ {\left(0, K _ {m} ^ {-} \right]}, \tag {3.6}
$$

for all  $0 \leq m \leq M$ . By assumption (3.6) is true for  $m = M$ . Assume now (3.6) holds for some  $1 \leq m \leq M$ . Substituting the equation into (3.2), applying Lemma 3.1, comparing the result with (3.5) and using (3.3), we observe that (3.6) holds for  $m - 1$ . The result then follows from induction.

Remark 3.5. Recursion formula (3.5) lies at the heart of our quadrature method and distinguishes our method from other quadrature methods, which are primarily based on (3.2) or one of its many variants. The significance of the formula (3.5)

lies in the fact that it makes explicit use of Black-Scholes formulas to separate the expectation integral  $\mathbb{E}\big[V_m(\cdot)|S\big]$  into a "quadrature part"

$$
F _ {m - 1} (S) = e ^ {- 2 r _ {m} \tau_ {m} / \sigma_ {m} ^ {2}} \int_ {K _ {m} ^ {-}} ^ {K _ {m} ^ {+}} \tilde {V} _ {m} (y) \rho_ {m} (y, S) d y,
$$

and an "early-exercise part"

$$
\begin{array}{l} E _ {m - 1} (S) = a _ {m} ^ {+} V _ {m} ^ {a} \left(S, K _ {m} ^ {+}, 1\right) + b _ {m} ^ {+} V _ {m} ^ {b} \left(S, K _ {m} ^ {+}, 1\right) \\ + a _ {m} ^ {-} V _ {m} ^ {a} (S, K _ {m} ^ {-}, - 1) + b _ {m} ^ {-} V _ {m} ^ {b} (S, K _ {m} ^ {-}, - 1). \\ \end{array}
$$

Since the function  $\tilde{V}_m(S)$  is smooth for  $S \in (K_m^-, K_m^+)$  (in fact for all  $S \in (0, \infty)$ , as we will show below), the integral  $F_{m-1}(S)$  can be evaluated accurately and efficiently using a high-order quadrature method such as Simpson's rule. In contrast, the integrand  $V_m(S)$  in the original recursion formula (3.2) is discontinuous on  $(0, \infty)$  (in either  $V_m$  itself or in its first derivative); this makes the accurate evaluation of the expectation integral a difficult and challenging task. Although (3.5) applies specifically to the Black-Scholes model, the same idea can be used for other asset price models, as long as a suitable analytical formula (exact or approximate) can be found for the probability density function  $\rho_m(y, S)$  and early-exercise part  $E_{m-1}(S)$ .

With  $\tilde{V}_{M - 1}(S)$  given in Proposition 3.3, Proposition 3.4 implies that we may apply (3.5) successively to obtain  $\tilde{V}_{M - 2}(S),\tilde{V}_{M - 3}(S),\ldots ,\tilde{V}_0(S)$ . The value of the option is equal to  $\tilde{V}_0(S(t_0))$ .

# 4. DETAILS OF IMPLEMENTATION

In (3.5),  $\tilde{V}_{m - 1}(S)$  is written as a sum of explicit functions and an integral, namely

$$
\begin{array}{l} \tilde {V} _ {m - 1} (S) = a _ {m} ^ {+} V _ {m} ^ {a} \left(S, K _ {m} ^ {+}, 1\right) + b _ {m} ^ {+} V _ {m} ^ {b} \left(S, K _ {m} ^ {+}, 1\right) \tag {4.1} \\ + a _ {m} ^ {-} V _ {m} ^ {a} (S, K _ {m} ^ {-}, - 1) + b _ {m} ^ {-} V _ {m} ^ {b} (S, K _ {m} ^ {-}, - 1) + F _ {m - 1} (S), \\ \end{array}
$$

where

$$
F _ {m - 1} (S) = e ^ {- 2 r _ {m} \tau_ {m} / \sigma_ {m} ^ {2}} \int_ {K _ {m} ^ {-}} ^ {K _ {m} ^ {+}} \tilde {V} _ {m} (y) \rho_ {m} (y, S) d y.
$$

We may truncate the integral by replacing its upper and lower bounds by

$$
L _ {m} ^ {+} = \min \bigl \{K _ {m} ^ {+}, S (t _ {0}) C \bigr \}, \qquad \mathrm {a n d} \qquad L _ {m} ^ {-} = \max \bigl \{K _ {m} ^ {-}, S (t _ {0}) / C \bigr \},
$$

respectively, where  $C > 1$  is a suitable constant. In practice, the choice

$$
\log C = 1 0 \sigma_ {0} \sqrt {t _ {M} - t _ {0}} + \left(1 + \frac {1}{2} \sigma_ {0} ^ {2}\right) (t _ {M} - t _ {0}),
$$

where  $\sigma_0 = \max_{1\leq m\leq M}\sigma_m$ , is sufficient to reduce the truncation errors to round-off level. Heuristically this is clear from (2.3), which suggests that the chance that  $S(t_{m})$  move outside the range  $(S(t_0) / C,S(t_0)C)$  is negligibly small. The rigorous derivation of the error bounds can be obtained using a recursive argument, as will be explained in the Appendix.

Now we consider the truncated integral

$$
\tilde {F} _ {m - 1} (S) = e ^ {- 2 r _ {m} \tau_ {m} / \sigma_ {m} ^ {2}} \int_ {L _ {m} ^ {-}} ^ {L _ {m} ^ {+}} \tilde {V} _ {m} (y) \rho_ {m} (y, S) d y.
$$

If  $K_{m}^{-} \geq S(t_{0})C$  or  $K_{m}^{+} \leq S(t_{0}) / C$ , then by convention the integral is zero. Thus in what follows we shall assume  $K_{m}^{-} < S(t_{0})C$  and  $K_{m}^{+} > S(t_{0}) / C$ . Let

$$
B _ {m} ^ {\pm} = \log \frac {L _ {m} ^ {\pm}}{S (t _ {0})},
$$

and denote

$$
S = S (t _ {0}) e ^ {x}, \qquad y = S (t _ {0}) e ^ {z},
$$

$$
\alpha_ {m} = \frac {1}{\sigma_ {m} ^ {2}} \big [ r _ {m} - q _ {m} - \frac {1}{2} \sigma_ {m} ^ {2} \big ], \qquad \beta_ {m} = \frac {1}{\sigma_ {m} ^ {4}} \big [ r _ {m} - q _ {m} - \frac {1}{2} \sigma_ {m} ^ {2} \big ] ^ {2} + \frac {2 r _ {m}}{\sigma_ {m} ^ {2}},
$$

$$
u _ {m} (x) = \tilde {V} _ {m} (S (t _ {0}) e ^ {x}), \qquad w _ {m} (x) = \exp \Bigl \{- \frac {x ^ {2}}{4 \tau_ {m}} - \alpha_ {m} x \Bigr \}.
$$

The truncated integral can be rewritten as

$$
\tilde {F} _ {m - 1} \left(S \left(t _ {0}\right) e ^ {x}\right) = \frac {e ^ {- \beta_ {m} \tau_ {m}}}{2 \sqrt {\pi \tau_ {m}}} \int_ {B _ {m} ^ {-}} ^ {B _ {m} ^ {+}} w _ {m} (x - z) u _ {m} (z) d z. \tag {4.2}
$$

One can show by differentiating (4.2) that  $\tilde{F}_m$ , and thus  $\tilde{V}_m$  and  $u_m$ , are smooth functions in  $x$ . This means we can compute the integrals efficiently using a high-order quadrature such as Simpson's rule.

In general,  $B_{m}^{\pm}$  are different for different values of  $m$ , so they cannot all be placed on one grid. Now we choose a uniform grid  $\mathbf{x} = \{x_{1}, x_{2}, \ldots, x_{N}\}$ , where  $x_{1} = -\log C$  and  $x_{N} = \log C$ . Let

$$
h = \frac {x _ {N} - x _ {1}}{N - 1} = \frac {2 \log C}{N - 1}.
$$

For each  $m$ , let

$$
p _ {m} ^ {-} = \min  \{i \colon x _ {i} \geq B _ {m} ^ {-} \}, \qquad p _ {m} ^ {+} = \max  \{i \colon x _ {i} <   B _ {m} ^ {+} \},
$$

where by definition  $p_m^- \geq 1$  and  $p_m^+ < N$ . Since we will use Simpson's rule which requires an odd number of grid points, we define

$$
p _ {0} = \left(p _ {m} ^ {+} - p _ {m} ^ {-}\right) \mod 2,
$$

and rewrite (4.2) as

(4.3)

$$
\begin{array}{l} \tilde{F}_{m - 1}(S(t_{0})e^{x}) = \frac{e^{-\beta_{m}\tau_{m}}}{2\sqrt{\pi\tau_{m}}}\left(\int_{x_{\substack{p_{m} + p_{0}\\ p_{m}}}}^{x_{p_{m} + p_{0}}}w_{m}(x - z)u_{m}(z)dz\right. \\ \left. + \int_ {B _ {m} ^ {-}} ^ {x _ {p _ {m} ^ {-}}} w _ {m} (x - z) u _ {m} (z)   d z + \int_ {x _ {p _ {m} ^ {+} + p _ {0}}} ^ {B _ {m} ^ {+}} w _ {m} (x - z) u _ {m} (z)   d z\right). \\ \end{array}
$$

For each  $2 \leq m \leq M - 1$ , we will compute  $\tilde{F}_{m-1}(S(t_0)e^x)$  for all

$$
x \in \left\{x _ {1}, x _ {2}, \dots , x _ {N}, B _ {m - 1} ^ {-}, B _ {m - 1} ^ {+}, \xi_ {m - 1} ^ {-}, \xi_ {m - 1} ^ {+} \right\},
$$

where

$$
\xi_ {m - 1} ^ {-} = \frac {1}{2} (x _ {p _ {m - 1} ^ {-}} + B _ {m - 1} ^ {-}), \qquad \xi_ {m - 1} ^ {+} = \frac {1}{2} (x _ {p _ {m - 1} ^ {+} + p _ {0}} + B _ {m - 1} ^ {+}).
$$

For  $m = 1$  we only need to compute  $\tilde{F}_{m - 1}(S(t_0)e^x)$  for  $x = 0$ , since the value of the option is given by  $\tilde{V}_0(S(t_0))$ .

4.1. Computation of the first integral in (4.3). To compute the first integral in (4.3) using Simpson's rule, we let

$$
U _ {m} (i) = \left\{ \begin{array}{l l} u _ {m} (x _ {i}), & i = p _ {m} ^ {-}, p _ {m} ^ {+} + p _ {0} \\ 4 u _ {m} (x _ {i}), & i = p _ {m} ^ {-} + 1, p _ {m} ^ {-} + 3, \ldots , p _ {m} ^ {+} + p _ {0} - 1 \\ 2 u _ {m} (x _ {i}), & i = p _ {m} ^ {-} + 2, p _ {m} ^ {-} + 4, \ldots , p _ {m} ^ {+} + p _ {0} - 2 \end{array} \right..
$$

The integral is discretized as

$$
\int_ {x _ {p _ {m} ^ {-}}} ^ {x _ {p _ {m} ^ {+} + p _ {0}}} w _ {m} (x - z) u _ {m} (z) d z = \frac {h}{3} \sum_ {i = p _ {m} ^ {-}} ^ {p _ {m} ^ {+} + p _ {0}} w _ {m} (x - x _ {i}) U _ {m} (i) + O \left(h ^ {4}\right), \tag {4.4}
$$

since Simpson's rule is of order 4 [8]. Note that  $U_{m}(i)$  is known from the previous step (or by Proposition 3.3 for  $m = M - 1$ ) for all  $i = 1,2,\ldots ,N$ . For  $x\in \{B_{m - 1}^{\pm},\xi_{m - 1}^{\pm},0\}$ , the sum (4.4) can be computed directly with complexity  $O(N)$ . For all grid points  $x\in \{x_1,x_2,\dots ,x_N\}$ , on the other hand, the sum (4.4) can be computed altogether using FFT with complexity  $O(N\log N)$ . This latter fact is crucial to the efficient implementation of our quadrature method and is a consequence of the following simple observation.

Proposition 4.1. Define  $(2N - 1)$ -periodic grid functions  $\hat{z}$ ,  $\hat{U}_m$ , and  $\hat{F}_m$  by

$$
\hat {z} (i) = z _ {i} = - 2 \log C + (i - 1) h, \qquad 1 \leq i \leq 2 N - 1,
$$

$$
\hat {U} _ {m} (i) = \left\{ \begin{array}{l l} 0, & 1 \leq i <   p _ {m} ^ {-} \\ U _ {m} (i), & p _ {m} ^ {-} \leq i \leq p _ {m} ^ {+} + p _ {0} \\ 0, & p _ {m} ^ {+} + p _ {0} <   i \leq 2 N - 1 \end{array} \right.,
$$

$$
\hat {F} _ {m} = \mathcal {F} ^ {- 1} \Bigl \{\mathcal {F} \big (w _ {m} (\hat {z}) \big) \mathcal {F} (\hat {U} _ {m}) \Bigr \},
$$

where  $\mathcal{F}$  and  $\mathcal{F}^{-1}$  denote the discrete Fourier transform and the inverse discrete Fourier transform of size  $2N - 1$ , respectively. Then

$$
\int_ {x _ {\overline {{p m}}} ^ {+}} ^ {x _ {p m} ^ {+} + p _ {0}} w _ {m} (x _ {j} - z) u _ {m} (z) d z = \frac {h}{3} \hat {F} _ {m} (j + N) + O (h ^ {4}),
$$

for all  $1 \leq j \leq N$ . The above discrete Fourier transforms and inverse discrete Fourier transform can be calculated using FFT, and the total computational complexity is  $O(N \log N)$ .

Proof. We consider the discrete convolution

$$
G _ {m} (j) = \sum_ {i = 1} ^ {2 N - 1} w _ {m} (z _ {j - i}) \hat {U} _ {m} (i),
$$

for  $j\in \mathbb{Z}$  . Note that by definition,

$$
z _ {j + N - i} = - 2 \log C + (j - i + N - 1) h = (j - i) h = x _ {j} - x _ {i},
$$

for all  $1 - N \leq j - i \leq N - 1$ . Thus for  $1 \leq j \leq N$ , we have

$$
\begin{array}{l} G _ {m} (j + N) = \sum_ {i = 1} ^ {2 N - 1} w _ {m} \left(z _ {j + N - i}\right) \hat {U} _ {m} (i) \\ = \sum_ {i = p _ {m} ^ {-}} ^ {p _ {m} ^ {+} + p _ {0}} w _ {m} (z _ {j + N - i}) U _ {m} (i) = \sum_ {i = p _ {m} ^ {-}} ^ {p _ {m} ^ {+} + p _ {0}} w _ {m} (x _ {j} - x _ {i}) U _ {m} (i). \\ \end{array}
$$

Therefore (4.4) with  $x = x_{j}$  can be written as

$$
\int_ {x _ {\bar {p _ {m}}}} ^ {x _ {\bar {p _ {m}} + p _ {0}}} w _ {m} (x _ {j} - z) u _ {m} (z) d z = \frac {h}{3} G _ {m} (j + N) + O \left(h ^ {4}\right). \tag {4.5}
$$

The discrete convolution  $G_{m}$  can be calculated using FFT as

$$
G _ {m} = \mathcal {F} ^ {- 1} \Big \{\mathcal {F} \big (w _ {m} (\hat {z}) \big) \mathcal {F} (\hat {U} _ {m}) \Big \} = \hat {F} _ {m},
$$

with a complexity of  $O(N \log N)$  [8].

![](images/8c16e9363885c5b0acc720d3011e9a104a84f879454e3dcf582652158011239d.jpg)

Remark 4.2. Our method differs from other well-known FFT-based methods (such as [21, 22]) in that we express the discrete quadrature rule (4.4) directly in terms of discrete Fourier transforms, instead of applying continuous Fourier transform to the integral and then discretizing the Fourier integrals (in other words, we have exchanged the order of Fourier transform and discretization). The direct application of the discrete Fourier transform (to the discrete quadrature rule) not only eliminates the need for artificially-introduced damping factors, which are required for the existence of the continuous Fourier transforms, but also eliminates the need for additional specially-designed computational grids which are required to satisfy Nyquist relations. This enables us to carry out the main recursion (3.5) on a fixed uniform grid, without any additional artificial parameters.

4.2. Computation of the last two integrals in (4.3). The last two integrals in (4.3) are calculated in similar ways using Simpson's rule. First, note that we may use Proposition 3.3 to calculate  $\tilde{V}_{M-1}(S(t_0)e^x)$  for  $x \in \{B_{M-1}^\pm, \xi_{M-1}^\pm\}$ . Generally all four points are needed if the option has two barriers, and only two are needed if the option has one barrier. For each  $2 \leq m \leq M-1$ , assume  $u_m(x) = \tilde{V}_m(S(t_0)e^x)$  has been calculated for  $x \in \{B_m^\pm, \xi_m^\pm\}$ . The last two integrals in (4.3) are calculated using Simpson's rule as follows:

(4.6)

$$
\begin{array}{l} \int_ {B _ {\bar {m}} ^ {-}} ^ {x _ {p _ {\bar {m}} ^ {-}}} w _ {m} (x - z) u _ {m} (z) d z = \frac {1}{6} \left(x _ {p _ {\bar {m}}} - B _ {m} ^ {-}\right) \left[ w _ {m} (x - B _ {m} ^ {-}) u _ {m} \left(B _ {m} ^ {-}\right) \right. \\ \left. + 4 w _ {m} (x - \xi_ {m} ^ {-}) u _ {m} (\xi_ {m} ^ {-}) + w _ {m} (x - x _ {p _ {m} ^ {-}}) u _ {m} (x _ {p _ {m} ^ {-}}) \right] + O (h ^ {4}), \\ \end{array}
$$

(4.7)

$$
\begin{array}{l} \int_{x_{p_{m}^{+} + p_{0}}}^{B_{m}^{+}}w_{m}(x - z)u_{m}(z)  dz = \frac{1}{6}\left(B_{m}^{+} - x_{p_{m}^{+} + p_{0}}\right)\bigl[w_{m}(x - B_{m}^{+})u_{m}(B_{m}^{+}) \\ \left. + 4 w _ {m} \left(x - \xi_ {m} ^ {+}\right) u _ {m} \left(\xi_ {m} ^ {+}\right) + w _ {m} \left(x - x _ {p _ {m} ^ {+} + p _ {0}}\right) u _ {m} \left(x _ {p _ {m} ^ {+} + p _ {0}}\right) \right] + O \left(h ^ {4}\right). \\ \end{array}
$$

# 5. FINDING OPTIMAL EXERCISE PRICES FOR BERMUDAN OPTIONS

Unlike autocollable products and barrier options, Bermudan options do not have pre-specified exercise levels. Instead, one needs to solve for  $K_{m}^{\pm}$  from the equations

$$
\tilde {V} _ {m} \left(K _ {m} ^ {+}\right) = K _ {m} ^ {+} - K,
$$

for call options and

$$
\tilde {V} _ {m} (K _ {m} ^ {-}) = K - K _ {m} ^ {-},
$$

for put options, where  $\tilde{V}_m$  is determined by (3.5). For simplicity we assume the yield rates  $q_{m} \geq 0$ , which is almost always the case in practice. We will demonstrate how to find  $K_m^-$ , as the same procedure applies to  $K_m^+$ . Let

$$
p = \min  \left\{i \colon \tilde {V} _ {m} (S (t _ {0}) e ^ {x _ {i}}) > K - S (t _ {0}) e ^ {x _ {i}} \right\}.
$$

If  $p = 1$  there is no early exercise, so  $K_{m}^{-} = 0$ . Otherwise we have  $S(t_0)e^{x_p - 1} \leq K_m^- < S(t_0)e^{x_p}$  by Corollary 8.3. The value of  $K_{m}^{-}$  can be found using classical root-finding methods such as the bisecting method or the secant method. Note that the bisecting method is guaranteed to converge by Corollary 8.3, and it takes  $O(\log N)$  steps to reduce the error of the approximate root to an order of  $O(h^4)$ . Since the cost for calculating  $\tilde{V}_m$  at one point using (4.1) is  $O(N)$ , the total cost for finding the optimal exercise price is  $O(N\log N)$ . The secant method is superlinear and converges faster than the bisecting method, though its error estimates are not as straightforward.

# 6. SUMMARY OF THE ALGORITHM

We summarize our algorithm as follows:  
1: Define the functions  $V_{m}^{a}, V_{m}^{b}$  as in Lemma 3.1  
2: Define the function  $\tilde{V}_{M - 1}$  as in Proposition 3.3  
3: if option style is Bermudan then  
4: Calculate  $K_{M - 1}^{\pm}$  as in Section 5  
5: end if  
6: Calculate  $p_{M-1}^{\pm}$  and  $p_0$  to find the bounds of integration in (4.3)  
7: Use Proposition 3.3 to compute  $\tilde{V}_{M - 1}(S(t_0)e^x)$  for  $x\in \{B_{M - 1}^{\pm},\xi_{M - 1}^{\pm}\}$ , and assign their values to  $v_{1}^{\pm},v_{2}^{\pm}$  respectively  
8: Define a vector  $\mathbf{S}$  as  $S(i)\gets S(t_0)e^{x_i}$  for  $i = 1,2,\ldots ,N$  
9: Define a vector  $\mathbf{y}$  as  $y(i)\gets \tilde{V}_{M - 1}(S(i))$  for  $i = 1,2,\ldots ,N$  
10: for  $m = M - 1$  downto 2 do  
11: Let  $\hat{z}$  and  $\hat{U}_m$  be as defined in Proposition 4.1  
12:  $\hat{F}_m\gets \mathcal{F}^{-1}\Bigl \{\mathcal{F}\big(w_m(\hat{z})\big)\mathcal{F}(\hat{U}_m)\Bigr \}$  
13: Define (or redefine) the vector  $\mathbf{Y_1}$  as  $Y_{1}(j)\gets \frac{h}{3}\hat{F}_{m}(j + N)$  for  $j = 1,2,\ldots ,N$  
14: Use (4.6), (4.7), and the values of  $v_{1}^{\pm}, v_{2}^{\pm}, y(p_{m}^{-}), y(p_{m}^{+} + p_{0})$  to compute the last two integrals in (4.3) at  $x_{i}$  for  $i = 1, 2, \ldots, N$ , and assign their values to  $\mathbf{Y}_{\mathbf{2}}$  and  $\mathbf{Y}_{\mathbf{3}}$  
15: if option style is Bermudan then  
16: Calculate  $K_{m-1}^{\pm}$  as in Section 5  
17: end if  
18: Calculate  $p_{m-1}^{\pm}$  and  $p_0$  to find the bounds of integration in (4.3)

19: Compute  $\tilde{V}_{m - 1}(S(t_0)e^x)$  for  $x\in \{B_{m - 1}^{\pm},\xi_{m - 1}^{\pm}\}$  using (4.1) (where the first integral in (4.3) is computed using (4.4), and the last two integrals using (4.6), (4.7), and the existing values of  $v_{1}^{\pm},v_{2}^{\pm},y(p_{m}^{-}),y(p_{m}^{+} + p_{0}))$  , and assign their values to  $v_{1}^{\pm},v_{2}^{\pm}$  respectively

20:

$$
\begin{array}{l} \mathbf {y} \gets \mathbf {Y _ {1}} + \mathbf {Y _ {2}} + \mathbf {Y _ {3}} + a _ {m} ^ {+} V _ {m} ^ {a} (\mathbf {S}, K _ {m} ^ {+}, 1) + b _ {m} ^ {+} V _ {m} ^ {b} (\mathbf {S}, K _ {m} ^ {+}, 1) \\ + a _ {m} ^ {-} V _ {m} ^ {a} (\mathbf {S}, K _ {m} ^ {-}, - 1) + b _ {m} ^ {-} V _ {m} ^ {b} (\mathbf {S}, K _ {m} ^ {-}, - 1), \\ \end{array}
$$

as in (4.1) (note that  $\mathbf{y}$  now stores  $\tilde{V}_{m - 1}(S(i))$  for  $i = 1,2,\ldots ,N$ )

# 21: end for

22: Compute  $\tilde{V}_0(S(t_0))$  using (4.1), where the first integral in (4.3) is computed using (4.4), and the last two integrals using (4.6), (4.7), and the existing values of  $v_{1}^{\pm}, v_{2}^{\pm}, y(p_{1}^{-}), y(p_{1}^{+} + p_{0})$

Since the computational complexity of each step of the loop is  $O(N \log N)$ , the total complexity is  $O(MN \log N)$ .

Remark 6.1. While other quadrature methods typically employ multiple uniform grids or specially-designed (nonuniform or shifted) grids, our method utilizes only a fixed one-dimensional uniform grid, which not only eliminates the need for complicated inter-grid data transfer procedures, but also eliminates the need for special subroutines that are often required to interpolate data across discontinuities. This makes our method particularly easy to implement.

# 7. NUMERICAL EXAMPLES

We will demonstrate the accuracy and efficiency of the proposed method using two examples, in which the value of an autocollable structured product and that of a double barrier option with time-dependent barriers are found.

7.1. Example 1: Autocollable Structured Product. We consider a knock-out autocollable structured product maturing in one year. The price of the underlying asset is 3000, the nominal amount is 1, and the volatility is  $20\%$ . The observation dates (in years from now), barrier levels, and risk-free rates (in  $\%$  ) are given below in Table 1.

TABLE 1. An autocollable structured product.  

<table><tr><td>Observation date</td><td>Barrier level</td><td>Risk-free rate</td></tr><tr><td>0.2</td><td>3050</td><td>2</td></tr><tr><td>0.4</td><td>3100</td><td>2.1</td></tr><tr><td>0.6</td><td>3150</td><td>2.2</td></tr><tr><td>0.8</td><td>3200</td><td>2.3</td></tr><tr><td>1</td><td>3250</td><td>2.4</td></tr></table>

If the asset price reaches or goes above the barrier level at some observation date  $t$ , the investor receives a payment of  $4\% \times t$ . If the asset price is below the barrier at every observation date, the investor will have to pay a premium of  $1\%$ . The relative errors of the computed option values with varying grid sizes are shown

![](images/8fb4272ce17514672b793cb25f0c9635c470a46a34caaac810333769d2dace80.jpg)  
(a)  
FIGURE 1. Errors for the autocollable structured product, computed using (a) the proposed method and (b) Monte-Carlo simulations.

![](images/ac20473aec1c8aa3bc1a96aaa84f36c7217d515e95ca7ccc833e02039e86ffd1.jpg)  
(b)

below in Figure 1(a), where the exact option value is taken to be the one computed on the grid of size 70,001.

As a comparison, the relative errors of the option values computed using Monte-Carlo simulations with antithetic variates technique are shown in Figure 1(b). As is clear from the figures, the error of the option value computed using the proposed method is well within  $10^{-5}$  with just 501 points in the grid, and drops very quickly as the grid size increases. In contrast, it takes more than ten million paths for Monte-Carlo simulations to reduce the error of the computed option value to within  $10^{-3}$ , and the error decays very slowly as the number of paths increases.

Figure 2 below shows the CPU time used by the proposed method to price the autocollable structured product, where the code is developed in Matlab and is run on a personal computer. As is clear from the figure, the CPU time required by

![](images/96531aeb8a75df06aadc76c7c8e90fe0eace010932c7c2fcc2c9ca203c2fb0dd.jpg)  
FIGURE 2. CPU time for autocollable structured product valuation, using the proposed method.

the proposed method is well within 0.01 seconds, and it increases approximately linearly as grid size increases. It is difficult to compare the speed of the proposed

method with that of Monte-Carlo simulations directly, since the CPU time required by the latter depends largely on specific implementations. Nevertheless, the typical CPU time consumed by a Monte-Carlo simulation with tens of millions of paths ranges from tens of seconds to a few minutes.

7.2. Example 2: Double Barrier Option. As another example, consider a knock-out double barrier put option with time-dependent barrier levels. The price of the underlying asset is 2500, the strike price is 2600, the nominal amount is 1, and the volatility is  $25\%$ . The option matures in two years. The observation dates (in years from now), barrier levels, and risk-free rates (in  $\%$  ) are given below in Table 2.

TABLE 2. A double barrier option.  

<table><tr><td>Observation date</td><td>Barrier level 1</td><td>Barrier level 2</td><td>Risk-free rate</td></tr><tr><td>0.25</td><td>2200</td><td>2800</td><td>1</td></tr><tr><td>0.50</td><td>2100</td><td>2900</td><td>1.1</td></tr><tr><td>0.75</td><td>2000</td><td>3000</td><td>1.2</td></tr><tr><td>1</td><td>1900</td><td>3100</td><td>1.3</td></tr><tr><td>1.25</td><td>1800</td><td>3200</td><td>1.2</td></tr><tr><td>1.50</td><td>1700</td><td>3300</td><td>1.3</td></tr><tr><td>1.75</td><td>1600</td><td>3400</td><td>1.4</td></tr><tr><td>2</td><td>-</td><td>-</td><td>1.5</td></tr></table>

If the asset price falls below barrier level 1 or rises above barrier level 2 at any observation date, the option ceases to exist. If the option is still valid at maturity, the payoff is the same as that of a vanilla put option. The relative errors of the computed option values with varying grid sizes are shown below in Figure 3(a), where the exact option value is taken to be the one computed on the grid of size 50,001.

![](images/115509d28b05d8ad52772dabda66f984b7b5668255e06aea363ad52d789c1484.jpg)  
(a)

![](images/6caa9b465cb6ac759efd177fb21c3072ba0cde46cfbb7789294c5d00f9c5e4fe.jpg)  
(b)  
FIGURE 3. Errors for the double barrier option, computed using (a) the proposed method and (b) Monte-Carlo simulations.

As a comparison, the relative errors of the option values computed using Monte-Carlo simulations with antithetic variates technique are shown in Figure 3(b). As is clear from the figures, the error of the option value computed using the proposed method is within  $10^{-5}$  with just 701 points in the grid, and drops very quickly as the grid size increases. In contrast, it takes more than ten million paths for Monte-Carlo simulations to reduce the error of the computed option value to within  $10^{-3}$ , and the error decays very slowly as the number of paths increases.

Figure 4 below shows the CPU time used by the proposed method to price the double barrier option, where the code is developed in Matlab and is run on a personal computer. As is clear from the figure, the CPU time required by the

![](images/7fd6b6374c8a854f3b45d766acbb916c38e54dd74f20a2d90e22744f1c9b5f3a.jpg)  
FIGURE 4. CPU time for double barrier option valuation, using the proposed method.

proposed method is within 0.01 seconds, and it increases approximately linearly as grid size increases. In contrast, the typical CPU time consumed by a Monte-Carlo simulation with tens of millions of paths ranges from tens of seconds to a few minutes.

Remark 7.1. Although the designed order of the proposed method is 4, in the above numerical examples, a lower order of convergence (close to 3) is actually observed for the grid sizes considered. It is interesting to note that this apparent "loss" of order of accuracy is not a defect of our method; rather, it is a manifestation of the subtle influences that barrier levels can have on option pricing algorithms. These influences can be understood from two perspectives. First, in the above examples, the barrier levels  $K_{m}^{\pm}$  are close to the spot price  $S(t_0)$ . This gives rise to a relatively small integration domain  $[B_m^-, B_m^+]$  compared with the entire computational domain  $[-\log C, \log C]$  (recall that

$$
B _ {m} ^ {+} - B _ {m} ^ {-} = \log \frac {L _ {m} ^ {+}}{S (t _ {0})} - \log \frac {L _ {m} ^ {-}}{S (t _ {0})} = \log \frac {L _ {m} ^ {+}}{L _ {m} ^ {-}} \leq \min \biggl \{\log \frac {K _ {m} ^ {+}}{K _ {m} ^ {-}}, \log \frac {K _ {m} ^ {+}}{C ^ {- 1}} \biggr \}
$$

), which means that the set of grid points that are available for the discrete quadrature rule (4.4) represents only a relatively small fraction of the set of grid points introduced on the entire computational domain. Secondly, in the above examples the option prices  $V_{m + 1}(S)$  contain discontinuities at each observation date  $t_{m + 1}$ . These discontinuities necessarily show up in the form of large gradients in the (smooth) functions  $\tilde{V}_m(S)$  (via the expectation integrals  $\mathbb{E}\big[V_{m + 1}(\cdot)|S\big]$ ), which means that the

discrete quadrature rule (4.4) is being applied to fast-varying functions with only a relatively small number of grid points, leaving the integrands only marginally resolved and hence explaining the degeneracy observed in the convergence rate. If the barrier levels  $K_{m}^{\pm}$  are pushed farther away from the spot price  $S(t_0)$ , so that the option becomes increasingly like a vanilla option, then the discrete quadrature rule (4.4) effectively applies to slow-varying functions with a relatively large number of grid points, which improves the resolution of the integrands and hence the convergence rate (to close to 4). Despite these caveats on convergence rate, we emphasize that our method is capable of pricing a sophisticated discretely monitored option and obtaining five to six significant digits within a fraction of a second, while at the same time being very easy to understand and to implement. Thus the (relatively technical) issue of convergence rate should pose no real concerns in practice.

Remark 7.2. Although a relative error of the order  $10^{-5}$  or  $10^{-6}$  may not always seem necessary for option pricing problems considered in the real financial world, this extra accuracy is actually needed in the calculation of the Greeks, which are typically approximated by finite difference formulas and which are much more sensitive to numerical errors incurred in the calculation of option prices.

# 8. APPENDIX

8.1. Estimate of Truncation Errors. To estimate the truncation error for the integral in (4.1), we first introduce

Lemma 8.1. Let

$$
\begin{array}{l} A = \max  \left\{\left| a _ {1} ^ {\pm} \right|, \dots , \left| a _ {M} ^ {\pm} \right|, \left| a _ {M} \right| \right\}, \quad B = \max  \left\{\left| b _ {1} ^ {\pm} \right|, \dots , \left| b _ {M} ^ {\pm} \right|, \left| b _ {M} \right| \right\}, \\ R = \min  \left\{r _ {1}, \dots , r _ {M}, 0 \right\}, \quad Q = \min  \left\{q _ {1}, \dots , q _ {M}, 0 \right\}. \\ \end{array}
$$

Then

$$
\left| \tilde {V} _ {m} (S) \right| \leq e ^ {Q \left(t _ {m} - t _ {M}\right)} A S + e ^ {R \left(t _ {m} - t _ {M}\right)} B, \quad \forall 0 \leq m \leq M, \forall S \in \left(K _ {m} ^ {-}, K _ {m} ^ {+}\right),
$$

and

$$
\left| V _ {m} (S) \right| \leq e ^ {Q \left(t _ {m} - t _ {M}\right)} A S + e ^ {R \left(t _ {m} - t _ {M}\right)} B, \quad \forall 0 \leq m \leq M, \forall S \in (0, \infty).
$$

Proof. Clearly, by assumption (cf. (3.4)),

$$
\left| \tilde {V} _ {M} (S) \right| = \left| V _ {M} (S) \right| \leq A S + B, \quad \forall S \in (0, \infty).
$$

Now assume

$$
\left| \tilde {V} _ {m} (S) \right| \leq e ^ {Q \left(t _ {m} - t _ {M}\right)} A S + e ^ {R \left(t _ {m} - t _ {M}\right)} B, \quad \forall S \in \left(K _ {m} ^ {-}, K _ {m} ^ {+}\right),
$$

and

$$
\left| V _ {m} (S) \right| \leq e ^ {Q \left(t _ {m} - t _ {M}\right)} A S + e ^ {R \left(t _ {m} - t _ {M}\right)} B, \quad \forall S \in (0, \infty),
$$

for some  $1 \leq m \leq M$ . By (3.2) and Proposition 3.4, we have

$$
\tilde {V} _ {m - 1} (S) = e ^ {- 2 r _ {m} \tau_ {m} / \sigma_ {m} ^ {2}} \int_ {0} ^ {\infty} V _ {m} (y) \rho_ {m} (y, S) d y,
$$

for  $K_{m-1}^{-} < S < K_{m-1}^{+}$ . By definition (3.1), it is clear that  $\rho_m(y, S) = \rho_m(y / S, 1) / S$ . Thus a simple change of variable  $z = y / S$  yields

$$
\begin{array}{l} | \tilde {V} _ {m - 1} (S) | = e ^ {- 2 r _ {m} \tau_ {m} / \sigma_ {m} ^ {2}} \left| \int_ {0} ^ {\infty} V _ {m} (S z) \rho_ {m} (z, 1) d z \right| \\ \leq e ^ {- 2 r _ {m} \tau_ {m} / \sigma_ {m} ^ {2}} \int_ {0} ^ {\infty} \left[ e ^ {Q (t _ {m} - t _ {M})} A S z + e ^ {R (t _ {m} - t _ {M})} B \right] \rho_ {m} (z, 1) d z \\ = e ^ {- 2 r _ {m} \tau_ {m} / \sigma_ {m} ^ {2}} \left(e ^ {Q (t _ {m} - t _ {M})} A S \int_ {0} ^ {\infty} z \rho_ {m} (z, 1) d z + e ^ {R (t _ {m} - t _ {M})} B\right). \\ \end{array}
$$

The integral  $\int_0^\infty z\rho_m(z,1)dz$  is the expectation of the lognormal distribution, which is simply  $\exp \{2(r_m - q_m)\tau_m / \sigma_m^2\}$  [19]. Thus we have (observe  $R\leq 0$  and  $Q\leq 0$ )

$$
\begin{array}{l} | \tilde {V} _ {m - 1} (S) | \leq e ^ {- 2 q _ {m} \tau_ {m} / \sigma_ {m} ^ {2} + Q (t _ {m} - t _ {M})} A S + e ^ {- 2 r _ {m} \tau_ {m} / \sigma_ {m} ^ {2} + R (t _ {m} - t _ {M})} B \\ \leq e ^ {Q \left(t _ {m - 1} - t _ {M}\right)} A S + e ^ {R \left(t _ {m - 1} - t _ {M}\right)} B, \\ \end{array}
$$

for all  $S \in (K_{m-1}^{-}, K_{m-1}^{+})$ . By (3.3) and Proposition 3.4, we then deduce

$$
\left| V _ {m - 1} (S) \right| \leq e ^ {Q \left(t _ {m - 1} - t _ {M}\right)} A S + e ^ {R \left(t _ {m - 1} - t _ {M}\right)} B,
$$

for all  $S\in (0,\infty)$ . The result then follows from induction.

![](images/3b8b38445eb6d42fc19d166399a8c263f598beb8780fa9ba83529d78d3f02b8f.jpg)

The possibly infinite integral in (3.5) is approximated by a finite integral. To be specific, let  $C > 1$ ,  $S_0 = S(t_0)$ , and  $\tilde{G}_{M - 1}(S) = \tilde{V}_{M - 1}(S)$ . We consider  $\tilde{G}_{m - 1}$  ( $1 \leq m \leq M - 1$ ) defined recursively by

$$
\begin{array}{l} \tilde {G} _ {m - 1} (S) = e ^ {- 2 r _ {m} \tau_ {m} / \sigma_ {m} ^ {2}} \mathbb {E} \left[ \tilde {G} _ {m} (\cdot) | S \right] \\ = e ^ {- 2 r _ {m} \tau_ {m} / \sigma_ {m} ^ {2}} \int_ {L _ {m} ^ {-}} ^ {L _ {m} ^ {+}} \tilde {G} _ {m} (y) \rho_ {m} (y, S) d y + a _ {m} ^ {+} V _ {m} ^ {a} (S, K _ {m} ^ {+}, 1) \\ + b _ {m} ^ {+} V _ {m} ^ {b} \left(S, K _ {m} ^ {+}, 1\right) + a _ {m} ^ {-} V _ {m} ^ {a} \left(S, K _ {m} ^ {-}, - 1\right) + b _ {m} ^ {-} V _ {m} ^ {b} \left(S, K _ {m} ^ {-}, - 1\right). \\ \end{array}
$$

A direct calculation shows that the errors  $\tilde{R}_m = \tilde{V}_m - \tilde{G}_m$  satisfy the recursion

$$
\begin{array}{l} \tilde {R} _ {m - 1} (S) = e ^ {- 2 r _ {m} \tau_ {m} / \sigma_ {m} ^ {2}} \left(\int_ {L _ {m} ^ {-}} ^ {L _ {m} ^ {+}} \tilde {R} _ {m} (y) \rho_ {m} (y, S) d y \right. \\ \left. + \int_ {K _ {m} ^ {-}} ^ {L _ {m} ^ {-}} \tilde {V} _ {m} (y) \rho_ {m} (y, S) d y + \int_ {L _ {m} ^ {+}} ^ {K _ {m} ^ {+}} \tilde {V} _ {m} (y) \rho_ {m} (y, S) d y\right). \\ \end{array}
$$

We use the operator notation

$$
\mathcal {T} _ {m} (f) (S) = e ^ {- 2 r _ {m} \tau_ {m} / \sigma_ {m} ^ {2}} \int_ {0} ^ {\infty} f (y) \rho_ {m} (y, S) d y,
$$

to write the recursion of  $\tilde{R}_m$  as

$$
\tilde {R} _ {m - 1} = \mathcal {T} _ {m} (\tilde {R} _ {m} \chi_ {(L _ {m} ^ {-}, L _ {m} ^ {+})}) + \mathcal {T} _ {m} (\tilde {V} _ {m} \chi_ {(K _ {m} ^ {-}, L _ {m} ^ {-} ] \cup [ L _ {m} ^ {+}, K _ {m} ^ {+})}),
$$

for  $1 \leq m \leq M - 1$ . Note also that  $\tilde{R}_{M - 1}(S) = 0$ .

Now we consider  $\tilde{Q}_m$  defined by the recursion

$$
\tilde {Q} _ {m - 1} = \mathcal {T} _ {m} (\tilde {Q} _ {m}) + \mathcal {T} _ {m} \left(\left(e ^ {Q \left(t _ {0} - t _ {M}\right)} A y + e ^ {R \left(t _ {0} - t _ {M}\right)} B\right) \chi_ {\left(0, S _ {0} / C \right] \cup \left[ S _ {0} C, \infty\right)}\right), \tag {8.1}
$$

and  $\tilde{Q}_{M - 1} = 0$ . It is easy to show using Lemma 8.1 and induction that  $|\tilde{R}_m| \leq \tilde{Q}_m$  for all  $0 \leq m \leq M - 1$ . We can also apply the recursion formula (8.1) to get the expansion

$$
\tilde {Q} _ {0} = \sum_ {m = 1} ^ {M - 1} \mathcal {T} _ {1} \circ \dots \circ \mathcal {T} _ {m} \big ((e ^ {Q (t _ {0} - t _ {M})} A y + e ^ {R (t _ {0} - t _ {M})} B) \chi_ {(0, S _ {0} / C ] \cup [ S _ {0} C, \infty)} \big).
$$

Since  $\mathcal{T}_m$  is the risk-neutral expectation operator discounted from  $t_m$  to  $t_{m - 1}$ ,  $\mathcal{T}_1\circ \dots \circ \mathcal{T}_m(f)$  is simply the value of a European option whose payoff at  $t_m$  is given by  $f$ . Thus  $\tilde{Q}_0$  is equal to the value of a sum of  $2(M - 1)$  binary call options with strike  $S_0C$  and  $2(M - 1)$  binary put options with strike  $S_0 / C$ . As a result,

$$
\begin{array}{l} \tilde {Q} _ {0} \left(S _ {0}\right) = e ^ {Q \left(t _ {0} - t _ {M}\right)} A \sum_ {m = 1} ^ {M - 1} \left[ C _ {a} \left(S _ {0}, S _ {0} C, t _ {m}\right) + P _ {a} \left(S _ {0}, S _ {0} / C, t _ {m}\right) \right] \tag {8.2} \\ + e ^ {R \left(t _ {0} - t _ {M}\right)} B \sum_ {m = 1} ^ {M - 1} \left[ C _ {b} \left(S _ {0}, S _ {0} C, t _ {m}\right) + P _ {b} \left(S _ {0}, S _ {0} / C, t _ {m}\right) \right], \\ \end{array}
$$

where  $C_a, P_a, C_b, P_b$  denote values of asset-or-nothing call, asset-or-nothing put, cash-or-nothing call, and cash-or-nothing put respectively. According to Lemma 3.1, the values of these binary options are given by

$$
\begin{array}{l} C _ {a} \left(S _ {0}, S _ {0} C, t _ {m}\right) = S _ {0} e ^ {- \bar {q} _ {m} \left(t _ {m} - t _ {0}\right)} N \left(d _ {3}\right), \quad P _ {a} \left(S _ {0}, S _ {0} / C, t _ {m}\right) = S _ {0} e ^ {- \bar {q} _ {m} \left(t _ {m} - t _ {0}\right)} N \left(d _ {4}\right), \\ C _ {b} (S _ {0}, S _ {0} C, t _ {m}) = e ^ {- \bar {r} _ {m} (t _ {m} - t _ {0})} N (d _ {5}), P _ {b} (S _ {0}, S _ {0} / C, t _ {m}) = e ^ {- \bar {r} _ {m} (t _ {m} - t _ {0})} N (d _ {6}), \\ \end{array}
$$

where

$$
d _ {3} = \frac {1}{\bar {\sigma} _ {m} \sqrt {t _ {m} - t _ {0}}} \bigl [ - \log C + (\bar {r} _ {m} - \bar {q} _ {m} + \frac {1}{2} \bar {\sigma} _ {m} ^ {2}) (t _ {m} - t _ {0}) \bigr ],
$$

$$
d _ {4} = \frac {1}{\bar {\sigma} _ {m} \sqrt {t _ {m} - t _ {0}}} \bigl [ - \log C - (\bar {r} _ {m} - \bar {q} _ {m} + \frac {1}{2} \bar {\sigma} _ {m} ^ {2}) (t _ {m} - t _ {0}) \bigr ],
$$

$$
d _ {5} = \frac {1}{\bar {\sigma} _ {m} \sqrt {t _ {m} - t _ {0}}} \bigl [ - \log C + (\bar {r} _ {m} - \bar {q} _ {m} - \frac {1}{2} \bar {\sigma} _ {m} ^ {2}) (t _ {m} - t _ {0}) \bigr ],
$$

$$
d _ {6} = \frac {1}{\bar {\sigma} _ {m} \sqrt {t _ {m} - t _ {0}}} \bigl [ - \log C - (\bar {r} _ {m} - \bar {q} _ {m} - \frac {1}{2} \bar {\sigma} _ {m} ^ {2}) (t _ {m} - t _ {0}) \bigr ],
$$

and  $\bar{r}_m, \bar{q}_m, \bar{\sigma}_m$  represent the time-weighted averages of  $\{r_n, q_n, \sigma_n\}_{n=1}^m$  respectively. Generally, in practice, the absolute values of annual interest and yield rates will not exceed  $50\%$ ,  $t_M - t_0$  will not exceed 10 years, and  $A$  will not exceed 1. For general autocollable structured products,  $B$  will not exceed  $t_M - t_0$ , and for Bermudan options  $B$  will not exceed  $K$ , which is not much larger than  $S_0$ . We may also assume  $M \leq 120$ , which corresponds to products that are not too frequently monitored, say monthly (for more frequently monitored products, such as daily monitored products, continuity correction methods [4, 5] are usually more appropriate). Let  $\sigma_0 = \max_{1 \leq m \leq M} \sigma_m$ . If we choose

$$
\log C = 1 0 \sigma_ {0} \sqrt {t _ {M} - t _ {0}} + \left(1 + \frac {1}{2} \sigma_ {0} ^ {2}\right) (t _ {M} - t _ {0}),
$$

we can make sure that

$$
d _ {3, 4, 5, 6} \leq - 1 0, \quad \text {a n d t h u s} \quad N (d _ {3, 4, 5, 6}) <   1 0 ^ {- 2 3}.
$$

A crude estimate using (8.2) then shows that the error bound  $\tilde{Q}_0(S_0)$  does not exceed  $10^{-15}(S_0 + 1)$ . This means the relative truncation error is negligible for all practical purposes.

8.2. Analysis of  $K_{m}^{\pm}$  for Bermudan Options. The proper application of Proposition 3.4 requires the uniqueness of the exercise prices  $K_{m}^{\pm}$ , which we now establish for Bermudan options.

To begin with, observe that the risk-neutral pricing formulas (3.2)-(3.3) applied to Bermudan options can be written in an alternative form as

$$
(8. 3) \quad V _ {m - 1} (S) = \max  \left\{\tilde {V} _ {m - 1} (S), \epsilon (S - K) \right\}, \quad \forall 1 \leq m \leq M, \forall S \in (0, \infty),
$$

where

$$
\epsilon = \left\{ \begin{array}{l l} 1, & \text {i f t h e o p t i o n i s a B e r m u d a n c a l l} \\ - 1, & \text {i f t h e o p t i o n i s a B e r m u d a n p u t} \end{array} \right.,
$$

and

$$
\begin{array}{l} \tilde {V} _ {m - 1} (S) = e ^ {- 2 r _ {m} \tau_ {m} / \sigma_ {m} ^ {2}} \int_ {0} ^ {\infty} V _ {m} (y) \rho_ {m} (y, S) d y \tag {8.4} \\ = e ^ {- 2 r _ {m} \tau_ {m} / \sigma_ {m} ^ {2}} \int_ {0} ^ {\infty} V _ {m} (S z) \rho_ {m} (z, 1) d z. \\ \end{array}
$$

Since, by definition,

$$
V _ {M} (S) = \max  \left\{0, \epsilon (S - K) \right\},
$$

(8.3) and (8.4) define  $V_{m}$  and  $\tilde{V}_{m}$  recursively for all  $0 \leq m \leq M - 1$ . It is easy to see that  $V_{m}(S) \geq \tilde{V}_{m}(S) > 0$  for all  $0 \leq m \leq M - 1$  and  $S \in (0, \infty)$ .

Proposition 8.2. Assume  $q_m \geq 0$  for all  $1 \leq m \leq M$ . For a Bermudan call option with strike  $K$ , the equation  $\hat{V}_m(K_m^+) = K_m^+ - K$  has at most one finite solution, and

$$
0 <   \tilde {V} _ {m} (S _ {2}) - \tilde {V} _ {m} (S _ {1}) <   S _ {2} - S _ {1}, \quad \forall 1 \leq m \leq M - 1, \forall S _ {2} > S _ {1} > 0.
$$

For a Bermudan put option with strike  $K$ , the equation  $\tilde{V}_m(K_m^-) = K - K_m^-$  has at most one finite solution, and

$$
0 <   \tilde {V} _ {m} (S _ {1}) - \tilde {V} _ {m} (S _ {2}) <   S _ {2} - S _ {1}, \quad \forall 1 \leq m \leq M - 1, \forall S _ {2} > S _ {1} > 0.
$$

Proof. Clearly  $Q = 0$  since  $q_{m} \geq 0$ . The proofs for call and put options are very similar, and we will present the argument for put options only which proceeds by induction. Since  $\tilde{V}_{M-1}$  is the value of a European vanilla put option and  $q_{m} \geq 0$ , its delta is between -1 and 0 [18]. Thus

$$
0 <   \tilde {V} _ {M - 1} (S _ {1}) - \tilde {V} _ {M - 1} (S _ {2}) = - \int_ {S _ {1}} ^ {S _ {2}} \tilde {V} _ {M - 1} ^ {\prime} (S) d S <   S _ {2} - S _ {1},
$$

for all  $S_2 > S_1 > 0$ . Now suppose  $\tilde{V}_{M-1}(S_3) = K - S_3$  and  $\tilde{V}_{M-1}(S_4) = K - S_4$  for some  $S_4 > S_3 > 0$ . This means

$$
S _ {4} - S _ {3} = \tilde {V} _ {M - 1} (S _ {3}) - \tilde {V} _ {M - 1} (S _ {4}) <   S _ {4} - S _ {3},
$$

which is a contradiction. So the proposition is true for  $m = M - 1$ .

Suppose now the results hold for some  $2 \leq m \leq M - 1$ , which implies that the function  $S - K + \tilde{V}_m(S)$  is strictly increasing. Since  $\tilde{V}_m(S) > 0$ , for sufficiently large  $S$  we have  $S - K + \tilde{V}_m(S) > 0$ , or  $\tilde{V}_m(S) > K - S$ . This means that

$$
K _ {m} ^ {-} = \inf  \left\{S > 0: \tilde {V} _ {m} (S) > K - S \right\}
$$

is well-defined and satisfies  $K_{m}^{-} < \infty$ . Consider now (cf. (8.3))

$$
V _ {m} (S) = \left\{ \begin{array}{l l} \tilde {V} _ {m} (S), & S \geq K _ {m} ^ {-} \\ K - S, & S <   K _ {m} ^ {-} \end{array} , \right. \tag {8.5}
$$

and any  $S_2 > S_1 > 0$ . If  $S_2 < K_m^-$  (and hence  $S_1 < K_m^-$ ), we have

$$
V _ {m} (S _ {1}) - V _ {m} (S _ {2}) = (K - S _ {1}) - (K - S _ {2}) = S _ {2} - S _ {1},
$$

by (8.5). If  $S_{1} \geq K_{m}^{-}$  (and hence  $S_{2} \geq K_{m}^{-}$ ), we have

$$
V _ {m} (S _ {1}) - V _ {m} (S _ {2}) = \tilde {V} _ {m} (S _ {1}) - \tilde {V} _ {m} (S _ {2}) \in (0, S _ {2} - S _ {1}),
$$

by (8.5) and inductive hypothesis. If  $S_{1} < K_{m}^{-} \leq S_{2}$ , we have

$$
\begin{array}{l} V _ {m} (S _ {1}) - V _ {m} (S _ {2}) = (K - S _ {1}) - \tilde {V} _ {m} (S _ {2}) \\ <   (K - S _ {1}) - (K - S _ {2}) = S _ {2} - S _ {1}, \\ \end{array}
$$

$$
V _ {m} (S _ {1}) - V _ {m} (S _ {2}) > (K - S _ {1}) - \tilde {V} _ {m} (S _ {1}) \geq 0,
$$

by (8.5), inductive hypothesis, and the definition of  $K_{m}^{-}$ . In conclusion, we have shown that

$$
\begin{array}{l} 0 <   V _ {m} (S _ {1}) - V _ {m} (S _ {2}) = S _ {2} - S _ {1}, \quad \forall K _ {m} ^ {-} > S _ {2} > S _ {1} > 0, (8.6a) \\ 0 <   V _ {m} \left(S _ {1}\right) - V _ {m} \left(S _ {2}\right) <   S _ {2} - S _ {1}, \quad \text {o t h e r w i s e}. (8.6b) \\ \end{array}
$$

With the aid of (8.4) and (8.6), we write

$$
\tilde {V} _ {m - 1} (S _ {1}) - \tilde {V} _ {m - 1} (S _ {2}) = e ^ {- 2 r _ {m} \tau_ {m} / \sigma_ {m} ^ {2}} (I _ {1} + I _ {2}),
$$

where

$$
\begin{array}{l} 0 <   I _ {1} = \int_ {0} ^ {K _ {m} ^ {-} / S _ {2}} \left[ V _ {m} (S _ {1} z) - V _ {m} (S _ {2} z) \right] \rho_ {m} (z, 1) d z \\ = \left(S _ {2} - S _ {1}\right) \int_ {0} ^ {K _ {m} ^ {-} / S _ {2}} z \rho_ {m} (z, 1) d z, \\ \end{array}
$$

$$
\begin{array}{l} 0 <   I _ {2} = \int_ {K _ {m} ^ {-} / S _ {2}} ^ {\infty} \left[ V _ {m} (S _ {1} z) - V _ {m} (S _ {2} z) \right] \rho_ {m} (z, 1) d z \\ <   \left(S _ {2} - S _ {1}\right) \int_ {K _ {m} ^ {-} / S _ {2}} ^ {\infty} z \rho_ {m} (z, 1) d z. \\ \end{array}
$$

As a result,

$$
\begin{array}{l} 0 <   \tilde {V} _ {m - 1} (S _ {1}) - \tilde {V} _ {m - 1} (S _ {2}) \tag {8.7} \\ <   e ^ {- 2 r _ {m} \tau_ {m} / \sigma_ {m} ^ {2}} (S _ {2} - S _ {1}) \int_ {0} ^ {\infty} z \rho_ {m} (z, 1) d z \leq S _ {2} - S _ {1}, \\ \end{array}
$$

since by elementary properties of lognormal distributions [19],

$$
e ^ {- 2 r _ {m} \tau_ {m} / \sigma_ {m} ^ {2}} \int_ {0} ^ {\infty} z \rho_ {m} (z, 1) d z = e ^ {- 2 q _ {m} \tau_ {m} / \sigma_ {m} ^ {2}} \leq 1.
$$

Now suppose  $\tilde{V}_{m - 1}(S_3) = K - S_3$  and  $\tilde{V}_{m - 1}(S_4) = K - S_4$ . This means

$$
\tilde {V} _ {m - 1} (S _ {3}) - \tilde {V} _ {m - 1} (S _ {4}) = S _ {4} - S _ {3},
$$

so by (8.7) we must have  $S_{3} = S_{4}$ . The proposition then follows from induction.

Corollary 8.3. Assume a Bermudan put option with strike  $K$  has an optimal early-exercise level  $K_{m}^{-} > 0$  for some  $1 \leq m \leq M - 1$ . Then we have  $\tilde{V}_{m}(S) > K - S$  for  $S > K_{m}^{-}$  and  $\tilde{V}_{m}(S) < K - S$  for  $S < K_{m}^{-}$ . Similarly, for a Bermudan call option we have  $\tilde{V}_{m}(S) < S - K$  for  $S > K_{m}^{+}$  and  $\tilde{V}_{m}(S) > S - K$  for  $S < K_{m}^{+}$ .

Proof. We will present the proof for put options only as the argument for call options is similar. It follows from Proposition 8.2 that the function  $\tilde{V}_m(S) + S - K$  is increasing in  $S$ . Since  $\tilde{V}_m(K_m^-) + K_m^- - K = 0$ , we have  $\tilde{V}_m(S) > K - S$  for  $S > K_m^-$  and  $\tilde{V}_m(S) < K - S$  for  $S < K_m^-$ .

# ACKNOWLEDGEMENTS

We are very grateful to Qingshuo Song for his valuable insights and helpful suggestions, as well as to Zhenan Sui for her careful reading and editing of the manuscript.

# REFERENCES

[1] D.-H. Ahn, S. Figlewski, and B. Gao, Pricing discrete barrier options with an adaptive mesh model, J. Deriv. 6(4), 33-43 (1999).  
[2] T. Alm, B. Harrach, D. Harrach, and M. Keller, A Monte Carlo pricing algorithm for autocallables that allows for stable differentiation, J. Comput. Financ. 17(1), 43-70 (2013).  
[3] A. D. Andricopoulos, M. Widdicks, P. W. Duck, and D. P. Newton, Universal option valuation using quadrature methods, J. Financ. Econ. 67(3), 447-471 (2003).  
[4] M. Broadie, P. Glasserman, and S. Kou, A continuity correction for discrete barrier options, Math. Financ. 7(4), 325-348 (1997).  
[5] M. Broadie, P. Glasserman, and S. G. Kou, Connecting discrete and continuous path-dependent options, Financ. Stoch. 3(1), 55-82 (1999).  
[6] M. Broadie and Y. Yamamoto, A double-exponential fast Gauss transform algorithm for pricing discrete path-dependent options, Oper. Res. 53(5), 764-779 (2005).  
[7] P. Buchen and O. Konstandatos, A new approach to pricing double-barrier options with arbitrary payoffs and exponential boundaries, Appl. Math. Financ. 16(6), 497-515 (2009).  
[8] R. L. Burden, J. D. Faires, and A. M. Burden, Numerical Analysis (10th Edition), Brooks Cole (2015).  
[9] G. Deng, J. Mallett, and C. McCann, Modeling autocollable structured products, J. Deriv. Hedge Funds 17(4), 326-340 (2011).  
[10] F. Fang and C. W. Oosterlee, Pricing early-exercise and discrete barrier options by Fourier-cosine series expansions, Numer. Math. 114, 27-62 (2009).  
[11] L. Feng and X. Lin, Pricing Bermudan options in Lévy process models, SIAM J. Finan. Math. 4(1), 474-493 (2013).  
[12] L. Feng and V. Linetsky, Pricing discretely monitored barrier options and defaultable bonds in Lévy process models: a fast Hilbert transform approach, Math. Financ. 18(3), 337-384 (2008).  
[13] C. P. Fries and M. S. Joshi, Perturbation stable conditional analytic Monte-Carlo pricing scheme for auto-callable products, Int. J. Theor. Appl. Finance 14(2), 197-219 (2011).  
[14] G. Fusai, I. D. Abrahams, and C. Sgarra, An exact analytical solution for discrete barrier options, Finance Stochast. 10(1), 1-26 (2006).  
[15] J. Gatheral, The Volatility Surface: A Practitioner's Guide, John Wiley & Sons (2006).  
[16] A. Golbabai, L. V. Ballestra, and D. Ahmadian, A highly accurate finite element method to price discrete double barrier options, Comput. Econ. 44(2), 153-173 (2014).  
[17] A. Ibanez and C. Velasco, The optimal method for pricing Bermudan options by simulation, Math. Financ. 28(4), 1143-1180 (2018).  
[18] J. C. Hull, Options, Futures, and Other Derivatives (9th Edition), Pearson (2014).  
[19] N. L. Johnson, S. Kotz, and N. Balakrishnan, “14: Lognormal Distributions”, Continuous Univariate Distributions (Volume 1, 2nd Edition), Wiley Series in Probability and Mathematical Statistics, John Wiley & Sons (1994).

[20] C. F. Lo, H. C. Lee, and C. H. Hui, A simple approach for pricing barrier options with time-dependent parameters, Quant. Financ. 3(2), 98-107 (2003).  
[21] R. Lord, F. Fang, F. Bervoets, and C. W. Oosterlee, A fast and accurate FFT-based method for pricing early-exercise options under Lévy processes, SIAM J. Sci. Comput. 30(4), 1678-1705 (2008).  
[22] C. O'Sullivan, Path dependant option pricing under Lévy processes, EFA 2005 Moscow Meetings Paper (2005).  
[23] E. Reiner, Convolution methods for path-dependent options, Financial Mathematics: Risk Management, Modeling and Numerical Methods, IPAM UCLA (Jan. 3–12, 2001).  
[24] S. E. Shreve, Stochastic Calculus for Finance II: Continuous-Time Models, Springer-Verlag (2004).  
[25] T. Guillaume, Autocollable structured products, J. Deriv. 22(3), 73-94 (2015).  
[26] T. Guillaume, Analytical valuation of autocollable notes, Int. J. Financ. Eng. 2(2), 1-23 (2015).  
[27] J. Z. Wei, Valuation of discrete barrier options by interpolations, J. Deriv. 6(1), 51-73 (1998).

# MIN HUANG

China Merchants Bank

7088 Shennan Boulevard, Shenzhen, Guangdong, China

Email: huang.479@osu.edu

# GUO LUO

Department of Mathematics, City University of Hong Kong

Tat Chee Avenue, Kowloon, Hong Kong

Email: guoluo@cityu.edu.hk
