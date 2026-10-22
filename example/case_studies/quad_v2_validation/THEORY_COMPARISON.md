# QUAD V1 versus V2: theory and numerical mathematics

Implementation comparison recorded on 2026-09-11. See the [validation report](README.md) for measured accuracy, performance and supported scope.

**V1 and V2 share the same pricing theory, but use substantially different numerical approximations.** The largest differences concern barrier discontinuities and Greeks.

## Common pricing model

Both assume Gaussian log-price increments under deterministic rates, carry and volatility:

\[
X_{i+1}=X_i+m_i+\sqrt{v_i}\,Z,\qquad
V_i(x)=D_i\int F_{i+1}(y)\,\phi_{v_i}(y-x-m_i)\,dy.
\]

Here \(X=\log S\), \(Z\) is standard normal, and \(F\) includes contractual events, cashflows and state changes. With cumulative rate, carry and variance denoted by \(R,Q,W\), respectively,

\[
m_i=\Delta R_i-\Delta Q_i-\tfrac12\Delta W_i,\qquad
v_i=\Delta W_i,\qquad D_i=e^{-\Delta R_i}.
\]

V1's transformed kernel using \(\alpha,\beta,\tau\) and V2's mean/variance kernel are algebraically equivalent. Completing the square in V1's exponential recovers the discounted Gaussian transition. V2 does not introduce a different stochastic model or new economic calibration parameters.

## Main numerical differences

| Mathematical aspect | V1 | V2 |
| --- | --- | --- |
| Value representation | Values on a uniform log-price grid | High-order polynomials within Gauss cells, retaining separate continuation branches |
| Discrete barriers | Default cell-average projection; alternative nodal/smoothing modes | Integrates each branch over its contractual barrier region |
| Integration | Trapezoidal/Simpson convolution, with configurable spectral filtering | Gaussian integration of cell polynomials; analytical cash/asset moments |
| Spot Greeks | Desk bump-and-reprice; curves generally use grid gradients | Adds Gaussian-kernel derivatives shared by scalar and curve APIs; retains desk bumps |
| FFT | Applies the nodal quadrature operator | Applies exactly the same block operator as V2's direct backend |
| Zero variance | The stochastic Snowball recursion requires positive variance steps | Explicit deterministic transport, including mixed zero/positive variance intervals |

## Barrier discontinuities

The barrier treatment is the main mathematical improvement. For a down-KI barrier \(b\) in log-price coordinates, the event value is

\[
F(y)=
\begin{cases}
B(y),&y\le b,\\
A(y),&y>b,
\end{cases}
\]

where \(A\) is the surviving branch and \(B\) the knocked-in branch.

V1 first represents this event on its grid. A cell straddling the barrier contains a projected mixture of the two branches. Cell averaging conserves the cell integral of the represented event, but the subsequent Gaussian-weighted integral generally differs because the transition density varies within that cell.

V2 instead computes

\[
V(x)=D\left[
\int_{-\infty}^{b}\widehat B(y)p(y\mid x)\,dy+
\int_b^\infty\widehat A(y)p(y\mid x)\,dy
\right].
\]

The barrier is an integration boundary. The continuation functions are still approximated, but the jump is not interpolated across. This matters most near barriers and shortly before observations, when the transition density changes rapidly within a grid cell. Terminal payoff kinks and jumps receive the same separate-region treatment.

The implementations are in [V1 event projection](../../quantark/asset/equity/engine/quad/snowball_quad_engine.py) and [V2 integration](../../quantark/asset/equity/engine/quad/v2/operator.py).

## Point Greeks and desk Greeks

V2 differentiates the Gaussian kernel for point Greeks. With \(z=y-x-m\),

\[
V_x=D\int F(y)\frac{z}{v}p\,dy,\qquad
V_{xx}=D\int F(y)\left(\frac{z^2}{v^2}-\frac1v\right)p\,dy,
\]

\[
\Delta=\frac{V_x}{S},\qquad
\Gamma=\frac{V_{xx}-V_x}{S^2}.
\]

The minus-\(V_x\) term in gamma is required when converting from log-price derivatives to spot derivatives. Analytical cash/asset subtraction reduces cancellation when variance is tiny. These are derivatives of the numerically represented expectation, not closed-form autocallable Greeks; finite integration, interpolation and tail approximations remain.

V1 already offers a transition-based **price** readout, so its price need not use final linear interpolation. V2 extends the consistent evaluator to point derivatives and curves. See [V2 Gaussian readout](../../quantark/asset/equity/engine/quad/v2/gaussian.py).

Desk finite-bump Greeks are different quantities. For example,

\[
\Delta_\varepsilon=
\frac{V(S(1+\varepsilon))-V(S(1-\varepsilon))}{2S\varepsilon}.
\]

At a finite bump, this need not equal the point derivative, particularly near a barrier. V2 preserves the existing desk bump convention through `calculate_greeks`; `calculate_point_greeks` and spot curves expose point derivatives. Vega, rate/carry risk and theta still use the existing bump-and-reprice conventions rather than new analytical or adjoint formulas.

## Continuous KI and term structures

Both use Brownian-bridge/reflection theory under constant coefficients. For a down barrier and endpoints \(x,y>b\), the survival kernel is

\[
p_{\mathrm{surv}}(y\mid x)
=p(y\mid x)\left[1-\exp\left(-\frac{2(x-b)(y-b)}{v}\right)\right].
\]

V2 additionally differentiates the full survival kernel, including its barrier factor. V1's transition price readout explicitly does not support continuous KI and requires its legacy readout for that case.

V2 also treats the within-interval assumptions more explicitly. Endpoint mean and variance alone do not determine continuous barrier crossing under arbitrary term structures. V1 applies its bridge using interval coefficients. V2 splits known coefficient knots and requires an explicit approximation for more general curves. Thus two runs with different within-interval path assumptions need not agree even if their terminal marginal distributions agree. See [V2 continuous monitoring](../../quantark/asset/equity/engine/quad/v2/continuous.py).

## What should converge, and what is not guaranteed

With matching contractual semantics and monitoring assumptions, and numerical approximations refined away, both should approach the same value. A fixed smoothing width or a different continuous-monitoring approximation must first be accounted for when making that comparison.

V2's advantage is preserving discontinuities and smooth branches more accurately with fewer degrees of freedom. Direct/FFT dispatch and prepared-context reuse improve execution cost without changing the intended pricing equation. This provides a numerical explanation for the measured improvements, while leaving truncation, interpolation and qualification limits in place.

Neither higher polynomial order nor agreement between two meshes proves a universal error bound. The [validation report](README.md) distinguishes independently checked cases, observed refinement, explicit approximations and unsupported capabilities.

## External references

V1's method is documented by its own paper. V2 has no single published counterpart; it is a QuantArk-internal architecture combining ideas from several published methods, anchored per component below.

**V1 (uniform-grid quadrature + FFT recursion).** Huang & Luo, *A Simple and Efficient Numerical Method for Pricing Discretely Monitored Early-Exercise Options* — working paper 2015, published in [Applied Mathematics and Computation (2022)](https://www.sciencedirect.com/science/article/pii/S0096300322000716); text mirrored in [docs/quad/quad.md](../../docs/quad/quad.md). Lineage: Andricopoulos, Widdicks, Duck & Newton (2003), *Universal option valuation using quadrature methods*, J. Financ. Econ. 67(3), and Chen, Härkönen & Newton (2014), *Advancing the universality of quadrature methods to any underlying process for option pricing*, J. Financ. Econ. 114.

**V2 piecewise-polynomial continuation, exact break points, analytic Greek curves.** The closest published analog is Chan & Hale (2018/2020), [*Pricing European-type, Early-Exercise and Discrete Barrier Options using an Algorithm for the Convolution of Legendre Series*](https://repository.uel.ac.uk/download/4f05111edaa6663c8ecdc138d7ce9395c8053f744e3e42f772f8703bea2603d5/341794/AlgoLegQFv2.pdf) (the CONLeg method): piecewise polynomial continuation on subintervals, barriers and payoff kinks as integration break points, Greeks by differentiating the convolution representation, and price/Greek curves rather than scalar output. CONLeg is quadrature-free (exact convolution of Legendre series); V2 instead uses Gauss-Legendre nodes with exact Gaussian kernel moments.

**V2 direct/FFT dual backends.** Lord, Fang, Bervoets & Oosterlee (2008), [*A fast and accurate FFT-based method for pricing early-exercise options under Lévy processes*](http://rogerlord.com/conv.pdf), SIAM J. Sci. Comput. 30(4) (the CONV method): translation-invariant transition operator applied as a padded FFT convolution. V2 applies the identical block operator directly or by FFT, but with high-order Gauss-cell kernels instead of Newton-Cotes weights. Gaussian-kernel convolution with Gaussian quadrature at operator speed also appears in Broadie & Yamamoto (2005), [*A Double-Exponential Fast Gauss Transform Algorithm for Pricing Discrete Path-Dependent Options*](https://www.columbia.edu/~mnb2/broadie/Assets/de-fgt-opr-res.pdf), Oper. Res. 53(5).

**V2 continuous-KI survival kernel.** The reflection (method-of-images) kernel above is standard; see for example Lo, Lee & Hui (2003), *A simple approach for pricing barrier options with time-dependent parameters*, Quant. Finance 3(2), and Buchen & Konstandatos (2009), *A new approach to pricing double-barrier options with arbitrary payoffs and exponential boundaries*, Appl. Math. Financ. 16(6).
