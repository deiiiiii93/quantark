# Valuing Convertible Bonds with Credit Risk

KOSTAS TSIVERIOTIS AND CHRIS FERNANDES

# KOSTAS

TSIVERIOTIS is vice president at Morgan Stanley, Dean Witter in New York, New York.

# CHRIS FERNANDES

is vice president at Morgan Stanley, Dean Witter.

Convertible bonds (CBs) are corporate debt securities that give the holder the right to forgo future coupon and/or principal payments and receive (i.e., convert to) a prespecified number of shares of common stock instead. In principle, a CB is a hybrid security consisting of a straight bond and a call on the underlying equity, but various terms of realistic CBs make it impossible to decouple the stock option from the bond part.

Conditions such as the possibility of early conversion, callability by the issuer, putability by the holder, and special provisions on accrued interest upon early termination introduce extra optionality that depends on both the equity and the fixed-income parts of a CB. Thus, in general, CBs can be accurately valued only by simultaneous pricing of the equity and fixed-income parts.

In other words, CBs should be viewed and valued as derivatives of the underlying equity and interest rates. To that end, a two-factor Black-Scholes model, which accounts for equity and interest rate optionality, is the usual choice.

An important implication in CB pricing comes from the fact that the value of a CB has components of different default risks. For example, if the underlying equity is that of the issuer — as is usually the case — the equity upside has zero default risk since the issuer can always deliver its own stock. On the other hand, coupon and principal pay

ments and any put provisions - allowing the holder to put the CB back to the issuer for a prespecified amount of cash - depend on the issuer's timely access to the required cash amounts, and thus introduce credit risk.

While the Black-Scholes model can account for credit risk by simply using a credit spread in discounting the derivative's value in time, this approach is not valid in the case of a CB because only a part of the CB's value is exposed to default risk — the part that is related to future payments in cash — and that part is not known in advance. In fact, the future cash flows of the CB depend on the possibility of early termination of the CB due to conversion, call, or put, which actions, in turn, are contingent on the random behavior of the underlying stock and interest rates.

Various approaches have been proposed to account for credit risk in CB pricing. Detailed valuation approaches, as reviewed by Nyborg [1996], use the total value of the firm as a stochastic variable and account for the debt obligations of the issuer in defining the random behavior of the firm value. While such approaches are self-consistent in that they address the firm's capital structure, they involve many parameters and can be impractical for use at a trading desk; e.g. they involve the volatility of the firm's value instead of the underlying equity.

On the other hand, practitioners usually account for credit risk in CBs by introducing an effective credit spread in regular

CB valuation tools. Such effective spreads are simple approximations based on the credit spread of a straight bond of the issuer conditioned for the hybrid nature of the CB, as discussed in more detail later.

We introduce a different approach that uses fewer parameters and better fits the needs of practitioners. In particular, we provide a self-consistent framework for the use of market-observed credit spreads of straight bonds in the valuation of CBs. Our approach hinges on the fact that the value of the future cash payments a rational CB holder will choose to receive is itself a derivative of the underlying equity and interest rates, and therefore amenable to the same valuation tools as the original CB.

This observation results in an extended valuation model that, besides the original Black-Scholes equation for the entire CB, involves one additional Black-Scholes equation for each different credit class of payments involved in the CB. Besides the volatilities of the underlying, the only inputs needed are market-observed credit spreads corresponding to the various credit classes involved in the CB's expected cash flows.

In most cases, the optionality due to stochastic moves of interest rates is only a small part of the value of a CB and can be ignored at a first approximation. Thus CBs are usually valued only as derivatives of the underlying stock, i.e., using a single-factor model. Since our focus is to introduce the credit risk in the valuation, we have elected to use the one-factor model to simplify our discussion. Thus, in what follows a CB is viewed as an equity-only derivative. Extension of this approach to two factors is straightforward.

# I. THE MODEL EQUATIONS

As we stated earlier, we view a CB as a contingent claim on the underlying equity. The value  $u$  of CB is thus governed by the Black-Scholes (B-S) equation:

$$
\frac {\partial u}{\partial t} + \frac {\sigma^ {2} S ^ {2}}{2} \frac {\partial^ {2} u}{\partial S ^ {2}} + r _ {g} S \frac {\partial u}{\partial S} - (r + r _ {c}) u + f (u, S, t) = 0 \tag {1}
$$

where S is the price of the underlying stock; r is the risk-free rate;  $\mathbf{r}_{\mathrm{g}}$  is the growth rate of the stock;  $\mathbf{r}_{\mathrm{c}}$  is a credit spread reflecting payoff default risk; and  $f(u,S,t)$  describes various predetermined external flows — in cash or equity — to the derivative; e.g., for a bond paying coupons  $c_{i}$  at times  $t_i$ :

$$
f (u, S, t) = f (t) = \Sigma c _ {i} \delta (t - t _ {j}) \tag {2}
$$

where delta is the Dirac function. For simplicity, we will assume that all external payments to the holder are exclusively in cash, and thus  $f(t)$  will stand for cash flows only.

Although Equation (1) is enough to value any CB with any set of conversion, call, and put conditions, it does not account for the different credit quality of the sources of value for the CBs. This means that one cannot assign a predetermined value to the credit spread  $\mathbf{r}_{\mathrm{c}}$ . For example, if the easily observable credit spread implied by a similar non-convertible bond of the same issuer were used for  $\mathbf{r}_{\mathrm{c}}$ , one would unnecessarily penalize the risk-free equity upside of the CB.

Practitioners often guess an appropriate adjustment to the credit spread to account for the hybrid nature of the CB, or use some function  $\mathbf{r}_{\mathrm{c}}(\mathbf{S})$  that increases for low stock prices. Such approaches — although reasonable — are ad hoc and do not allow for consistent comparison between a CB and a straight corporate bond of the same issuer, or between two CBs of the same issuer but with considerably different features.

A better alternative might be to calculate the probability of conversion and use it to adjust the credit spread. Such an approach, however, would fail to take account of coupon payments correctly or any contingent cash flows occurring due to call and put provisions. Thus, it is clear that a general way is needed to to correctly and simply introduce the issuer's credit spread into CB valuation model.

For any CB, let us define a related security we will refer to as the "cash-only part of the CB" or COCB. This new hypothetical security is defined as follows: The holder of a COCB is entitled to all cash flows, and no equity flows, that an optimally behaving holder of the corresponding CB would receive. By definition, the value  $v$  of the COCB is determined by the behavior of the CB price  $u$ , the stock price  $S$ , and time  $t$ .

Taking into account that the CB is a derivative security of the stock S, we conclude that the COCB is also a derivative security with the same stock as its single underlying. Thus, the COCB price v should follow the Black-Scholes equation as does the CB value u. Furthermore, since the COCB involves only cash payments by the CB issuer, the relevant B-S equation should explicitly involve the issuer's credit spread.

On the other hand,  $(\mathbf{u} - \mathbf{v})$  represents the value of the CB related to payments in equity, and it should

therefore be discounted using the risk-free rate. This leads to a new formulation of the CB valuation problem as a system of two coupled Black-Scholes equations:

$$
\begin{array}{l} \mathrm {C B}: \quad \frac {\partial u}{\partial t} + \frac {\sigma^ {2} S ^ {2}}{2} \frac {\partial^ {2} u}{\partial S ^ {2}} + r _ {\mathrm {g}} S \frac {\partial u}{\partial S} - \\ \mathbf {r} (\mathbf {u} - \mathbf {v}) - \left(\mathbf {r} + \mathbf {r} _ {\mathrm {c}}\right) \mathbf {v} + \mathbf {f} (\mathbf {t}) = 0 \tag {3} \\ \end{array}
$$

$$
\mathrm {C O C B}: \frac {\partial v}{\partial t} + \frac {\sigma^ {2} S ^ {2}}{2} \frac {\partial^ {2} v}{\partial S ^ {2}} +
$$

$$
\mathrm {r _ {g} S} \frac {\partial \mathrm {v}}{\partial \mathrm {S}} - (\mathrm {r} + \mathrm {r _ {c}}) \mathrm {v} + \mathrm {f (t)} = 0 \tag {4}
$$

where  $\mathbf{r}_{\mathrm{c}}$  here is the observable credit spread implied by the non-convertible bonds of the same issuer for similar maturities with the CB. As expected, Equations (3) and (4) are the same except for their discounting terms  $\mathbf{r}(\mathbf{u} - \mathbf{v}) + (\mathbf{r} + \mathbf{r}_{\mathrm{c}})\mathbf{v}$  and  $(\mathbf{r} + \mathbf{r}_{\mathrm{c}})\mathbf{v}$ , which reflect the different credit treatment of cash payments and equity upside.

As is well known, Equations (3) and (4) are parabolic partial differential equations in inverse time  $\tau = T - t$ , where  $T$  is the tenor of the CB; i.e., if we use time to maturity instead of natural time.

The parabolic type of Equations (5-6) means that given: 1) any set of final conditions on  $[\mathbf{u}(\mathrm{S},\mathrm{T}),\mathrm{v}(\mathrm{S},$  T)], and 2) any set of boundary conditions involving any combination of  $\mathbf{u}(\mathbf{S},\mathbf{t})$  .  $\mathbf{v}(\mathbf{S},\mathbf{t})$  .  $\partial \mathbf{u} / \partial \mathbf{S}$  ; and  $\partial \mathbf{v} / \partial \mathbf{S}$  , the system of (5) and (6) is a well posed problem and guaranteed to have a solution; see Carrier and Pearson [1976]. Here, to relate to CB terms, final conditions are determined by the payoff at expiration, while boundary conditions are determined by the early exercise features permissible at all or parts of the life of the CB.

Although Equation (4) might appear to be independent of Equation (3), it is fundamentally coupled to it as shown in greater detail below. The coupling of the two equations comes from the fact that the CB valuation problem is a free boundary problem. In other words, the CB is, in general, an American-style derivative where early call, put, and/or conversion is possible. The stock prices at which these early termination events occur are the free boundaries for Equations (3-4); that is Equations (3-4) are not valid beyond these boundaries.

The free boundaries together with u and v are the unknowns of the problem. Thus, early exercise conditions involving the CB value u, the stock price S,

and time  $t$  define the location of the free boundaries at each time, which, in turn, define the boundary conditions for Equation (4) directly affecting the value  $v$  of the COCB. Thus it is through their common free boundaries — reflecting the early termination events — that Equation (3) and (4) are coupled and thus need to be solved simultaneously.

# II. THE FINAL AND BOUNDARY CONDITIONS

We complete the formulation of the problem by introducing the appropriate final and boundary conditions for solving Equations (3-4). Since these conditions reflect the terms of each individual CB, we deal here only with a generic case of a CB. We consider a CB maturing in time  $T$ , convertible at any time to a shares of the underlying stock  $S$ , and paying a principal  $B$  at expiration if not converted. The CB pays a fixed coupon amount  $c$  at times  $t_i$ . In addition, the CB is callable by the issuer at a price  $B_c$  at any time after  $T_c$  and putable by the holder for a cash amount of  $B_p$  at any time after  $T_p$ .

Given these terms, we obtain the conditions:

Final conditions at expiration:

$$
\begin{array}{l} u (S, T) = \left\{ \begin{array}{l l} a S & \text {f o r} S \geq B / a; \\ B & \text {e l s e w h e r e} \end{array} \right. (5) \\ v (S, T) = \left\{ \begin{array}{l l} 0 & \text {f o r S \geq B / a ;} \\ B & \text {e l s e w h e r e} \end{array} \right. (6) \\ \end{array}
$$

Upside constraints due to conversion:

$$
\mathrm {u} \geq \mathrm {a S} \quad \text {f o r} \mathrm {t} \in [ 0, \mathrm {T} ] \tag {7}
$$

$$
v = 0 \text {i f} u \leq a S \quad \text {f o r} t \in [ 0, T ] \tag {8}
$$

Upside constraints due to callability by the CB issuer:

$$
u \leq \max  \left(B _ {c}, a S\right) \quad f o r t \in \left[ T _ {c}, T \right] \tag {9}
$$

$$
v = 0 \text {i f} u \geq B _ {c} \quad \text {f o r} t \in [ T _ {c}, T ] \tag {10}
$$

where it is assumed that the holder has the right to con

vert if the issuer calls the CB.

Downside constraints due to putability by the CB holder:

$$
\mathrm {u} \geq \mathrm {B} _ {\mathrm {p}} \quad \text {f o r} \mathrm {t} \in \left[ \mathrm {T} _ {\mathrm {p}}, \mathrm {T} \right] \tag {11}
$$

$$
\mathrm {v} = \mathrm {B} _ {\mathrm {p}} \mathrm {i f} \mathrm {u} \leq \mathrm {B} _ {\mathrm {p}} \quad \mathrm {f o r} \mathrm {t} \in [ \mathrm {T} _ {\mathrm {p}}, \mathrm {T} ] \tag {12}
$$

As expected, the COCB assumes a non-zero value in Equation (5-12) only when a cash payment takes place; i.e., in the cases of cash redemption at maturity [Equation (6)] and put [Equation (12)].

The conditions (7-12) suggest that at each time, there are, in general, two stock prices  $S_{\mathrm{d}}$  and  $S_{\mathrm{u}}$  where downside and upside constraints start being binding. These limiting stock prices are unknown, and are a part of the problem's desired solution. In other words,  $S_{\mathrm{d}}$  and  $S_{\mathrm{u}}$  are free boundaries beyond which Equations (3-4) do not apply. Notice that, although the free boundaries  $S_{\mathrm{d}}$  and  $S_{\mathrm{u}}$  are inherent to the problem, they do not appear explicitly in the equations under the present formulation.

Before discussing methods for solving the valuation problem defined by Equations (3-12), it is important to understand the mathematical type of this problem. The system of Equations (3-12) is inherently nonlinear, although the governing Equations (3-4) are linear partial differential equations. This structural nonlinearity comes from the fact that this problem is a free boundary problem.

To see how the non-linearity comes about, one could take the coordinate transformation  $S' = (S - S_d) / (S_u - S_d)$ . This transformation recasts the problem in a fixed domain [0, 1] but also transforms the linear terms  $\frac{\partial u}{\partial S}$  and  $\frac{\partial^2 u}{\partial S^2}$  of Equation (3) to non-linear terms involving all three unknowns  $S_d, S_u$ , and  $u$ ; similar non-linear terms also appear in Equation (4).

Although this transformation has the advantage of explicitly introducing all the unknowns — u, v, and free boundaries  $S_{\mathrm{d}}$  and  $S_{\mathrm{u}}$  — in the partial differential equations and simplifying the boundary conditions, it is not used in our solution method. We use the transformation here only to demonstrate the non-linearity inherent in the valuation of the CB (and any other American-style derivative).

For more on change of coordinates for free and moving boundary problems, see Tsiveriotis and Brown [1993].

# III. NUMERICAL SOLUTION

A first step toward numerical solution is discretization of the partial differential equations in S and t. Various approaches have been proposed for such problems ranging from well-known numerical methods for parabolic partial differential equations, such as finite difference and finite elements, to methods reflecting the underlying stock process, such as lattices.

We have followed the finite difference approach. We describe the method here only briefly since it is covered extensively elsewhere; see, for example, Willmot, Dewynne, and Howison [1994] and Tsiveriotis and Chriss [1998].

First, two standard coordinate transformations are introduced,  $\mathbf{x} = \ln \mathbf{S}$  and  $\tau = \mathbf{T} - \mathbf{t}$  that recast Equation (3-4) to simple diffusion equations:

$$
\begin{array}{l} \frac {\partial u}{\partial \tau} = \frac {\sigma^ {2}}{2} \frac {\partial^ {2} u}{\partial x ^ {2}} + (r _ {g} - \frac {\sigma^ {2}}{2}) \frac {\partial u}{\partial S} - \\ \mathrm {r} (\mathrm {u} - \mathrm {v}) - \left(\mathrm {r} + \mathrm {r} _ {\mathrm {c}}\right) \mathrm {v} + \mathrm {f} (\mathrm {t}) \tag {13} \\ \end{array}
$$

$$
\frac {\partial \mathrm {v}}{\partial \tau} = \frac {\sigma^ {2}}{2} \frac {\partial^ {2} \mathrm {v}}{\partial \mathrm {x} ^ {2}} + \left(\mathrm {r} _ {\mathrm {g}} - \frac {\sigma^ {2}}{2}\right) \frac {\partial \mathrm {v}}{\partial \mathrm {S}} - \left(\mathrm {r} + \mathrm {r} _ {\mathrm {c}}\right) \mathrm {v} + \mathrm {f} (\mathrm {t}) \tag {14}
$$

Then the solution  $\mathbf{u}(\mathbf{x},\tau)$  and  $\mathbf{v}(\mathbf{x},\tau)$  is discretized on a set of grid points  $(\mathbf{x}_i,i = 1,\dots ,\mathbf{N})$  equally spaced at distance h from each other. Thus the unknowns become two N-dimensional vectors  $\mathbf{u}(\tau)$  and  $\mathbf{v}(\tau)$ . Furthermore, we take finite time steps  $\Delta \tau$ , and thus we deal only with  $\mathbf{u}^{\mathrm{k}} = \mathbf{u}(\mathrm{k}\Delta \tau)$  and  $\mathbf{v}^{\mathrm{k}} = \mathbf{v}(\mathrm{k}\Delta \tau)$ .

Using explicit time stepping, the partial differential equations are transformed to difference equations:

$$
\begin{array}{l} \frac {u _ {i} ^ {k + 1} - u _ {i} ^ {k}}{\Delta \tau} = \frac {\sigma^ {2}}{2} \frac {u _ {i + 1} ^ {k} - 2 u _ {i} ^ {k} + u _ {i - 1} ^ {k}}{h ^ {2}} + \\ \left(\mathrm {r} _ {\mathrm {g}} - \frac {\sigma^ {2}}{2}\right) \frac {\mathrm {u} _ {\mathrm {i} + 1} ^ {\mathrm {k}} - \mathrm {u} _ {\mathrm {i} - 1} ^ {\mathrm {k}}}{2 \mathrm {h}} - \\ \mathrm {r} \left(\mathrm {u} _ {\mathrm {i}} ^ {\mathrm {k}} - \mathrm {v} _ {\mathrm {i}} ^ {\mathrm {k}}\right) - \left(\mathrm {r} + \mathrm {r} _ {\mathrm {c}}\right) \mathrm {v} _ {\mathrm {i}} ^ {\mathrm {k}} + \mathrm {f} (\mathrm {k} \Delta \tau) \tag {15} \\ \end{array}
$$

$$
\begin{array}{l} \frac {v _ {i} ^ {k + 1} - v _ {i} ^ {k}}{\Delta \tau} = \frac {\sigma^ {2}}{2} \frac {v _ {i + 1} ^ {k} - 2 v _ {i} ^ {k} + u _ {i - 1} ^ {k}}{h ^ {2}} + \\ \left(\mathrm {r} _ {\mathrm {g}} - \frac {\sigma^ {2}}{2}\right) \frac {\mathrm {u} _ {\mathrm {i} + 1} ^ {\mathrm {k}} - \mathrm {u} _ {\mathrm {i} - 1} ^ {\mathrm {k}}}{2 \mathrm {h}} - \\ \left(\mathrm {r} + \mathrm {r} _ {\mathrm {c}}\right) \mathrm {v} _ {\mathrm {i}} ^ {\mathrm {k}} + \mathrm {f} (\mathrm {k} \Delta \tau) \tag {16} \\ \end{array}
$$

ExHIBIT 1 Example Convertible Bond  

<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Maturity</td><td>5 years</td></tr><tr><td>Conversion</td><td>0 to 5 years into 1 share</td></tr><tr><td>Conversion Ratio</td><td>1:0</td></tr><tr><td>Call</td><td>2 to 5 years at 110</td></tr><tr><td>Put</td><td>at year 3 at 105</td></tr><tr><td>Coupons</td><td>Coupon of 4 paid semiannually</td></tr></table>

The solution proceeds as follows. At time step  $k + 1$  (time  $\tau = (k + 1)\Delta \tau$ ), we start with  $(\mathbf{u}^{\mathrm{k}},\mathbf{v}^{\mathrm{k}})$ . Using Equation (15), we calculate  $\mathbf{u}^{k + 1}$  and then the conditions (7), (9), and (11) are applied to  $\mathbf{u}^{k + 1}$ . Next, Equation (16) is used to derive  $\mathbf{v}^{k + 1}$ , and then conditions (8), (10), and (12) are applied to  $\mathbf{v}^{k + 1}$ . This completes the method for each time step.

The final conditions (5) and (6) — which define the payoff — are used to find the starting values  $(\mathbf{u}^0, \mathbf{v}^0)$ .

Discrete cash flows such as coupons are simply added to the solution. In particular, each time we are within  $\Delta \tau$  from a coupon payment time  $\tau_{c}$  (i.e., if  $k\Delta \tau < \tau_{c} \leq (k + 1)\Delta \tau$ ), we temporarily adjust the time step to  $\Delta \tau' = (\tau_{c} - k\Delta \tau)$  so that the algorithm steps exactly on the coupon payment time. Then we 1) derive  $\mathbf{u}^{k+1}$ ,  $\mathbf{v}^{k+1}$  from  $\mathbf{u}^k$ ,  $\mathbf{v}^k$  as described, 2) add the coupon amount to  $\mathbf{u}^{k+1}$ ,  $\mathbf{v}^{k+1}$ , and 3) apply conditions (7-12) again on  $(\mathbf{u}^{k+1}, \mathbf{v}^{k+1})$ .

In other words, we treat coupon payments by taking a time step of zero length at each coupon payment time. Such zero-length time steps can have a finite effect on the solution only if the source term in the B-S equation is of the form  $\delta (\tau -\tau_{c})$  (where  $\tau_{c}$  is the coupon payment time), which is exactly how coupons are described in Equation (2).

Although we use explicit finite differences in our description for the sake of simplicity, the steps can be implemented using an implicit time stepping, which has the benefit of allowing larger time steps due to the unconditional stability of such schemes; for details see Morton and Mayers [1994].

# IV. EXAMPLE

To demonstrate our approach, we consider some realistic examples. The terms for the example are given

in Exhibit 1. We assume a flat  $5\%$  zero curve and graph the dirty price of the CB against stock price with varying spreads in Exhibit 2. Characteristics such as notice periods and call lags are ignored to keep the illustration simple.

To visualize the evolution of the solution from expiration to valuation date, we provide graphs of the CB and COCB values for a sequence of convertible bonds of increasing complexity. The zero CB (overall and cash-only versions) obtained by turning off call, put,

![](images/161796c0ba70170db8f0cb16d08a6c930da2444e97aaa9f2880491b8c31d642a.jpg)  
EXHIBIT 2 Effect of spread

![](images/ea763ec67eccebd30718557bc7957b5dddd63ac1350b2927b2add1e05ed9b694.jpg)

# ExHIBIT 3

# CO and COCB prices for Zero CB

![](images/25cdbb943c233bae7f881fc862f78dc09e22b451ead0375c585a426eacdef037.jpg)  
Zero CB  
Zero COCB

![](images/3dc535a95d428dc363000ebd15734c4cd3db1c69a8896231c29afc1eb91c0e6e.jpg)

and coupons in the convertible of Exhibit 1 are graphed in Exhibit 2. Corresponding graphs for the coupon paying convertible resulting from adding back coupons are in Exhibit 4. Graphs of the price of the callable convertible bond obtained by adding back the call feature are in Exhibit 5. Graphs are provided for the putable and callable convertible obtained by restoring the put feature in Exhibit 6.

The plots for the dirty price of the overall convertibles should be familiar. The zero convertible has a kinked profile at maturity that smooths out toward valuation date. The coupons add jumps to the dirty price at half-yearly intervals. The call feature removes time

# ExHIBIT 4

# CO and COCB prices for Coupon-Paying CB

![](images/1f95babef8062757e8efae1f228d62200cd2efc742232d209b8325608ab0373d.jpg)  
Coupon CB

![](images/6458f3dc243ec076bd6d740cff2002029759ca299b95fc6879132872b0d120bc.jpg)  
Coupon COCB

value during the call period, and thus maintains the kink in the profile throughout the call period. The put feature causes a jump in the dirty price for low stock prices at the put date.

In other words, the zero COCB profile at maturity has height zero or par, depending on whether it is converted into stock or not. The coupon-paying COCB profile includes the jumps due to the addition of the coupon at half-yearly intervals. Inclusion of the call feature forces the COCB value to zero in the region where the holder is forced to convert. Finally, the restoration of the put feature raises the COCB value to the put price at the stock prices at which the put is exercised.

# EXHIBIT 5 CO and COCB prices for Callable CB

![](images/91cbd6773ea29fc1d3ad61280399b6c1d730658cdb0d164ba78c5f3ed57578dc.jpg)

![](images/7ec378bafc14b3d444d8fe9b43ef619323a3b92fdb64bb7e20add100fbc393c5.jpg)

# V. SOME EXTENSIONS OF CREDIT CLASSES

Our aim has been to incorporate the issuer's debt spread into the pricing of the CB. More detailed spread information can also be incorporated by partitioning the inflows into more than two disjoint credit classes. For instance, if the spread to put differs from the spread to maturity on a zero CB with put at an intermediate time, three credit classes may be appropriate, with the additional credit class reserved for the cash flow of the put. In the basic formulation, spread was allowed to be a function of time, but additional credit classes permit spread to be a function of maturity as well.

Addition of a credit class results in an additional B-S equation of the type of Equation (4) and an extra

# EXHIBIT 6 CO and COCB prices for Callable and Putable CB

![](images/119e8cbd3b5dd937e824f87fb762d4b6f84ca9bc835aee1462f281a6fb6140b4.jpg)

![](images/446dcd60aea14e9a849f98adbaacfaffbfd72b3405b84c760b1863f629815a26.jpg)

term in the B-S Equation (3) that governs the entire CB. Note that this does not increase the number of free variables; i.e., the partial differential equations involved remain one-dimensional diffusion equations in the stock price S. The size of the solution increases linearly with the number of classes, so using credit classes based on time bucketing is a manageable problem.

In the examples we assume that the CB issuer and the CB holder use an optimal exercise policy. The same methodology can also be applied to asset stripping of CBs. Depending on contract details, the policy would be optimal for the holder of one of the credit classes rather than the CB holder. For instance, the policy may maximize value for the recipient of cash flows without regard to the value for the holder of the remaining flows.

# ENDNOTE

The authors thank James Bedell for motivating the search for a consistent treatment of credit risk in the context of asset stripping of convertible bonds.

# REFERENCES

Carrier, G.F., and C. E. Pearson. Partial Differential Equations: Theory and Technique. Orlando, FL: Academic Press, 1976.  
Morton, K.W., and D.F. Mayers. Numerical Solution of Partial Differential Equations. Oxford: Oxford University Press, 1994.

Nyborg, K.G. "The Use and Pricing of Convertible Bonds." Applied Mathematical Finance, 3 (1996), pp. 167-190.  
Tsiveriotis, K., and R.A. Brown. "Solution of Free-Boundary Problems using Finite-Elements/Newton Methods and Locally Refined Grids." International Journal for Numerical Methods in Fluids, 16 (1993), pp. 827-843.  
Tsiveriotis, K., and N. Chriss. "Pricing with a Difference." RISK, 11, No. 2 (1998), pp. 80-83.  
Willmott, P., J. Dewynne, and S. Howison. Option Pricing. Oxford: Oxford Financial Press, 1993.