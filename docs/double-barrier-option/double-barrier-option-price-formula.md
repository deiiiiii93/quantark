# 4.17.3 Double-Barrier Options

A double-barrier option is knocked either in or out if the underlying price touches the lower boundary $L$ or the upper boundary U prior to expiration. The formulas below pertain only to double knock-out options. The price of a double knock-in call is equal to the portfolio of a long standard call and a short double knock-out call, with identical strikes and time to expiration. Similarly, a double knock-in put is equal to a long standard put and a short double knock-out put. Doublebarrier options can be priced using the Ikeda and Kuintomo (1992) formula. 7

# Call Up-and-Out-Down-and-Out

Payoff: $c ( S , U , L , T ) = \operatorname* { m a x } ( S - X ; 0 )$ if $L < S < U$ before $T$ else 0.

$$
\begin{array}{l} c = S e ^ {(b - r) T} \sum_ {n = - \infty} ^ {\infty} \left\{\left(\frac {U ^ {n}}{L ^ {n}}\right) ^ {\mu_ {1}} \left(\frac {L}{S}\right) ^ {\mu_ {2}} [ N (d _ {1}) - N (d _ {2}) ] \right. \\ \left. - \left(\frac {L ^ {n + 1}}{U ^ {n} S}\right) ^ {\mu_ {3}} [ N (d _ {3}) - N (d _ {4}) ] \right\} \\ - X e ^ {- r T} \sum_ {n = - \infty} ^ {\infty} \left\{\left(\frac {U ^ {n}}{L ^ {n}}\right) ^ {\mu_ {1} - 2} \left(\frac {L}{S}\right) ^ {\mu_ {2}} \right. \\ \times \left[ N \left(d _ {1} - \sigma \sqrt {T}\right) - N \left(d _ {2} - \sigma \sqrt {T}\right) \right] \\ \left. - \left(\frac {L ^ {n + 1}}{U n S}\right) ^ {\mu_ {3} - 2} [ N (d _ {3} - \sigma \sqrt {T}) - N (d _ {4} - \sigma \sqrt {T}) ] \right\}, \tag {4.57} \\ \end{array}
$$

where

$$
\begin{array}{l} d _ {1} = \frac {\ln (S U ^ {2 n} / (X L ^ {2 n})) + (b + \sigma^ {2} / 2) T}{\sigma \sqrt {T}} \\ d _ {2} = \frac {\ln (S U ^ {2 n} / (F L ^ {2 n})) + (b + \sigma^ {2} / 2) T}{\sigma \sqrt {T}} \\ d _ {3} = \frac {\ln \left(L ^ {2 n + 2} / \left(X S U ^ {2 n}\right)\right) + \left(b + \sigma^ {2} / 2\right) T}{\sigma \sqrt {T}} \\ d _ {4} = \frac {\ln (L ^ {2 n + 2} / (F S U ^ {2 n})) + (b + \sigma^ {2} / 2) T}{\sigma \sqrt {T}}, \\ \end{array}
$$

# 4.17. BARRIER OPTIONS

$$
\mu_ {1} = \frac {2 [ b - \delta_ {2} - n (\delta_ {1} - \delta_ {2}) ]}{\sigma^ {2}} + 1 \quad \mu_ {2} = 2 n \frac {\left(\delta_ {1} - \delta_ {2}\right)}{\sigma^ {2}}
$$

$$
\mu_ {3} = \frac {2 [ b - \delta_ {2} + n (\delta_ {1} - \delta_ {2}) ]}{\sigma^ {2}} + 1 \qquad F = U e ^ {\delta_ {1} T},
$$

where $\delta _ { 1 }$ and $\delta _ { 2 }$ determine the curvature $L$ and U. The case of

1. $\delta _ { 1 } = \delta _ { 2 } = 0$ corresponds to two flat boundaries.   
2. $\delta _ { 1 } < 0 < \delta _ { 2 }$ 2 corresponds to a lower boundary exponentially growing as time elapses, while the upper boundary will be exponentially decaying.   
3. $\delta _ { 1 } > 0 > \delta _ { 2 }$ corresponds to a convex downward lower boundary and a convex upward upper boundary

# Put Up-and-Out-Down-and-Out

Payoff: $p ( S , U , L , T ) = \operatorname* { m a x } ( X - S ; 0 )$ if $L < S < U$ before $T$ else 0.

$$
\begin{array}{l} p = + X e ^ {- r T} \sum_ {n = - \infty} ^ {\infty} \left\{\left(\frac {U ^ {n}}{L ^ {n}}\right) ^ {\mu_ {1} - 2} \left(\frac {L}{S}\right) ^ {\mu_ {2}} \right. \\ \times \left[ N \left(y _ {1} - \sigma \sqrt {T}\right) - N \left(y _ {2} - \sigma \sqrt {T}\right) \right] \\ \left. - \left(\frac {L ^ {n + 1}}{U ^ {n} S}\right) ^ {\mu_ {3} - 2} [ N (y _ {3} - \sigma \sqrt {T}) - N (y _ {4} - \sigma \sqrt {T}) ] \right\} \\ - S e ^ {(b - r) T} \sum_ {n = - \infty} ^ {\infty} \left\{\left(\frac {U ^ {n}}{L ^ {n}}\right) ^ {\mu_ {1}} \left(\frac {L}{S}\right) ^ {\mu_ {2}} [ N (y _ {1}) - N (y _ {2}) ] \right. \\ \left. - \left(\frac {L ^ {n + 1}}{U ^ {n} S}\right) ^ {\mu_ {3}} [ N (y _ {3}) - N (y _ {4}) ] \right\}, \tag {4.58} \\ \end{array}
$$

where

$$
y _ {1} = \frac {\ln (S U ^ {2 n} / \left(E L ^ {2 n}\right)) + (b + \sigma^ {2} / 2) T}{\sigma \sqrt {T}}
$$

$$
y _ {2} = \frac {\ln (S U ^ {2 n} / (X L ^ {2 n})) + (b + \sigma^ {2} / 2) T}{\sigma \sqrt {T}}
$$

$$
y _ {3} = \frac {\ln \left(L ^ {2 n + 2} / \left(E S U ^ {2 n}\right)\right) + (b + \sigma^ {2} / 2) T}{\sigma \sqrt {T}}
$$

$$
\begin{array}{l} y _ {4} = \frac {\ln \left(L ^ {2 n + 2} / \left(X S U ^ {2 n}\right)\right) + (b + \sigma^ {2} / 2) T}{\sigma \sqrt {T}} \\ E = L e ^ {\delta_ {2} T} \\ \end{array}
$$

The double-barrier options are expressed as infinite series of weighted normal distribution functions. However, numerical studies

TABLE 4-15   

<table><tr><td colspan="10">Examples of Call Up-and-Out-Down-and-Out Values (S = 100, X = 100, r = 0.1, b = 0.1)</td></tr><tr><td></td><td></td><td></td><td></td><td colspan="3">T = 0.25</td><td colspan="3">T = 0.5</td></tr><tr><td>L</td><td>U</td><td>δ1</td><td>δ2</td><td>σ = 0.15</td><td>σ = 0.25</td><td>σ = 0.35</td><td>σ = 0.15</td><td>σ = 0.25</td><td>σ = 0.35</td></tr><tr><td>50</td><td>150</td><td>0</td><td>0</td><td>4.3515</td><td>6.1644</td><td>7.0373</td><td>6.9853</td><td>7.9336</td><td>6.5088</td></tr><tr><td>60</td><td>140</td><td>0</td><td>0</td><td>4.3505</td><td>5.8500</td><td>5.7726</td><td>6.8082</td><td>6.3383</td><td>4.3841</td></tr><tr><td>70</td><td>130</td><td>0</td><td>0</td><td>4.3139</td><td>4.8293</td><td>3.7765</td><td>5.9697</td><td>4.0004</td><td>2.2563</td></tr><tr><td>80</td><td>120</td><td>0</td><td>0</td><td>3.7516</td><td>2.6387</td><td>1.4903</td><td>3.5805</td><td>1.5098</td><td>0.5635</td></tr><tr><td>90</td><td>110</td><td>0</td><td>0</td><td>1.2055</td><td>0.3098</td><td>0.0477</td><td>0.5537</td><td>0.0441</td><td>0.0011</td></tr><tr><td>50</td><td>150</td><td>-0.1</td><td>0.1</td><td>4.3514</td><td>6.0997</td><td>6.6987</td><td>6.8974</td><td>6.9821</td><td>5.2107</td></tr><tr><td>60</td><td>140</td><td>-0.1</td><td>0.1</td><td>4.3478</td><td>5.6351</td><td>5.2463</td><td>6.4094</td><td>5.0199</td><td>3.1503</td></tr><tr><td>70</td><td>130</td><td>-0.1</td><td>0.1</td><td>4.2558</td><td>4.3291</td><td>3.1540</td><td>4.8182</td><td>2.6259</td><td>1.3424</td></tr><tr><td>80</td><td>120</td><td>-0.1</td><td>0.1</td><td>3.2953</td><td>1.9868</td><td>1.0351</td><td>1.9245</td><td>0.6455</td><td>0.1817</td></tr><tr><td>90</td><td>110</td><td>-0.1</td><td>0.1</td><td>0.5887</td><td>0.1016</td><td>0.0085</td><td>0.0398</td><td>0.0002</td><td>0.0000</td></tr><tr><td>50</td><td>150</td><td>0.1</td><td>-0.1</td><td>4.3515</td><td>6.2040</td><td>7.3151</td><td>7.0086</td><td>8.6080</td><td>7.7218</td></tr><tr><td>60</td><td>140</td><td>0.1</td><td>-0.1</td><td>4.3512</td><td>5.9998</td><td>6.2395</td><td>6.9572</td><td>7.4267</td><td>5.6620</td></tr><tr><td>70</td><td>130</td><td>0.1</td><td>-0.1</td><td>4.3382</td><td>5.2358</td><td>4.3859</td><td>6.6058</td><td>5.3761</td><td>3.3446</td></tr><tr><td>80</td><td>120</td><td>0.1</td><td>-0.1</td><td>4.0428</td><td>3.2872</td><td>2.0048</td><td>5.0718</td><td>2.6591</td><td>1.1871</td></tr><tr><td>90</td><td>110</td><td>0.1</td><td>-0.1</td><td>1.9229</td><td>0.6451</td><td>0.1441</td><td>1.7079</td><td>0.3038</td><td>0.0255</td></tr></table>

show that the convergence of the formulas is rapid. The numerical study of Ikeda and Kuintomo (1992) suggests that it suffices to calculate the leading two or three terms for most cases. The Ikeda and Kuntomo formula only holds when the strike price is inside the barrier range. For double barrier options when the strike is outside the barrier range see section on "Double-Barrier Option Using Barrier Symmetry."

Table 4-15 gives examples of call-up-and-out-down-and-out option values for different choices of lower $L$ and upper $U$ barrier, barrier curvatures $\delta _ { 1 }$ and $\delta _ { 1 }$ , volatility $\sigma$ , and time to maturity $T$ .
