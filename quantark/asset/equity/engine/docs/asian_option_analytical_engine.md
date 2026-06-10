# 4.20 ASIAN OPTIONS

Asian options are especially popular in the energy over the counter (OTC) market and many commodity markets. An average is less volatile than the underlying asset itself and will lower the price of an average-rate option compared with a similar standard option.[12] If the option is based on an average, an attempt to manipulate the asset price just before expiration will normally have little or no effect on the option's value. Asian options should therefore be of particular interest in markets for thinly traded assets.

# 4.20.1 Geometric Average-Rate Options

If the underlying asset is assumed to be lognormally distributed, the geometric average  $((x_{1}\dots x_{n})^{1 / n})$  of the asset will itself be lognormally distributed.

# Geometric Continuous Average-Rate Options

As originally shown by Kemna and Vorst (1990), the geometric average option can be priced as a standard option by changing the volatility and cost-of-carry term:

$$
c = S e ^ {(b _ {A} - r) T} N \left(d _ {1}\right) - X e ^ {- r T} N \left(d _ {2}\right) \tag {4.91}
$$

$$
p = X e ^ {- r T} N (- d _ {2}) - S e ^ {(b _ {A} - r) T} N (- d _ {1}), \tag {4.92}
$$

where

$$
d _ {1} = \frac {\ln (S / X) + (b _ {A} + \sigma_ {A} ^ {2} / 2) T}{\sigma_ {A} \sqrt {T}} \quad d _ {2} = d _ {1} - \sigma_ {A} \sqrt {T},
$$

and the adjusted volatility is equal to

$$
\sigma_ {A} = \frac {\sigma}{\sqrt {3}}
$$

Moreover, the adjusted cost-of-carry is set to

$$
b _ {A} = \frac {1}{2} \left(b - \frac {\sigma^ {2}}{6}\right).
$$

# Example

What is the value of a geometric average-rate put option with three months to maturity? The strike is 85, the asset price is 80, the risk-free rate is  $5\%$ , the cost-of-carry is  $8\%$ , and the volatility is  $20\%$ .  $S = 80$ ,  $X = 85$ ,  $T = 0.25$ ,  $r = 0.05$ ,  $b = 0.08$ , and  $\sigma = 0.2$ .

$$
\sigma_ {A} = \frac {0 . 2}{\sqrt {3}} = 0. 1 1 5 5 \quad b _ {A} = \frac {1}{2} \left(0. 0 8 - \frac {0 . 2 ^ {2}}{6}\right) = 0. 0 3 6 6
$$

$$
d _ {1} = \frac {\ln (8 0 / 8 5) + (0 . 0 3 6 6 + 0 . 1 1 5 5 ^ {2} / 2) 0 . 2 5}{0 . 1 1 5 5 \sqrt {0 . 2 5}} = - 0. 8 6 2 4
$$

$$
d _ {2} = d _ {1} - 0. 1 1 5 5 \sqrt {0 . 2 5} = - 0. 9 2 0 1
$$

$$
N (- d _ {1}) = N (0. 8 6 2 4) = 0. 8 0 5 8 \quad N (- d _ {2}) = N (0. 9 2 0 1) = 0. 8 2 1 3
$$

$$
p = 8 5 e ^ {- 0. 0 5 \times 0. 2 5} N (- d _ {2}) - 8 0 e ^ {(0. 0 3 6 6 - 0. 0 5) 0. 2 5} N (- d _ {1}) = 4. 6 9 2 2
$$

The value of a similar standard European put option is 5.2186.

# Geometric Discrete Average-Rate Options

In practice, all Asian options have discrete monitoring of the average. We now show how to value geometric average-rate options with discrete monitoring. We will present the more general case where we

can calibrate the geometric average volatility to a term structure of implied volatilities from plain vanilla options. We are thus assuming a spot rate process with time dependent deterministic volatility:

$$
d S _ {t} = \mu S _ {t} d t + v _ {t} S _ {t} d z _ {t}
$$

The formula for the geometric average volatility, described in detail by Haug, Haug, and Margrabe (2003), is

$$
\sigma_ {G} ^ {2} = \frac {1}{n ^ {3}} \sum_ {i = 0} ^ {n - 1} (n - i) ^ {2} v _ {i} ^ {2} \tag {4.93}
$$

where  $v_{i} \equiv v_{t_{i}}$  is the local volatility between each fixing. For each time step, we need the local volatility. The local implied forward volatilities can be computed from global implied BSM volatilities by the formula

$$
v _ {i} = \sqrt {\frac {\sigma_ {i} ^ {2} t _ {i} - \sigma_ {i - 1} ^ {2} t _ {i - 1}}{t _ {i} - t _ {i - 1}}},
$$

where  $\sigma_{i}$  is the implied global volatility for an option expiring at time  $t_i$ , and  $\sigma_{i-1}$  is the implied volatility for an option expiring at time  $t_{i-1} < t_i$ .

# Computer algorithm

Function GeometricVolFromLocalVolTermStructure(v As Object, n As Long) As Double

Dim sum As Double

Dim i As Long

For  $\mathrm{i} = 0$  To n-1

$$
\operatorname {s u m} = \operatorname {s u m} + v (i + 1) ^ {\wedge} 2 * (n - i) ^ {\wedge} 2
$$

Next

GeometricVolFromLocalVolTermStructure = Sqr(sum / n^3)

# End Function

Alternatively, we can find the Asian geometric volatility directly from the plain vanilla global volatilities shown by Levy (1997):

$$
\sigma_ {G} ^ {2} = \frac {1}{n ^ {2} T} \left[ \sum_ {i = 1} ^ {n} \sigma_ {i} ^ {2} t _ {i} + 2 \sum_ {i = 1} ^ {n - 1} (n - i) \sigma_ {i} ^ {2} t _ {i} \right], \tag {4.94}
$$

where  $\sigma_{i}$  now is the implied BSM global volatility from an option that expires at  $t_i$ , and  $t_i$  is the time to fixing  $i$ .

# Computer algorithm

Function GeometricVolFromGlobalVol(T As Double, v As Object, n As Long) As Double

Dim sum As Double, dt As Double

Dim i As Long

dt  $\equiv$  T/n

For i = 1 To n - 1

$$
\operatorname {s u m} = \operatorname {s u m} + v (i) ^ {\wedge} 2 * d t * i + 2 * (n - i) * v (i) ^ {\wedge} 2 * d t * i
$$

Next

$$
\operatorname {s u m} = \operatorname {s u m} + v (n) ^ {\wedge} 2 * T
$$

GeometricVolFromGlobalVol = Sqr(sum / (n^2 * T))

# End Function

Formulas (4.93) and (4.94) both yield the same result; the only difference is that one of them takes local volatilities as input, while the other takes global volatilities as input. As global volatilities are the ones observable in the market, formula (4.94) seems to be the most practical—saving you some calculations.

The value of geometric average options that are calibrated to the term structure can now be computed with the BSM formula:

$$
c = S e ^ {(b _ {G} - r) T} N \left(d _ {1}\right) - X e ^ {- r T} N \left(d _ {2}\right), \tag {4.95}
$$

where  $X$  is the strike price,  $N(\cdot)$  is the cumulative normal distribution,

$$
d _ {1} = \frac {\ln (S / X) + (b _ {G} + \sigma_ {G} ^ {2} / 2) T}{\sigma_ {G} \sqrt {T}}
$$

and

$$
d _ {2} = d _ {1} - \sigma_ {G} \sqrt {T}
$$

This is the BSM formula where we have replaced the volatility with  $\sigma_G$ , and the cost-of-carry with

$$
b _ {G} = \frac {\sigma_ {G} ^ {2}}{2} + \frac {1}{n T} \sum_ {i = 1} ^ {n} (b - \sigma_ {i} ^ {2} / 2) t _ {i}
$$

Here  $t_i$  is the time to each fixing,  $T$  is the time to maturity, and  $\sigma_i$  is the global BSM volatility for an option with expiration  $t_i$ .

# Variable Time between Fixings

We have so far assumed equal time between fixings. In real applications the time between fixings can vary. Consider the case of daily fixings. Most markets are closed on weekends, which results in

longer time periods over weekends. The formula for calculation of the variance for a geometric average with variable time between fixings is

$$
\sigma_ {G} ^ {2} = \frac {1}{n ^ {2} T} \sum_ {i = 0} ^ {n - 1} (n - i) ^ {2} \Delta t _ {i} v _ {i} ^ {2} \tag {4.96}
$$

We have simply assigned a variable time to each fixing,  $\Delta t_{i}$

# 4.20.2 Arithmetic Average-Rate Options

It is not possible (or very hard) to find a closed-form solution for the value of options on an arithmetic average  $\left(\frac{x_1 + \cdots + x_n}{n}\right)$ . The main reason is that when the asset is assumed to be lognormally distributed, the arithmetic average will not itself have a lognormal distribution. Arithmetic average rate options can be priced by analytical approximations, as presented below, or with Monte Carlo simulations, presented in Chapter 8.

# The Turnbull and Wakeman Approximation

The approximation formula below is based on the work of Turnbull and Wakeman (1991). The approximation adjusts the mean and variance so that they are consistent with the exact moments of the arithmetic average. The adjusted mean,  $b_{A}$ , and variance,  $\sigma_{A}^{2}$ , are then used as input in the generalized BSM formula:

$$
c \approx S e ^ {(b _ {A} - r) T} N (d _ {1}) - X e ^ {- r T} N (d _ {2}) \tag {4.97}
$$

$$
p \approx X e ^ {- r T} N (d _ {2}) - S e ^ {(b _ {A} - r) T} N (d _ {1}) \tag {4.98}
$$

$$
d _ {1} = \frac {\ln (S / X) + (b _ {A} + \sigma_ {A} ^ {2} / 2) T}{\sigma_ {A} \sqrt {T}} d _ {2} = d _ {1} - \sigma_ {A} \sqrt {T},
$$

where  $T$  is the time to maturity in years. The volatility and the cost-of-carry of the average are given by

$$
\sigma_ {A} = \sqrt {\frac {\ln (M _ {2})}{T} - 2 b _ {A}}
$$

$$
b _ {A} = \frac {\ln (M _ {1})}{T}
$$

The exact first and second moments of the arithmetic average are

$$
M _ {1} = \frac {e ^ {b T} - e ^ {b t _ {1}}}{b (T - t _ {1})}
$$

$$
M _ {2} = \frac {2 e ^ {(2 b + \sigma^ {2}) T}}{(b + \sigma^ {2}) (2 b + \sigma^ {2}) (T - t _ {1}) ^ {2}} + \frac {2 e ^ {(2 b + \sigma^ {2}) t _ {1}}}{b (T - t _ {1}) ^ {2}} \left[ \frac {1}{2 b + \sigma^ {2}} - \frac {e ^ {b (T - t _ {1})}}{b + \sigma^ {2}} \right],
$$

where in the case of  $b = 0$  we have

$$
M _ {1} = 1
$$

$$
M _ {2} = \frac {2 e ^ {\sigma^ {2} T} - 2 e ^ {\sigma^ {2} t _ {1}} [ 1 + \sigma^ {2} (T - t _ {1}) ]}{\sigma^ {4} (T - t _ {1}) ^ {2}},
$$

where  $t_1$  is the time to the beginning of the average period. If the option is into the average period, the strike price must be replaced by  $\hat{X}$ , and the option value must be multiplied by  $\frac{T_2}{T}$ , where

$$
\hat {X} = \frac {T _ {2}}{T} X - \frac {\tau}{T} S _ {A},
$$

where  $S_A$  is the average asset price during the realized or observed time period so far.  $\tau$  is the reminding time in the average period  $\tau = T_2 - T$ .

If we are into the average period,  $\tau > 0$ , and  $\frac{T_2}{T} X - \frac{\tau}{T} S_A < 0$ , then a call option will for certain be exercised and is equal to the expected value of the average at maturity minus the strike price  $e^{-rT}(E[S_A] - X)$ . The expected average at maturity is equal to<sup>14</sup>

$$
E \left[ S _ {A} \right] = S _ {A} \frac {T _ {2} - T}{T _ {2}} + S M _ {1} \frac {T}{T _ {2}}
$$

The put will in this case for certain not be in-the-money and will have value zero.

# Computer algorithm

The computer code below calculates an adjusted cost-of-carry term,  $b_{A}$ , and volatility,  $v_{A}$ , and then calls the general BSM formula described in Chapter 1.

```txt
' // CallPutFlag = "c" for call and "p" for put option  
' // S = Asset price  
' // SA= Realized average so far  
' // X = Strike price  
' // t1 = Time to start of average period in years  
' // T = Time to maturity in years of option T  
' // T2 = Original time in average period in years,  
' // constant over life of option  
' // r = risk-free rate  
' // b = cost-of-carry underlying asset can be positive and negative  
' // v = annualized volatility of asset price
```

Dim m1 As Double, m2 As Double, tau As Double, t1 As Double  
Dim bA As Double, vA As Double

//tau:remindingtimeofaverageperios

```latex
$\begin{array}{rl}{\mathbf{t1}}&{=\operatorname*{Max}(\mathbf{\Delta 0},\mathbf{T}-\mathbf{\Delta T2})}\\{\mathbf{tau}}&{=\mathbf{T2}-\mathbf{\Delta T}}\end{array}$
```

```txt
If  $\mathbf{b} = 0$  Then
[ \mathbf{m1} = 1 ]
```

Else

```latex
$\mathbf{m1} = (\mathbf{Exp}(\mathbf{b}*\mathbf{T}) - \mathbf{Exp}(\mathbf{b}*\mathbf{t1})) / (\mathbf{b}*(\mathbf{T} - \mathbf{t1}))$  End If
```

'//Take into account when option will be exercised If tau > 0 Then

```vba
If T2 / T * X - tau / T * SA < 0 Then
    If CallPutFlag = "c" Then
        ' // Expected average at maturity:
        SA = SA * (T2 - T) / T2 + S * m1 * T / T2
        TurnbullWakemanAsian = Max(0, SA - X) * Exp(-r * T)
    Else
        TurnbullWakemanAsian = 0
End If
Exit Function
```

End If End If

'// Extended to hold for options on futures May 16, 1999 Espen G. Haug If  $\mathbf{b} = 0$  Then

```txt
m2 = 2 * Exp(v * v * T) / (v^4 * (T - t1)^2) - 2 * Exp(v * v * t1) * (1 + v * v * (T - t1)) / (v^4 * (T - t1)^2)
```

Else

```latex
$\mathbf{m}2 = 2*\mathbf{Exp}((2*b + \mathbf{v}*\mathbf{v})*\mathbf{T}) / ((\mathbf{b} + \mathbf{v}*\mathbf{v})\_ * (2*b + \mathbf{v}*\mathbf{v}) * (\mathbf{T} - t1)^{\wedge}2)$ $+2*\mathbf{Exp}((2*b + \mathbf{v}*\mathbf{v})*t1) / (\mathbf{b}*(T - t1)^{\wedge}2)$
```

```latex
$\begin{array}{rl} & {\mathrm{*}(1 / (2*b + v*v) - Exp(b*(T - t1)) / (b + v*v))}\\ & {\mathrm{End~If}}\\ & {\mathrm{bA = Log(m1) / T}}\\ & {\mathrm{vA = Sqr(Log(m2) / T - 2*bA)}}\\ & {\mathrm{If~tau > 0~Then}}\\ & {\mathrm{X = T2 / T * X - tau / T * SA}}\\ & {\mathrm{TurnbullWakemanAsian = GBlackScholes(CallPutFlag, S, X, T, r, _}}\\ & {\mathrm{bA, vA) * T / T2}}\\ & {\mathrm{Else}}\\ & {\mathrm{TurnbullWakemanAsian = GBlackScholes(CallPutFlag, S, X, T, r, bA, vA)}}\\ & {\mathrm{End~If}} \end{array}$
```

# End Function

Example: TurnbullWakemanAsian("p", 90, 88, 95, 0, 0.25, 0.25, 0.07, 0.02, 0.25) will return an arithmetic average put value of 5.6093.

# Asian Futures Options

In the case of Asian options on futures, only the formulas above can be simplified. If we assume the arithmetic average is approximately lognormally distributed, all we need to value an Asian futures option is to adjust the volatility of the Black-76 formula. This entails replacing the futures volatility  $\sigma$  with the volatility of the average on the futures  $\sigma_{A}$ :

$$
c _ {A} \approx e ^ {- r T} \left[ F N \left(d _ {1}\right) - X N \left(d _ {2}\right) \right] \tag {4.99}
$$

$$
p _ {A} \approx e ^ {- r T} [ X N (- d _ {2}) - F N (- d _ {1})) ], \tag {4.100}
$$

where  $T$  is the time to maturity,  $r$  is the risk-free rate,  $F$  is the futures price, and  $X$  is the strike price.

$$
d _ {1} = \frac {\ln (F / X) + T \sigma_ {A} ^ {2} / 2}{\sigma_ {A} \sqrt {T}}, \qquad d _ {2} = d _ {1} - \sigma_ {A} \sqrt {T},
$$

where

$$
\sigma_ {A} = \sqrt {\frac {\ln (M)}{T}} \qquad M = \frac {2 e ^ {\sigma^ {2} T} - 2 e ^ {\sigma^ {2} \tau} [ 1 + \sigma^ {2} (T - \tau) ]}{\sigma^ {4} (T - \tau) ^ {2}},
$$

where  $\tau$  is the time to the beginning of the average period. If the option is into the average period, the strike price must be replaced by  $\hat{X}$  and the option value must be multiplied by  $\frac{T}{T_2}$ , where

$$
\hat {X} = X \frac {T _ {2}}{T} - F _ {A} \frac {\left(T _ {2} - T\right)}{T},
$$

where  $T_{2}$  is the original time in the average period and  $F_{A}$  is the average futures price during the realized or observed time period  $T_{2} - T$ .

If  $\hat{X}$  should be negative, the call option will for sure be exercised at maturity and the value becomes the discounted value of the expected

average at maturity  $E_{Q}[A]$  minus the strike price:  $E_{Q}[A] - X$ . The expected average is equal to

$$
E _ {Q} [ A ] = \frac {F _ {A} (T _ {2} - T)}{T _ {2}} + F \frac {T}{T _ {2}}
$$

For a put, the value will be 0 if  $\hat{X}$  should be negative. This is basically the Turnbull-Wakeman formula extended to Asian options on futures.

# Levy's Approximation

An alternative to the Turnbull and Wakeman formula is the Levy (1992) Asian option approximation:

$$
c _ {A s i a n} \approx S _ {E} N \left(d _ {1}\right) - X ^ {*} e ^ {- r T _ {2}} N \left(d _ {2}\right), \tag {4.101}
$$

where

$$
S _ {E} = \frac {S}{T b} (e ^ {(b - r) T _ {2}} - e ^ {- r T _ {2}})
$$

$$
d _ {1} = \frac {1}{\sqrt {V}} \left[ \frac {\ln (D)}{2} - \ln (X ^ {*}) \right] \quad d _ {2} = d _ {1} - \sqrt {V}
$$

$$
X ^ {*} = X - \frac {T - T _ {2}}{T} S _ {A} \quad V = \ln (D) - 2 \left[ r T _ {2} + \ln \left(S _ {E}\right) \right] \quad D = \frac {M}{T ^ {2}}
$$

$$
M = \frac {2 S ^ {2}}{b + \sigma^ {2}} \left[ \frac {e ^ {(2 b + \sigma^ {2}) T _ {2}} - 1}{2 b + \sigma^ {2}} - \frac {e ^ {b T _ {2}} - 1}{b} \right]
$$

The Asian put value can be found by using the following put-call parity:

$$
p _ {\text {A s i a n}} = c _ {\text {A s i a n}} - S _ {E} + X ^ {*} e ^ {- r T _ {2}}
$$

where

$S_A =$  Arithmetic average of the known asset price fixings.

$S =$  Asset price.

$X =$  Strike price of option.

$r =$  Risk-free interest rate.

$b =$  Cost-of-carry rate.

$T_{2} =$  Remaining time to maturity.

$T =$  Original time to maturity.

$\sigma =$  Volatility of natural logarithms of return of the underlying asset.

The formula does not allow for  $b = 0$ . Table 4-25 illustrates this.[15]

TABLE 4-25  

<table><tr><td colspan="7">Examples of Arithmetic Average Call Option Values (S = SA = 100, T2 = 0.75, r = 0.1, b = 0.05)</td></tr><tr><td></td><td colspan="3">σ = 0.15</td><td colspan="3">σ = 0.35</td></tr><tr><td>X</td><td>T = 0.75</td><td>T = 0.5</td><td>T = 0.25</td><td>T = 0.75</td><td>T = 0.5</td><td>T = 0.25</td></tr><tr><td colspan="7">Turnbull and Wakeman Approximation</td></tr><tr><td>95</td><td>7.0544</td><td>5.6731</td><td>5.0806</td><td>10.1213</td><td>6.9705</td><td>5.1411</td></tr><tr><td>100</td><td>3.7845</td><td>1.9964</td><td>0.6722</td><td>7.5038</td><td>4.0687</td><td>1.4222</td></tr><tr><td>105</td><td>1.6729</td><td>0.3565</td><td>0.0004</td><td>5.4071</td><td>2.1359</td><td>0.1552</td></tr><tr><td colspan="7">Levy&#x27;s Approximation</td></tr><tr><td>95</td><td>7.0544</td><td>5.6731</td><td>5.0806</td><td>10.1213</td><td>6.9705</td><td>5.1411</td></tr><tr><td>100</td><td>3.7845</td><td>1.9964</td><td>0.6722</td><td>7.5038</td><td>4.0687</td><td>1,4222</td></tr><tr><td>105</td><td>1.6729</td><td>0.3565</td><td>0.0004</td><td>5.4071</td><td>2.1359</td><td>0.1552</td></tr></table>

# Example

Consider an arithmetic average currency option with a time to expiration of six months. The spot price is 6.80, the strike is 6.90, the domestic risk-free interest rate is  $7\%$  per year, the foreign interest rate is  $9\%$  per year, and the volatility of the spot rate is  $14\%$ . The option is on the average of the next six months.  $S = 6.80, S_A = 6.80, X = 6.90, T = 0.5, T_2 = 0.5, r = 0.07, b = r - r_f = 0.07 - 0.09 = -0.02,$  and  $\sigma = 0.14$ .

$$
\begin{array}{l} S _ {E} = \frac {6 . 8}{0 . 5 (- 0 . 0 2)} \left(e ^ {(- 0. 0 2 - 0. 0 7) \times 0. 5} - e ^ {- 0. 0 7 \times 0. 5}\right) = 6. 5 3 3 4 \\ X ^ {*} = 6. 9 0 - \frac {0 . 5 - 0 . 5}{0 . 5} 6. 8 0 = 6. 9 0 0 0 \\ M = \frac {2 \times 6 . 8 0 ^ {2}}{- 0 . 0 2 + 0 . 1 4 ^ {2}} \\ \times \left[ \frac {e ^ {(2 (- 0 . 0 2) + 0 . 1 4 ^ {2}) 0 . 5} - 1}{2 (- 0 . 0 2) + 0 . 1 4 ^ {2}} - \frac {e ^ {(- 0 . 0 2) 0 . 5} - 1}{- 0 . 0 2} \right] = 1 1. 4 8 2 5 \\ D = \frac {1 1 . 4 8 2 5}{0 . 5 ^ {2}} = 4 5. 9 2 9 8 \\ \end{array}
$$

$$
V = \ln (4 5. 9 2 9 8) - 2 [ 0. 0 7 \times 0. 5 + \ln (6. 5 3 3 4) ] = 0. 0 0 3 3
$$

$$
d _ {1} = \frac {1}{\sqrt {0 . 0 0 3 3}} \left[ \frac {\ln (4 5 . 9 2 9 8)}{2} - \ln (6. 9 0 0 0) \right] = - 0. 3 1 4 6
$$

$$
d _ {2} = d _ {1} - \sqrt {0 . 0 0 3 3} = - 0. 3 7 1 7
$$

$$
\begin{array}{l} N \left(d _ {1}\right) = N (- 0. 3 1 4 6) = 0. 3 7 6 5 \quad N \left(d _ {2}\right) = N (- 0. 3 7 1 7) = 0. 3 5 5 1 \\ c \approx 6. 5 3 3 4 N \left(d _ {1}\right) - 6. 9 0 0 0 e ^ {- 0. 0 7 \times 0. 5} N \left(d _ {2}\right) \approx 0. 0 9 4 4 \\ p \approx 0. 0 9 4 4 - 6. 5 3 3 4 + 6. 9 0 0 0 e ^ {- 0. 0 7 \times 0. 5} \approx 0. 2 2 3 7 \\ \end{array}
$$

# 4.20.3 Discrete Arithmetic Average-Rate Options

In practice, all traded Asian options have discrete fixings of their average, for instance, every day or week. We next present several approximations for discrete average Asian options. The first method is basically a discrete average version of the Turnbull-Wakeman formula. The next method is the Curran approximation. Both of these implementations assume a flat term structure of volatility for plain vanilla options. In practice, there is often an upward- or downward-sloping volatility term structure. The last method implements a volatility term structure.

# Discrete Asian Approximation

The value of a Asian call can be valued as (see Levy, 1997, and Haug, Haug, and Margrabe, 2003)

$$
c _ {A} \approx e ^ {- r T} \left[ F _ {A} N \left(d _ {1}\right) - X N \left(d _ {2}\right) \right], \tag {4.102}
$$

and the value of a Asian put as

$$
p _ {A} \approx e ^ {- r T} X N (- d _ {2}) - \left[ F _ {A} N (- d _ {1}) \right], \tag {4.103}
$$

where

$$
\begin{array}{l} d _ {1} = \frac {\ln \left(F _ {A} / X\right) + T \sigma_ {A} ^ {2} / 2}{\sigma_ {A} \sqrt {T}} \\ d _ {2} = d _ {1} - \sigma_ {A} \sqrt {T} \\ \end{array}
$$

$F_{A}$  is defined as  $E[A_T]$ , and

$$
\begin{array}{l} \sigma_ {A} = \sqrt {\frac {\ln (E [ A _ {T} ^ {2} ]) - 2 \ln (E [ A _ {T} ])}{T}} \\ E [ A _ {T} ] = \frac {S}{n} e ^ {b t _ {1}} \frac {1 - e ^ {b h n}}{1 - e ^ {b h}} \\ \end{array}
$$

and

$$
\begin{array}{l} E \left[ A _ {T} ^ {2} \right] = \frac {S ^ {2} e ^ {(2 b + \sigma^ {2}) t _ {1}}}{n ^ {2}} \left[ \frac {1 - e ^ {(2 b + \sigma^ {2}) h n}}{1 - e ^ {(2 b + \sigma^ {2}) h}} \right. \\ \left. + \frac {2}{1 - e ^ {(b + \sigma^ {2}) h}} \left(\frac {1 - e ^ {b h n}}{1 - e ^ {b h}} - \frac {1 - e ^ {(2 b + \sigma^ {2}) h n}}{1 - e ^ {(2 b + \sigma^ {2}) h}}\right) \right], \\ \end{array}
$$

where  $h = \frac{T - t_1}{n - 1}$ . In the case of  $b = 0$  we have

$$
E \left[ A _ {T} \right] = S
$$

$$
E \left[ A _ {T} ^ {2} \right] = \frac {S ^ {2} e ^ {\sigma^ {2} t _ {1}}}{n ^ {2}} \left[ \frac {1 - e ^ {\sigma^ {2} h n}}{1 - e ^ {\sigma^ {2} h}} + \frac {2}{1 - e ^ {\sigma^ {2} h}} \left(n - \frac {1 - e ^ {\sigma^ {2} h n}}{1 - e ^ {\sigma^ {2} h}}\right) \right]
$$

If we are inside the average period,  $m > 0$ , then the strike price should be replaced by

$$
X = \frac {n X - m S _ {A}}{n - m} n - \frac {m}{n - m}
$$

Moreover, if  $S_A > \frac{n}{m} X$ , then the exercise is certain for a call, and in the case of a put, it must end up out-of-the-money. So the value of the put must be zero, while the value of the call must be

$$
c _ {A} = e ^ {- r T} (\hat {S} _ {A} - X),
$$

where  $\hat{S}_A = S_A\frac{m}{n} +E[A]\frac{n - m}{n}$

If there is only one fixing left to maturity, then the value can be calculated using the BSM formula weighted with time left to maturity and an adjusted strike price. The value of an Asian call option is then

$$
c _ {A} = c _ {B S M} (S, \hat {X}, T, r, b, \sigma) \frac {1}{n},
$$

where  $c_{BSM}$  the generalized BSM call formula, and

$$
\hat {X} = n X - (n - 1) S _ {A},
$$

and  $S_A$  is the realized average so far. Similarly, the value of a Asian put with one fixing left is

$$
p _ {A} = p _ {B S M} (S, \hat {X}, T, r, b, \sigma) \frac {1}{n},
$$

where  $p_{BSM}$  the generalized BSM put formula.

Table 4-26 gives values for discrete arithmetic average call options. Different choices for time to next average point  $t_1$  and volatility  $\sigma$  are reported.

# Computer algorithm

Function DiscreteAsianHHM(CallPutFlag As String, S As Double, SA As Double, X As Double, t1 As Double, T As Double, n As Double, m As Double, r As Double, b As Double, v As Double) As Double

'/'This is a modified version of the Levy formula published in

TABLE 4-26  

<table><tr><td colspan="7">Discrete Arithmetic Average Call Option Values(X = 100,T = 0.5 + t1, Δt = 1/52, r = 0.08, b = 0.03, n = 27, m = 0)</td></tr><tr><td>t1</td><td>S</td><td>σ = 0.1</td><td>σ = 0.2</td><td>σ = 0.3</td><td>σ = 0.4</td><td>σ = 0.5</td></tr><tr><td rowspan="3">0 weeks</td><td>95</td><td>0.2719</td><td>1.4166</td><td>2.8005</td><td>4.2572</td><td>5.7480</td></tr><tr><td>100</td><td>1.9484</td><td>3.4961</td><td>5.0557</td><td>6.6219</td><td>8.1951</td></tr><tr><td>105</td><td>5.7150</td><td>6.7212</td><td>8.0874</td><td>9.5713</td><td>11.1094</td></tr><tr><td rowspan="3">10/52(10 weeks)</td><td>95</td><td>0.8805</td><td>2.8800</td><td>5.0164</td><td>7.1874</td><td>9.3708</td></tr><tr><td>100</td><td>2.9570</td><td>5.1974</td><td>7.4551</td><td>9.7156</td><td>11.9757</td></tr><tr><td>105</td><td>6.5087</td><td>8.2935</td><td>10.4171</td><td>12.6364</td><td>14.8936</td></tr><tr><td rowspan="3">20/52(20 weeks)</td><td>95</td><td>1.4839</td><td>4.0658</td><td>6.7249</td><td>9.3973</td><td>12.0679</td></tr><tr><td>100</td><td>3.7669</td><td>6.4983</td><td>9.2546</td><td>12.0106</td><td>14.7590</td></tr><tr><td>105</td><td>7.2363</td><td>9.5520</td><td>12.2008</td><td>14.9356</td><td>17.6981</td></tr></table>

"Asian Pyramid Power" By Haug, Haug and Margrabe

Dim d1 As Double, d2 As Double, h As Double, EA As Double, EA2 As Double  
Dim vA As Double, OptionValue As Double

```txt
$\begin{array}{l}\mathrm{h} = (\mathrm{T} - \mathrm{t1}) / (\mathrm{n} - 1)\\ \mathrm{If~b} = 0\mathrm{Then}\\ \mathrm{EA} = \mathrm{S}\\ \mathrm{EA} = \mathrm{S} / \mathrm{n}*\mathbf{Exp}(\mathrm{b}*\mathrm{t1})*(1 - \mathbf{Exp}(\mathrm{b}*\mathrm{h}*\mathrm{n})) / (1 - \mathbf{Exp}(\mathrm{b}*\mathrm{h}))\\ \mathrm{End~If}\\ \mathrm{If~m} > 0\mathrm{Then}\\ \mathrm{'//Exercise~is~certain~for~call,~put~must~be~out-of-the-money}\\ \mathrm{If~SA > n / m*X~Then}\\ \mathrm{If~CallPutFlag = "p"~Then}\\ \mathrm{DiscreteAsianHHM = 0}\\ \mathrm{ElseIf~CallPutFlag = "c"~Then}\\ \mathrm{SA = SA*m / n + EA*(n - m) / n}\\ \mathrm{DiscreteAsianHHM = (SA - X)*Exp(-r*T)}\\ \mathrm{End~If}\\ \mathrm{Exit~Function}\\ \end{array}$    
If  $m > 0$  Then   
'//Exercise is certain for call，put must be out-of-the-money   
If SA>n/m*X Then   
If CallPutFlag  $=$  "p" Then   
DiscreteAsianHHM  $= 0$    
ElseIf CallPutFlag  $=$  "c" Then   
SA  $=$  SA*m/n+EA*(n-m)/n   
DiscreteAsianHHM  $=$  (SA-X)*Exp(-r*T)   
End If   
End If   
End If   
//Only one fix left use Black-Scholes weighted with time   
f m=n-1 Then   
 $\mathbf{X} = \mathbf{n}*\mathbf{X} - (\mathbf{n} - 1)*\mathbf{SA}$    
DiscreteAsianHHM  $=$  GBlackScholes(CallPutFlag,S,X,T,r,b,v) _   
 $*1/\mathfrak{n}$    
Exit Function   
End If
```

If  $b = 0$  Then

$$
\begin{array}{l} E A 2 = S * S * E x p (v * v * t 1) / (n * n) \\ \left. \star \left(\left(1 - \operatorname {E x p} (\mathrm {v} * \mathrm {v} * \mathrm {h} * \mathrm {n})\right) / \left(1 - \operatorname {E x p} (\mathrm {v} * \mathrm {v} * \mathrm {h})\right) \right. \right. _ {-} \\ + 2 / (1 - \operatorname {E x p} (\mathbf {v} * \mathbf {v} * \mathbf {h})) \\ * (n - (1 - \operatorname {E x p} (v * v * h * n)) / (1 - \operatorname {E x p} (v * v * h))) \\ \end{array}
$$

Else

$$
\begin{array}{l} E A 2 = S * S * E x p ((2 * b + v * v) * t 1) / (n * n) \\ \ast \left(\left(1 - \mathbf {E x p} \left(\left(2 * b + v * v\right) * h * n\right)\right) _ {-} \right. \\ / (1 - \mathbf {E x p} ((2 * b + v * v) * h)) \\ + 2 / (1 - \mathbf {E x p} ((\mathrm {b} + \mathrm {v} * \mathrm {v}) * \mathrm {h})) * ((1 - \mathbf {E x p} (\mathrm {b} * \mathrm {h} * \mathrm {n})) \\ / (1 - \operatorname {E x p} (\mathrm {b} * \mathrm {h})) - (1 - \operatorname {E x p} ((2 * \mathrm {b} + \mathrm {v} * \mathrm {v}) * \mathrm {h} * \mathrm {n})) \\ / \left(\mathbf {1} - \mathbf {E x p} \left(\left(2 * \mathbf {b} + \mathbf {v} * \mathbf {v}\right) * \mathbf {\bar {h}}\right)\right)\left. \right) \\ \end{array}
$$

End If

$$
\mathrm {v A} = \mathbf {S q r} ((\mathbf {L o g} (\mathrm {E A 2}) - 2 * \mathbf {L o g} (\mathrm {E A})) / \mathrm {T})
$$

OptionValue  $= 0$

If  $m > 0$  Then

$$
\mathrm {X} = \mathrm {n} / (\mathrm {n} - \mathrm {m}) * \mathrm {X} - \mathrm {m} / (\mathrm {n} - \mathrm {m}) * \mathrm {S A}
$$

End If

$$
d 1 = \left(\operatorname {L o g} (\mathrm {E A} / \mathrm {X}) + \mathrm {v A} ^ {\wedge} 2 / 2 * \mathrm {T}\right) / \left(\mathrm {v A} * \operatorname {S q r} (\mathrm {T})\right)
$$

$$
\mathrm {d} 2 = \mathrm {d} 1 - \mathrm {v A} * \mathbf {S q r} (\mathrm {T})
$$

If CallPutFlag = "c" Then

$$
\text {O p t i o n V a l u e} = \mathbf {E x p} (- r * T) * (\mathbf {E A} * \mathbf {C N D} (d 1) - \mathbf {X} * \mathbf {C N D} (d 2))
$$

$$
\begin{array}{l} E l s e I f (C a l l P u t F l a g = " p") T h e n \\ \text {O p t i o n V a l u e} = \mathbf {E x p} (- r * T) * (X * C N D (- d 2) - E A * C N D (- d 1)) \\ \end{array}
$$

End If

DiscreteAsianHHM = OptionValue * (n - m) / n

# End Function

Example: DiscreteAsianHHM("c", 100, 110, 105, 0, 0.5, 360, 180, 0.07, 0.02, 0.25) will return an arithmetic average call value of 2.0971.

# Curran's Approximation

Curran (1992) has developed an approximation method for pricing Asian options based on the geometric conditioning approach.16 Curran (1992) claims that this method is more accurate than other

closed-form approximations presented earlier.

$$
\begin{array}{l} c \approx e ^ {- r T} \left[ \frac {1}{n} \sum_ {i = 1} ^ {n} e ^ {\mu_ {i} + \sigma_ {i} ^ {2} / 2} N \left(\frac {\mu - \ln (\hat {X})}{\sigma_ {x}} + \frac {\sigma_ {x i}}{\sigma_ {x}}\right) \right. \\ \left. - X N \left(\frac {\mu - \ln (\hat {X})}{\sigma_ {x}}\right) \right], \tag {4.104} \\ \end{array}
$$

where

$S =$  Initial asset price.

$X =$  Strike price of option.

$r =$  Risk-free interest rate.

$b =$  Cost-of-carry.

$T =$  Time to expiration in years.

$t_1 = \text{Time to first averaging point.}$

$\Delta t = \text{Time between averaging points.}$

$n =$  Number of averaging points.  
$\sigma =$  Volatility of asset.

$N(x) =$  The cumulative normal distribution function.

$$
\mu_ {i} = \ln (S) + (b - \sigma^ {2} / 2) t _ {i}
$$

$$
\sigma_ {i} = \sqrt {\sigma^ {2} [ t _ {1} + (i - 1) \Delta t ]}
$$

$$
\sigma_ {x i} = \sigma^ {2} \{t _ {1} + \Delta t [ (i - 1) - i (i - 1) / (2 n) ] \}
$$

$$
\mu = \ln (S) + (b - \sigma^ {2} / 2) [ t _ {1} + (n - 1) \Delta t / 2 ]
$$

$$
\sigma_ {x} = \sqrt {\sigma^ {2} [ t _ {1} + \Delta t (n - 1) (2 n - 1) / 6 n ]}
$$

and

$$
\hat {X} = 2 X - \frac {1}{n} \sum_ {i = 1} ^ {n} \exp \left\{\mu_ {i} + \frac {\sigma_ {x i} [ \ln (X) - \mu ]}{\sigma_ {x} ^ {2}} + \frac {\sigma_ {i} ^ {2} - \sigma_ {x i} ^ {2} / \sigma_ {x} ^ {2}}{2} \right\}
$$

If we are inside the average period,  $m > 0$ , then the strike price should be replaced by

$$
X = \frac {n X - m S _ {A}}{n - m} n - \frac {m}{n - m}
$$

Further, if  $S_A > \frac{n}{m} X$ , then exercise is certain for a call, and in the case of a put, it must end up out-of-the-money. So the value of the put must be zero, while the value of the call must be

$$
c _ {A} = e ^ {- r T} \left(\hat {S} _ {A} - X\right),
$$

where  $\hat{S}_A = S_A\frac{m}{n} +E[A]\frac{n - m}{n}$

TABLE 4-27  

<table><tr><td colspan="7">Asian Call Options Using the Geometric Conditioning Approach (X = 100, T = 26 weeks, Δt = 1 week, r = 0.08, b = 0.03, n = 27)</td></tr><tr><td>t1</td><td>S</td><td>σ = 0.1</td><td>σ = 0.2</td><td>σ = 0.3</td><td>σ = 0.4</td><td>σ = 0.5</td></tr><tr><td rowspan="3">0</td><td>95</td><td>0.2758</td><td>1.4262</td><td>2.8099</td><td>4.2581</td><td>5.7298</td></tr><tr><td>100</td><td>1.9466</td><td>3.4899</td><td>5.0395</td><td>6.5878</td><td>8.1320</td></tr><tr><td>105</td><td>5.7110</td><td>6.7024</td><td>8.0489</td><td>9.5053</td><td>11.0051</td></tr><tr><td rowspan="3">10/52 (10 weeks)</td><td>95</td><td>0.8819</td><td>2.8814</td><td>5.0139</td><td>7.1753</td><td>9.3417</td></tr><tr><td>100</td><td>2.9560</td><td>5.1934</td><td>7.4443</td><td>9.6923</td><td>11.9321</td></tr><tr><td>105</td><td>6.5066</td><td>8.2852</td><td>10.3991</td><td>12.6029</td><td>14.8369</td></tr><tr><td rowspan="3">20/52 (20 weeks)</td><td>95</td><td>1.4844</td><td>4.0655</td><td>6.7207</td><td>9.3847</td><td>12.0409</td></tr><tr><td>100</td><td>3.7661</td><td>6.4952</td><td>9.2461</td><td>11.9920</td><td>14.7243</td></tr><tr><td>105</td><td>7.2348</td><td>9.5466</td><td>12.1885</td><td>14.9116</td><td>17.6564</td></tr></table>

If there is only one fixing left to maturity, then the value can be calculated using the generalized BSM formula weighted with time left to maturity and an adjusted strike price. The value of an Asian call option is then

$$
c _ {A} = c _ {B S M} (S, \hat {X}, T, r, b, \sigma) \frac {1}{n},
$$

where  $c_{BSM}$  is the generalized BSM call formula

$$
\hat {X} = n X - (n - 1) S _ {A},
$$

and  $S_A$  is the realized average so far. Similarly, the value of an Asian put with one fixing left is

$$
p _ {A} = p _ {B S M} (S, \hat {X}, T, r, b, \sigma) \frac {1}{n},
$$

where  $p_{BSM}$  the generalized BSM put formula.

Table 4-27 reports Asian option values based on Curran's approximation method.

# Computer algorithm

The computer code below calculates the Asian option value using Curran's approximation.

Function AsianCurranApprox(CallPutFlag As String, S As Double, SA As Double, X As Double, t1 As Double, T As Double, n As Long m As Long, r As Double, b As Double, v As Double) As Double

Dim dt As Double, my As Double, myi As Double

```vba
Dim vxi As Double, vi As Double, vx As Double
Dim Km As Double, sum1 As Double, sum2 As Double
Dim ti As Double, EA As Double
Dim z As Integer, i As Long
z = 1
If CallPutFlag = "p" Then
z = -1
End If
dt = (T - t1) / (n - 1)
If b = 0 Then
EA = S
Else
EA = S / n * Exp(b * t1) * (1 - Exp(b * dt * n)) / (1 - Exp(b * dt))
End If
If m > 0 Then
If SA > n / m * X Then
//Exercise is certain for call, put must be out-of-the-money:
If CallPutFlag = "p" Then
AsianCurranApprox = 0
ElseIf CallPutFlag = "c" Then
SA = SA * m / n + EA * (n - m) / n
AsianCurranApprox = (SA - X) * Exp(-r * T)
End If
Exit Function
End If
End If
If m = n - 1 Then
//Only one fix left use Black-Scholes weighted with time
X = n * X - (n - 1) * SA
AsianCurranApprox = GBlackScholes(CallPutFlag, S, X, T, r, b, v) _
* 1 / n
Exit Function
End If
If m > 0 Then
X = n / (n - m) * X - m / (n - m) * SA
End If
vx = v * Sqr(t1 + dt * (n - 1) * (2 * n - 1) / (6 * n))
my = Log(S) + (b - v * v * 0.5) * (t1 + (n - 1) * dt / 2)
sum1 = 0
For i = 1 To n Step 1
ti = dt * i + t1 - dt
vi = v * Sqr(t1 + (i - 1) * dt)
vxi = v * v * (t1 + dt * ((i - 1) - i * (i - 1) / (2 * n)))
myi = Log(S) + (b - v * v * 0.5) * ti
sum1 = sum1 + Exp(myi + vxi / (vx * vx) *
(Log(X) - my) + (vi * vi - vxi * vxi / (vx * vx)) * 0.5)
Next
Km = 2 * X - 1 / n * sum1
sum2 = 0
```

For  $\textbf{i} = 1$  To n Step 1

$$
\begin{array}{l} t i = d t * i + t 1 - d t \\ \mathbf {v} \mathbf {i} = \mathbf {v} * \mathbf {S q r} (\mathbf {t} 1 + (\mathbf {i} - \mathbf {1}) * \mathbf {d t}) \\ v x i = v * v * (t 1 + d t * ((i - 1) - i * (i - 1) / (2 * n))) \\ \operatorname {m y i} = \operatorname {L o g} (\mathrm {S}) + (\mathrm {b} - \mathrm {v} * \mathrm {v} * 0. 5) * \mathrm {t i} \\ \operatorname {s u m 2} = \operatorname {s u m 2} + \operatorname {E x p} (\text {m y i} + \text {v i} * \text {v i} * 0. 5) \\ * \mathrm {C N D} (\mathrm {z} * ((\mathrm {m y} - \mathrm {L o g} (\mathrm {K m})) / \mathrm {v x} + \mathrm {v x i} / \mathrm {v x})) \\ \end{array}
$$

Next

$$
\begin{array}{l} \text {A s i a n C u r r a n A p p r o x} = \mathbf {E x p} (- \mathbf {r} * \mathrm {T}) * \mathbf {z} * (1 / \mathbf {n} * \operatorname {s u m 2} - \mathbf {X} _ {-} \\ * \mathrm {C N D} (\mathrm {z} * (\mathrm {m y} - \mathrm {L o g} (\mathrm {K m})) / \mathrm {v x})) * (\mathrm {n} - \mathrm {m}) / \mathrm {n} \\ \end{array}
$$

# End Function

Example: AsianCurranApprox("c", 100, 110, 105, 0, 0.5, 360, 180, 0.07, 0.02, 0.25) will return an arithmetic average call value of 2.0928.

# 4.20.4 Equivalence of Floating-Strike and Fixed-Strike Asian Options

We have mainly been looking at how to value what is known as fixed-strike Asian options. In a floating-strike Asian option, the strike is set equal to the average, and a floating-strike call option will at maturity pay out the maximum of the spot price minus the realized average and zero,  $\max[S - A, 0]$ . Similarly, a floating-strike put will at maturity pay out  $\max[A - S, 0]$ . One way to find the value of a floating-strike Asian option, or vice versa, is by using what is known as fixed-floating Asian value symmetry, aka fixed-floating Asian Symmetry. Henderson and Wojakowski (2001) describe how to go from the value of a fixed-strike Asian option to a floating-strike Asian option, and vice versa.

Note on fixing convention: the discrete symmetry relation below is stated for an average over fixings that exclude the terminal spot $S_T$ (or equivalently a continuous average over $[0,T)$). If the arithmetic average includes $S_T$, then $S_T - A = \frac{n-1}{n}\left(S_T - A_{\text{excl}}\right)$, so the floating payoff gains a factor $(n-1)/n$ relative to the symmetry using $A_{\text{excl}}$.

$$
c _ {f} = (S, 1, T, r, b, \sigma) = p _ {X} (S, S, T, r - b, - b, \sigma), \tag {4.105}
$$

where  $c_f$  stands for a floating-strike Asian call and  $p_X$  stands for fixed-strike Asian put. Similarly, we have

$$
c _ {X} = (X, S, T, r, b, \sigma) = p _ {f} \left(S, \frac {X}{S}, T, r - b, - b, \sigma\right) \tag {4.106}
$$

This result holds for arithmetic Asian options when we are still not in the average period.

# 4.20.5 Asian Options with Volatility Term-Structure

Plain vanilla options on the same security but with different time to maturity typically trade at different (implied) volatilities. In other words, we typically observe a nontrivial volatility term structure. The

Asian option formulas mentioned so far assume a flat term structure of volatility. We now describe a more realistic model that can be calibrated to the term structure of plain vanilla option volatilities as described by Haug, Haug, and Margrabe (2003); see also Levy (1997).

The volatility of an arithmetic discrete average, calibrated to the term structure of implied volatilities, can be found as

$$
\sigma_ {A} = \sqrt {\frac {\ln \left(E \left[ A _ {T} ^ {2} \right]\right) - 2 \ln \left(E \left[ A _ {T} \right]\right)}{T}}, \tag {4.107}
$$

where

$$
E [ A _ {T} ] = \frac {1}{n} \sum_ {i = 1} ^ {n} F _ {i},
$$

where  $F_{i}$  is the forward price at fixing  $i$ . Moreover,

$$
E \left[ A _ {T} ^ {2} \right] = \frac {S ^ {2}}{n ^ {2}} \sum_ {i = 1} ^ {n} e ^ {(2 b + \sigma_ {i} ^ {2}) t _ {i}} + 2 \sum_ {i = 1} ^ {n} \sum_ {j = i + 1} ^ {n} e ^ {(b + \sigma_ {i} ^ {2}) t _ {i}} e ^ {b t _ {j}}
$$

$\sigma_{i}$  is the plain vanilla BSM volatility for an option with expiration  $t_i$ , where  $t_i$  is the time to fixing  $i$ . Defining  $F_A = E[A_T]$ , we can now approximate the value of the arithmetic call option as<sup>17</sup>

$$
c \approx e ^ {- r T} \left[ F _ {A} N \left(d _ {1}\right) - X N \left(d _ {2}\right) \right] \tag {4.108}
$$

and a put option as

$$
p \approx e ^ {- r T} \left[ X N \left(- d _ {2}\right) - F _ {A} N \left(- d _ {1}\right) \right], \tag {4.109}
$$

where  $N(\cdot)$  is the cumulative normal distribution function,

$$
d _ {1} = \frac {\ln (F _ {A} / X) + T \sigma_ {A} ^ {2} / 2}{\sigma_ {A} \sqrt {T}}
$$

and

$$
d _ {2} = d _ {1} - \sigma_ {A} \sqrt {T}
$$

Even if this basically is the Black-76 formula with a modified asset price and volatility, it still holds for Asian options on stocks, stock indexes, and futures.

It is well known that this type of model works best for reasonably low volatilities—for instance, spot volatility less than  $30\%$ . However, it is in general far better to use a relatively simple approximation that takes into account the term structure of volatility than using a more accurate model that does not calibrate to the term structure.

TABLE 4-28  

<table><tr><td colspan="7">Arithmetic Asian Options with Volatility Term Structure (S = 100, t1 = 1/52, T = 0.5, r = 0.05, b = 0, σ = 0.2, n = 26, m = 0)</td></tr><tr><td rowspan="2">X</td><td rowspan="2">Flat</td><td colspan="2">Call Values</td><td colspan="3">Put Values</td></tr><tr><td>Up +0.5%</td><td>Down -0.5%</td><td>Flat</td><td>Up +0.5%</td><td>Down -0.5%</td></tr><tr><td>80</td><td>19.5152</td><td>19.5063</td><td>19.5885</td><td>0.0090</td><td>0.0001</td><td>0.0823</td></tr><tr><td>90</td><td>10.1437</td><td>9.8313</td><td>10.7062</td><td>0.3906</td><td>0.0782</td><td>0.9531</td></tr><tr><td>100</td><td>3.2700</td><td>2.2819</td><td>4.3370</td><td>3.2700</td><td>2.2819</td><td>4.3370</td></tr><tr><td>110</td><td>0.5515</td><td>0.1314</td><td>1.2429</td><td>10.3046</td><td>9.8845</td><td>10.9960</td></tr><tr><td>120</td><td>0.0479</td><td>0.0016</td><td>0.2547</td><td>19.5541</td><td>19.5078</td><td>19.7609</td></tr></table>

Table 4-28 shows arithmetic Asian option values. The first column is values using  $20\%$  flat volatility term structure, while the next column is an upward-sloping term structure. We assume the plain vanilla implied volatility is increasing with  $0.5\%$  for every week to maturity. A six-month plain vanilla option thus trades for  $20\%$  implied Black-Scholes volatility, and a one-week option trades at  $7.5\%$  volatility. The third column is calibrated to a downward-sloping volatility term structure where the plain vanilla volatility for a one-week option trades at  $32.5\%$  volatility and a six-month option trades at  $20\%$  volatility.

# Computer algorithm

Function AsianDiscreteTermStructure(CallPutFlag As String, S As Double, SA As Double, X As Double, t1 As Double, T As Double, n As Long, m As Long, r As Double, b As Double, v As Object) As Double

Dim d1 As Double, d2 As Double, h As Double, EA As Double, EA2 As Double  
Dim vA As Double, OptionValue As Double

Dim i As Long, j As Long

Dim sum1 As Double, sum2 As Double

$$
\mathbf {h} = (\mathbf {T} - \mathbf {t} \mathbf {1}) / (\mathbf {n} - \mathbf {1})
$$

$$
\text {I f} \quad b = 0 \text {T h e n}
$$

$$
\mathbf {E A} = \mathbf {S}
$$

Else

$$
E A = S / n * \operatorname {E x p} (b * t 1) * (1 - \operatorname {E x p} (b * h * n)) / (1 - \operatorname {E x p} (b * h))
$$

End If

If  $m > 0$  Then

If SA > n / m * X Then

'// Exercise is certain for call, put must be out-of-the-money

If CallPutFlag = "p" Then

```latex
AsianDiscreteTermStructure  $= 0$  ElseIf CallPutFlag  $\equiv$  "c" Then SA  $=$  SA \*m/n+EA\*n-m)/n AsianDiscreteTermStructure  $= (\mathrm{SA - X})$  \*Exp(-r\*T) End If Exit Function End If   
End If   
If m=n-1 Then / Only one fix left use Black-Scholes weighted with time:  $\mathbf{X} = \mathbf{n}\ast \mathbf{X} - (\mathbf{n} - 1)\ast \mathbf{SA}$  AsianDiscreteTermStructure  $=$  GBlackScholes(CallPutFlag,S,X, T,r,b,v(n))  $\ast 1 / n$  Exit Function   
End If   
sum1  $= 0$    
sum2  $= 0$    
For i  $= 1$  To n-1 sum1  $=$  sum1  $^+$  Exp((2 \*b+v(i)^2)\* (t1+(i-1)*h)) For j  $= \mathrm{i} + 1$  To n sum2  $=$  sum2  $^+$  Exp((b+v(i)^2)\* (t1+(i-1)\*h)) _  $\star$  Exp(b\* (t1+(j-1)\*h)) Next   
Next   
sum1  $=$  sum1  $^+$  Exp((2 \*b+v(n)^2)\* (t1+(n-1)\*h))   
EA2  $= S^{\wedge}2 / (n^{\wedge}2)\ast$  (sum1+2\*sum2)   
vA  $=$  Sqr((Log(EA2)-2\* Log(EA))/T)   
If  $(\mathfrak{m} > 0)$  Then  $\mathbf{X} = \mathbf{n} / (\mathbf{n} - \mathbf{m})\ast \mathbf{X} - \mathbf{m} / (\mathbf{n} - \mathbf{m})\ast \mathbf{SA}$    
End If   
d1  $= (\operatorname {Log}(\operatorname {EA} / \operatorname {X}) + \operatorname {vA}^{\wedge}2 / 2\ast \operatorname {T}) / (\operatorname {vA}\ast \operatorname {Sqr}(\operatorname {EA})) / T)$    
d2  $= d1 - vA\ast Sqr(T)$    
If CallPutFlag  $= ^{\prime \prime}\mathrm{c}^{\prime \prime}$  Then OptionValue  $= \mathbf{Exp}(-\mathbf{r}\ast \mathbf{T})$  \* (EA\*CND(d1)-X\*CND(d2)) ElseIf (CallPutFlag  $= ^{\prime \prime}\mathrm{p}^{\prime \prime}$  ) Then OptionValue  $= \mathbf{Exp}(-\mathbf{r}\ast \mathbf{T})$  \* (X\*CND(-d2)-EA\*CND(-d1))   
End If   
AsianDiscreteTermStructure  $=$  OptionValue  $\ast$  (n-m)/n
```

End Function
