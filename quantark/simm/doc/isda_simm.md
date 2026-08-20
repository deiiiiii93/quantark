# ISDA SIMM<sup>®,1</sup> Methodology, version 2.6 (based on v2.5.6: 16 Aug 2023)  
Effective Date: December 2, 2023

# This Document

This document gives the official methodology for the calculation of the ISDA Standard Initial Margin Model (SIMM). This uses a risk-based approach incorporating Delta risk, Vega risk, Curvature risk, Inter-curve basis risk, Credit Base Correlation risk and Concentration risk.

# Contents

A. General provisions 2  
B. Structure of the methodology. 2  
C. Definition of the risk factors and the sensitivities 9  
D. Interest Rate risk 14  
E. Credit Qualifying risk 16  
F. Credit Non-Qualifying risk 18  
G. Equity risk 19  
H. Commodity risk 22

I. Foreign Exchange risk 24  
J. Concentration Thresholds 26  
K. Correlation between risk classes within product classes 29  
L. Additional Initial Margin Formulas 30

# A. General provisions

1. This document describes the calculations and methodology for calculating the initial margin under the ISDA Standard Initial Margin Model (SIMM) for non-cleared OTC derivatives.  
2. SIMM uses sensitivities as inputs. Risk factors and sensitivities must meet the definitions provided within Section C. This model includes complex trades, which should be handled in the same way as other trades.  
3. Sensitivities are used as inputs into aggregation formulae which are intended to recognize hedging and diversification benefits of positions in different risk factors within an asset class. Risk weights and correlations are provided in Sections D-I.  
4. The SIMM also includes procedures for bilateral remediation of portfolios for which the standard calculation is insufficient. These are described in the SIMM Governance Framework and section L of this document.

# B. Structure of the methodology

5. There are six risk classes:

- Interest Rate  
Credit (Qualifying)  
Credit (Non-Qualifying)  
Equity  
Commodity  
FX

and the margin for each risk class is defined to be the sum of the Delta Margin, the Vega Margin, the Curvature Margin and the Base Corr Margin (if applicable) for that risk class. That is

$$
I M _ {X} = D e l t a M a r g i n _ {X} + V e g a M a r g i n _ {X} + C u r v a t u r e M a r g i n _ {X} + B a s e C o r r M a r g i n _ {X},
$$

for each risk class  $X$ , where the BaseCorrMargin term is only present in the Credit (Qualifying) risk class.

6. There are four product classes:

- Interest Rates and Foreign Exchange (RatesFX)  
Credit  
Equity  
Commodity

Every trade is assigned to an individual product class and SIMM is considered separately for each product class. Buckets are still defined in risk terms, but within each product class the risk class takes its component risks only from trades of that product class. For example, equity derivatives would have risk in the Interest Rate risk class, as well as the Equity risk class. But all those risks are kept separate from the risks of trades in the RatesFX product class.

Within each product class, the initial margin (IM) for each of the six risk classes is calculated as in paragraph 5 above. The total margin for that product class is given by the formula:

$$
S I M M _ {p r o d u c t} = \sqrt {\sum_ {r} I M _ {r} ^ {2} + \sum_ {r} \sum_ {s \neq r} \psi_ {r s} I M _ {r} I M _ {s}} ,
$$

where product is one of the four product classes, and the sums on  $r$  and  $s$  are taken over the six risk classes. The correlation matrix  $\psi_{rs}$  of correlations between the risk classes is given in Section K.

The total SIMM is the sum of these four product class SIMM values:

$$
S I M M = S I M M _ {R a t e s F X} + S I M M _ {C r e d i t} + S I M M _ {E q u i t y} + S I M M _ {C o m m o d i t y}
$$

The SIMM equation can be extended to incorporate notional-based add-ons for specified products and/or multipliers to the individual product class SIMM values. Annex A contains the modified version of the SIMM in that case.

7. (Interest Rate risk only) The following step by step approach to capture delta risk should be applied to the interest-rate risk class only:

(a) Find a net sensitivity across instruments to each risk factor  $(k,i)$ , where  $k$  is the rate tenor and  $i$  is the index name of the sub yield curve, as defined in Sections C.1 and C.2 for the interest-rate risk class.  
(b) Weight the net sensitivity,  $s_{k,i}$ , to each risk factor  $(k,i)$  by the corresponding risk weight  $RW_{k}$  according to the vertex structure set out in Section D.

$$
W S _ {k, i} = R W _ {k} \mathrm {s} _ {k, i} C R _ {b},
$$

where  $CR$  is the concentration risk factor defined as:

$$
C R _ {b} = \max  \left(1, \left(\frac {\left| \sum_ {k , i} s _ {k , i} \right|}{T _ {b}}\right) ^ {\frac {1}{2}}\right),
$$

for concentration threshold  $T_{b}$ , defined for each currency  $b$  in section J. Note that inflation sensitivities to currency  $b$  are included in  $\left|\sum_{k,i} s_{k,i}\right|$ , but cross-currency basis swap sensitivities are not. Neither should cross-currency basis swap sensitivities be scaled by the concentration risk factor.

(c) The weighted sensitivities should then be aggregated within each currency. The sub-curve correlations  $\phi_{i,j}$  and the tenor correlation parameters  $\rho_{k,l}$  are set out in Section D.

$$
K = \sqrt {\sum_ {i , k} W S _ {k , i} ^ {2} + \sum_ {i , k} \sum_ {(j , l) \neq (i , k)} \phi_ {i , j} \rho_ {k , l} W S _ {k , i} W S _ {l , j}}.
$$

(d) Delta Margin amounts should then be aggregated across currencies within the risk class. The correlation parameters  $\gamma_{bc}$  applicable are set out in Section D.

$$
D e l t a M a r g i n = \sqrt {\sum_ {b} K _ {b} ^ {2} + \sum_ {b} \sum_ {c \neq b} \gamma_ {b c} g _ {b c} S _ {b} S _ {c}},
$$

where

$$
S _ {b} = \max  \left(\min  \left(\sum_ {i, k} W S _ {k, i}, K _ {b}\right), - K _ {b}\right) \quad \text {a n d} \quad g _ {b c} = \frac {\min  \left(C R _ {b} , C R _ {c}\right)}{\max  \left(C R _ {b} , C R _ {c}\right)},
$$

for all currencies  $b$  and  $c$ .

8. (non-Interest Rate risk classes) The following step by step approach to capture delta risk should be separately applied to each risk class other than Interest Rate:

(a) Find a net sensitivity across instruments to each risk factor  $k$ , which are defined in Sections C.1 and C.2

for each risk class.

(b) Weight the net sensitivity,  $s_k$ , to each risk factor  $k$  by the corresponding risk weight  $RW_k$  according to the bucketing structure for each risk class set out in Sections E-I.

$$
W S _ {k} = R W _ {k} s _ {k} C R _ {k},
$$

where  $CR_{k}$  is the concentration risk factor:

$$
C R _ {k} = \max  \left(1, \left(\frac {\left| \sum_ {j} s _ {j} \right|}{T _ {b}}\right) ^ {\frac {1}{2}}\right) \text {f o r c r e d i t s p r e a d r i s k},
$$

with the sum  $j$  taken over all the risk factors that have the same issuer and seniority as the risk factor  $k$ , irrespective of the tenor or payment currency, and

$$
C R _ {k} = \max  \left(1, \left(\frac {\left| s _ {k} \right|}{T _ {b}}\right) ^ {\frac {1}{2}}\right) \text {f o r e q u i t y , c o m m o d i t y , F X r i s k},
$$

where  $T_{b}$  is the concentration threshold for the bucket (or FX category)  $b$ , as given in Section J. Note that base correlation sensitivities are not included in the concentration risk, and the concentration risk factor for those risk factors should be taken as 1.

(c) Weighted sensitivities should then be aggregated within each bucket. The buckets and correlation parameters applicable to each risk class are set out in Sections E-I.

$$
K = \sqrt {\sum_ {k} W S _ {k} ^ {2} + \sum_ {k} \sum_ {l \neq k} \rho_ {k l} f _ {k l} W S _ {k} W S _ {l}},
$$

where

$$
f _ {k l} = \frac {\operatorname* {m i n} \left(C R _ {k} , C R _ {l}\right)}{\operatorname* {m a x} \left(C R _ {k} , C R _ {l}\right)}.
$$

(d) Delta Margin amounts should then be aggregated across buckets within each risk class. The correlation parameters  $\gamma_{bc}$  applicable to each risk class are set out in Sections E-I.

$$
D e l t a M a r g i n = \sqrt {\sum_ {b} K _ {b} ^ {2} + \sum_ {b} \sum_ {c \neq b} \gamma_ {b c} S _ {b} S _ {c}} + K _ {r e s i d u a l},
$$

where

$$
S _ {b} = \max  \left(\min  \left(\sum_ {k = 1} ^ {K} W S _ {k}, K _ {b}\right), - K _ {b}\right)
$$

for all risk factors in bucket  $b$ .

9. Instruments that are options or include an option, including a prepayment option or have volatility sensitivity (instruments subject to optionality) are subject to additional margin requirements for vega risk and curvature risk, as described in paragraphs 10 and 11. Instruments not subject to optionality and with no volatility sensitivity are not subject to vega risk or curvature risk.

10. The following step by step approach to capture vega risk exposure should be separately applied to each risk class:

(a) For Interest Rate and Credit instruments, the volatility  $\sigma_{kj}$  for risk factor  $k$  and maturity  $j$ , is defined to be the implied at-the-money volatility of the swaption with expiry time equal to the tenor  $k$ , and at some swap maturity  $j$ . The volatility can be quoted as normal volatility, log-normal volatility or similar.

In the case where  $k$  is the inflation risk-factor, the inflation volatility  $\sigma_{kj}$  of an inflation swaption of type  $j$  is defined to be the at-the-money volatility of the swaption, where the type  $j$  comprises an initial inflation observation date and a final inflation observation date. The option expiry date shall be defined to be the final inflation observation date, and risk should be expressed on a set of option expires equal to the same tenor buckets as interest-rate delta. The volatility can be quoted as normal volatility, log-normal volatility or similar.

(b) For Equity, FX and Commodity instruments, the volatility  $\sigma_{kj}$  of the risk factor  $k$  at each vol-tenor  $j$  is given by the following formula:

$$
\sigma_ {k j} = \frac {R W _ {k} \sqrt {365 / 14}}{\alpha}, \quad \text {where} \alpha = \Phi^ {- 1} (99 \%),
$$

where  $\alpha$  is the 99th percentile of the cumulative standard normal distribution and  $RW_{k}$  is the corresponding delta risk weight of the risk factor  $k$ , and the "vol-tenor"  $j$  is the option expiry time, which should use the same tenor buckets as interest-rate delta risk: 2 weeks, 1 month, 3 months, 6 months, 1 year, 2 years, 3 years, 5 years, 10 years, 15 years, 20 years and 30 years. For commodity index volatilities, the risk weight to use is that of the "Indexes" bucket. For FX vega (which depends on a pair of currencies), the risk weight to use here is the entry from the FX delta risk weight table, given in section I, whose row is the FX volatility group of the first currency and whose column is the FX volatility group of the second currency.

(c) The vega risk for each instrument  $i$  to risk factor  $k$  is estimated using the formula:

$$
\begin{array}{l} V R _ {i k} = \sum_ {j} \sigma_ {k j} \frac {\partial V _ {i}}{\partial \sigma}, \quad \text {f o r i n t e r e s t r a t e s a n d c r e d i t , o r} \\ V R _ {i k} = H V R _ {c} \sum_ {j} \sigma_ {k j} \frac {\partial V _ {i}}{\partial \sigma}, \quad \text {f o r e q u i t y , c o m m o d i t y a n d F X}, \\ \end{array}
$$

where:

-  $\sigma_{kj}$  is the volatility defined in clauses (a) and (b);  
-  $\partial V_{i} / \partial \sigma$  is the sensitivity of the price of the instrument  $i$  with respect to the implied at-the-money volatility (i.e. "vega"), as defined in section C.3, but must match the definition used in clause (a).  
-  $HVR_{c}$  is the historical volatility ratio for the risk class concerned,  $c$ , set out in sections G-I, which corrects for inaccuracy in the volatility estimate  $\sigma_{kj}$ .

For example, the 5-year Interest Rate vega is the sum of all vol-weighted interest rate caplet and swaption vegas which expire in 5 years' time; the USD/JPY FX vega is the sum of all vol-weighted USD/JPY FX vegas. For inflation, the inflation vega is the sum of all vol-weighted inflation swaption vegas in the particular currency.

(d) Find a net vega risk exposure  $VR_{k}$  across instruments  $i$  to each risk factor  $k$ , which are defined in Sections C.1 and C.2, as well as the vega concentration risk factor. For interest-rate vega risk, these are given by the formulas

$$
V R _ {k} = V R W \left(\sum_ {i} V R _ {i k}\right) V C R _ {b}, \quad \text {w h e r e} \quad V C R _ {b} = \max  \left(1, \left(\frac {\left| \sum_ {i k} V R _ {i k} \right|}{V T _ {b}}\right) ^ {\frac {1}{2}}\right),
$$

where  $b$  is the bucket which contains the risk factor  $k$ . For credit spread vega risk, the corresponding formulas are

$$
V R _ {k} = V R W \left(\sum_ {i} V R _ {i k}\right) V C R _ {k}, \quad \text {w h e r e} \quad V C R _ {k} = \max  \left(1, \left(\frac {\left| \sum_ {i j} V R _ {i j} \right|}{V T _ {b}}\right) ^ {\frac {1}{2}}\right),
$$

where the sum  $j$  is taken over tenors of the same issuer/seniority curve as the risk factor  $k$ , irrespective of the tenor or payment currency. For Equity, FX and Commodity vega risk, the corresponding formulas are

$$
V R _ {k} = V R W \left(\sum_ {i} V R _ {i k}\right) V C R _ {k}, \quad \text {w h e r e} V C R _ {k} = \max  \left(1, \left(\frac {\left| \sum_ {i} V R _ {i k} \right|}{V T _ {b}}\right) ^ {\frac {1}{2}}\right).
$$

Here  $VRW$  is the vega risk weight for the risk class concerned, set out in Sections D-I, and  $VT_{b}$  is the vega concentration threshold for bucket (or FX category)  $b$ , as given in section J. Note that there is special treatment for index volatilities in Credit Qualifying, Equity and Commodity risk classes.

(e) The vega risk exposure should then be aggregated within each bucket. The buckets and correlation parameters applicable to each risk class are set out in Sections D-I.

$$
K _ {b} = \sqrt {\sum_ {k} V R _ {k} ^ {2} + \sum_ {k} \sum_ {l \neq k} \rho_ {k l} f _ {k l} V R _ {k} V R _ {l}},
$$

where the inner correlation adjustment factors  $f_{kl}$  are defined to be identically 1 in the interest-rate risk class and for all other risk classes are defined to be:

$$
f _ {k l} = \frac {\operatorname* {m i n} \left(V C R _ {k} , V C R _ {l}\right)}{\operatorname* {m a x} \left(V C R _ {k} , V C R _ {l}\right)}.
$$

(f) Vega Margin should then be aggregated across buckets within each risk class. The correlation parameters applicable to each risk class are set out in Sections D-I.

$$
V e g a M a r g i n = \sqrt {\sum_ {b} K _ {b} ^ {2} + \sum_ {b} \sum_ {c \neq b} \gamma_ {b c} g _ {b c} S _ {b} S _ {c}} + K _ {r e s i d u a l},
$$

where

$$
S _ {b} = \max  \left(\min  \left(\sum_ {k = 1} ^ {K} V R _ {k}, K _ {b}\right), - K _ {b}\right),
$$

for all risk factors in bucket  $b$ . The outer correlation adjustment factors  $g_{bc}$  are identically 1 for all risk classes other than interest-rates, and for interest rates they are defined to be:

$$
g _ {b c} = \frac {\operatorname* {m i n} \left(V C R _ {b} , V C R _ {c}\right)}{\operatorname* {m a x} \left(V C R _ {b} , V C R _ {c}\right)}
$$

for all pairs of buckets  $b, c$ .

11. The following step by step approach to capture curvature risk exposure should be separately applied to each risk class:

(a) The curvature risk exposure for each instrument  $i$  to risk factor  $k$  is estimated using the formula:

$$
C V R _ {i k} = \sum_ {j} S F (t _ {k j}) \sigma_ {k j} \frac {\partial V _ {i}}{\partial \sigma},
$$

where:

-  $\sigma_{kj}$  and  $\partial V_i / \partial \sigma$  are the volatility and vega defined in paragraph 10(a-c) above.  
$t_{kj}$  is the expiry time (in calendar days) from the valuation date until the expiry date of the

standard option corresponding to this volatility and vega.

-  $SF(t)$  is the value of the scaling function obtained from the linkage between vega and gamma for vanilla options.

$$
S F (t) = 0. 5 \min  \left(1, \frac {1 4 \text {d a y s}}{t \text {d a y s}}\right).
$$

The scaling function is a function of expiry only, which is independent of both vega and vol, as shown in the example table below.

<table><tr><td>Expiry</td><td>2w</td><td>1m</td><td>3m</td><td>6m</td><td>12m</td><td>2y</td><td>3y</td><td>5y</td><td>10y</td></tr><tr><td>SF</td><td>50.0%</td><td>23.0%</td><td>7.7%</td><td>3.8%</td><td>1.9%</td><td>1.0%</td><td>0.6%</td><td>0.4%</td><td>0.2%</td></tr></table>

Here, we convert tenors to calendar days using the convention that “12m” equals 365 calendar days, with pro-rata scaling for other tenors so that  $1\mathrm{m} = 365 / 12$  days and  $5\mathrm{y} = 365 * 5$  days.

- For curvature margin calculations, netting across expiry times of volatility sensitivities to the same risk factor should be carried out by the formula above, using the scaling function weights, and not earlier in the calculation.

(b) The curvature risk exposure  $CVR_{ik}$  then can be netted across instrument  $i$  to each risk factor  $k$ , which are defined in Sections C.1 and C.2. Note that the same special treatment as for vega applies for indexes in Credit, Equity and Commodity risk classes. The curvature risk exposure for bucket 12 (Volatility Indexes) in the equity risk class shall be taken to be zero.  
(c) The curvature risk exposure should then be aggregated within each bucket using the following formula:

$$
K _ {b} = \sqrt {\sum_ {k} C V R _ {b , k} ^ {2} + \sum_ {k} \sum_ {l \neq k} \rho_ {k l} ^ {2} C V R _ {b , k} C V R _ {b , l}},
$$

where

-  $\rho_{kl}$  is the assumed correlation applicable to each risk class as set out in Sections D-I. Note the use of  $\rho_{kl}^2$  rather than  $\rho_{kl}$ .

(d) Margin should then be aggregated across buckets within each risk class:

$$
\theta = \min  \left(\frac {\sum_ {b , k} C V R _ {b , k}}{\sum_ {b , k} \left| C V R _ {b , k} \right|}, 0\right), \quad \text {a n d} \quad \lambda = (\Phi^ {- 1} (99.5 \%) ^ {2} - 1) (1 + \theta) - \theta ,
$$

where the sums are taken over all the non-residual buckets in the risk class, and  $\Phi^{-1}(99.5\%)$  is the  $99.5^{\text{th}}$  percentile of the standard normal distribution. Then the non-residual curvature margin is

$$
C u r v a t u r e M a r g i n _ {n o n - r e s} = \max \left(\sum_ {b, k} C V R _ {b, k} + \lambda \sqrt {\sum_ {b} K _ {b} ^ {2} + \sum_ {b} \sum_ {c \neq b} \gamma_ {b c} ^ {2} S _ {b} S _ {c}} , 0\right),
$$

where

$$
S _ {b} = \max  \left(\min  \left(\sum_ {k} C V R _ {b, k}, K _ {b}\right), - K _ {b}\right).
$$

Similarly, the residual equivalents are defined as

$$
\theta_ {r e s i d u a l} = \min \left(\frac {\sum_ {k} C V R _ {r e s i d u a l , k}}{\sum_ {k} \left| C V R _ {r e s i d u a l , k} \right|}, 0\right), \quad \mathrm {a n d}
$$

$$
\lambda_ {r e s i d u a l} = \big (\Phi^ {- 1} (99.5 \%) ^ {2} - 1 \big) (1 + \theta_ {r e s i d u a l}) - \theta_ {r e s i d u a l},
$$

$$
C u r v a t u r e M a r g i n _ {r e s i d u a l} = \max  \left(\sum_ {k} C V R _ {r e s i d u a l, k} + \lambda_ {r e s i d u a l} K _ {r e s i d u a l}, 0\right)
$$

Here

- the correlation parameters  $\gamma_{bc}$  applicable to each risk class are set out in Sections D-I. Note the use of  $\gamma_{bc}^{2}$  rather than  $\gamma_{bc}$ .

Then the total curvature margin is defined to be the sum of the two terms:

$$
\text {C u r v a t u r e M a r g i n} = \text {C u r v a t u r e M a r g i n} _ {\text {n o n - r e s}} + \text {C u r v a t u r e M a r g i n} _ {\text {r e s i d u a l}}.
$$

For the interest-rate risk class only, the CurvatureMargin must be multiplied by a scale factor of  $HVR_{IR}^{-2}$ , where  $HVR_{IR}$  is the historical volatility ratio for the interest-rate risk class.

12. Credit Qualifying Only: Instruments whose price is sensitive to correlation between the defaults of different credits within an index or basket, such as CDO tranches, are subject to Base Correlation margin charge described in paragraph 13. Instruments not sensitive to base correlation are not subject to base correlation margin requirements.  
13. The following step by step approach to capture Base Correlation risk exposure should be applied to the Credit (Qualifying) risk class:  
(a) Find a net sensitivity across instruments to each Base Correlation risk factor  $k$ , where  $k$  is the index family such as CDX IG.  
(b) Weight the net sensitivity,  $s_k$ , to each risk factor  $k$  by the corresponding risk weight  $RW_k$ , specified in section E:

$$
W S _ {k} = R W _ {k} s _ {k}.
$$

(c) Weighted sensitivities should then be aggregated to give the Base Correlation Margin, as follows:

$$
B a s e C o r r M a r g i n = \sqrt {\sum_ {k} W S _ {k} ^ {2} + \sum_ {k} \sum_ {l \neq k} \rho_ {k l} W S _ {k} W S _ {l}}.
$$

The correlation parameters are set out in Section E.

# C. Definition of the risk factors and the sensitivities

# C.1 Definition of the risk factors

14. The Interest Rate risk factors are the 12 yields at the following vertices, for each currency: two weeks, 1 month, 3 months, 6 months, 1 year, 2 years, 3 years, 5 years, 10 years, 15 years, 20 years and 30 years.

The relevant yield curve is the yield curve of the currency in which an instrument is denominated.

For a given currency, there are a number of sub yield curves used, named "OIS", "Libor1m", "Libor3m", "Libor6m", "Libor12m" and (for USD only) "Prime" and "Municipal". Each sub curve has an index name  $i$ . Risk should be separately bucketed by currency, tenor and curve index, expressed as risk to the outright rate of the sub curve. Any sub curve not given on the above list should be mapped to its closest equivalent.

The Interest Rate risk factors also include a flat inflation rate for each currency. When at least one contractual payment obligation depends on an inflation rate, the inflation rate for the relevant currency is used as a risk factor. All sensitivities to inflation rates for the same currency are fully offset.

For cross-currency swap products whose notional exchange is eligible for exclusion from the margin calculation, the interest rate risk factors also include a flat cross-currency basis swap spread for each currency. Cross-currency basis swap spreads should be quoted as a spread to the non-USD Libor versus a flat USD Libor leg. All sensitivities to cross-currency basis swap spreads for the same currency are fully offset.

If fallback provisions take effect for a Libor or IBOR rate to an RFR-based fallback, then the relevant Libor sub curve should no longer be used for risk to that index and the OIS sub curve should be used instead. If fallback provisions take effect for a Libor or IBOR rate which is used in a standard cross-currency basis swap leg, then that leg should be redefined to use the RFR which would be used to calculate the fallback rate in the 2006 ISDA Definitions.

15. The Credit Qualifying risk factors are five credit spreads for each issuer/seniority pair, separately by payment currency, at each of the following vertices: 1 year, 2 years, 3 years, 5 years and 10 years.

For a given issuer/seniority, if there is more than one relevant credit spread curve, then the credit spread risk at each vertex should be the net sum of risk at that vertex over all the credit spread curves of that issuer and seniority, which may differ by documentation (such as restructuring clause), but not by currency. Note that delta and vega sensitivities arising from different payment currencies (such as Quanto CDS) are considered different risk factors to the same issuer/seniority from each other.

For Credit Qualifying indexes and bespoke baskets (including securitizations and non-securitizations), delta sensitivities should be computed to the underlying issuer/seniority risk factors. Vega sensitivities of credit indexes need not be allocated to underlying risk factors, but rather the entire index vega risk should be classed into the appropriate Credit Qualifying bucket, using the Residual bucket for cross-sector indexes.

The Credit Qualifying risk factors can also include Base Correlation risks from CDO tranches on the CDX or iTraxx families of credit indices. There is one flat risk factor for each index family. Base Correlation risks to the same index family (such as CDX IG, iTraxx Main, and so on) should be fully offset, irrespective of series, maturity or detachment point.

16. The Credit Non-Qualifying risk factors are five credit spreads for each issuer/tranche at each of the following vertices: 1 year, 2 years, 3 years, 5 years and 10 years.

Sensitivities should be computed to the tranche. For a given tranche, if there is more than one relevant

credit spread curve, then the credit spread risk at each vertex should be the net sum of risk at that vertex over all the credit spread curves of that tranche. Vega sensitivities of credit indexes need not be allocated to underlying issuers, but rather the entire index vega should be classed into the appropriate Non-qualifying bucket, using the Residual bucket for cross-sector indexes.

17. The Equity risk factors are all the equity prices: each equity spot price is a risk factor. Sensitivities to equity indices, funds and ETFs can be handled in one of two ways: either (standard preferred approach) the entire delta and can be put into the "Indexes, Funds, ETFs" Equity bucket, or (alternative approach if bilaterally agreed) the delta can be allocated back to individual equities. The choice between standard and alternative approach should be made on a portfolio-level basis. Delta sensitivities to bespoke baskets should always be allocated back to individual equities. Vega sensitivities of equity indexes, funds and ETFs need not be allocated back to individual equities, but rather the entire vega risk should be classed into the "Indexes, Funds, ETFs" Equity bucket. Vega sensitivities to bespoke baskets should be allocated back to individual equities. Note that not all institutions may be able to perform the allocation of vega for equities as described, however, it is the preferred approach. For equity volatility indexes, the index risk should be treated as equity volatility risk and put into the "Volatility Index" bucket.  
18. The Commodity risk factors are all the commodity prices: each commodity spot price is a risk factor. Examples include "Coal Europe", "Precious Metals Gold" and "Livestock Lean Hogs". Risks to commodity forward prices should be allocated back to spot price risks and aggregated, assuming that each commodity forward curve moves in parallel. Sensitivities to commodity indices can be handled in one of two ways: either (standard approach) the entire delta can be put into the "Indexes" bucket, or (advanced approach) the delta can be allocated back to individual commodities. The choice between standard and advanced approaches should be made on a portfolio-level basis. Delta sensitivities to bespoke baskets should always be allocated back to individual commodities. Vega sensitivities of commodity indexes should not be allocated back to individual commodities, but rather the entire index vega risk should be classed into the "Indexes" bucket.  
19. The FX risk factors are all the exchange rates between the calculation currency and any currency, or currency of any FX cross rate, on which the value of an instrument may depend. This excludes the calculation currency itself. The FX vega and curvature risk factors are all the currency pairs to which an instrument has FX volatility risk.

# C.2 Definition of "sensitivity" for delta margin calculation

20. The following sections define the sensitivity  $s$  that should be used as input into the delta margin calculation. The forward difference is specified in each section for illustrative purposes:

For Interest Rate and Credit:

$$
s = V (x + 1 \mathrm {b p}) - V (x)
$$

For Equity, Commodity, and FX risk:

$$
s = V (x + 1 \%. x) - V (x)
$$

where:

$s$  is the sensitivity to the risk factor  $x$  
$V(x)$  is the value of the instrument, given the value of the risk factor  $x$

21. However, banks may also make use of the central or backward difference methods, or use a smaller shock size and scale-up:

For Interest Rate and Credit:

$$
\begin{array}{l} s = V (x + 0. 5 \mathrm {b p}) - V (x - 0. 5 \mathrm {b p}) \\ s = V (x) - V (x - 1 \mathsf {b p}) \\ s = (V (x + \varepsilon . 1 \mathrm {b p}) - V (x)) / \varepsilon , \text {w h e r e} 0 <   | \varepsilon | \leq 1. \\ \end{array}
$$

For Equity, Commodity and FX risk:

$$
\begin{array}{l} s = V (x + 0.5 \%. x) - V (x - 0.5 \%. x) \\ s = V (x) - V (x - 1 \%. x) \\ s = (V (x + 1 \%. \varepsilon . x) - V (x)) / \varepsilon , \text {w h e r e} 0 <   | \varepsilon | \leq 1. \\ \end{array}
$$

# 22. For Interest Rate risk factors, the sensitivity is defined as the PV01.

The PV01 of an instrument  $i$  with respect to tenor  $t$  of the risk-free curve  $r$  (ie the sensitivity of instrument  $i$  with respect to the risk factor  $r_t$ ) is defined as:

$$
s (i, r _ {t}) = V _ {i} \left(r _ {t} + 1 \mathrm {b p}, c s _ {t}\right) - V _ {i} \left(r _ {t}, c s _ {t}\right)
$$

with

-  $r_t$ : the risk-free interest rate at tenor  $t$  
$cS_{t}$  : the credit spread at tenor  $t$  
-  $V_{i}$ : the market value of an instrument  $i$  as a function of the risk-free interest rate and credit spread curve  
1bp: 1 basis point, i.e., 0.0001 or  $0.01\%$ .

For the interest rate risk factors, "market rates" (and not "zero coupon rates") should be used to construct the risk-free yield curve, except for inflation risk which should be taken with respect to zero-coupon inflation swap market rates.

# 23. For Credit non-securitisation risk factors, the sensitivity is defined as the CS01.

The CS01 of an instrument with respect to tenor t is defined as:

$$
s (i, c s _ {t}) = V _ {i} \left(r _ {t}, c s _ {t} + 1 \mathfrak {b p}\right) - V _ {i} \left(r _ {t}, c s _ {t}\right)
$$

# 24. For Credit Qualifying and Non-Qualifying securitisations, including nth-to-default risk factors, the sensitivity is defined as the CS01.

If all the following criteria are met, the position is deemed to be a qualifying securitisation, and the CS01 (as defined for Credit (non-securitisations) above) should be computed with respect to the names underlying the securitisation or nth-to-default instrument:

- The positions are not re-securitisation positions, nor derivatives of securitisation exposures that do not provide a pro-rata share in the proceeds of a securitisation tranche  
- All reference entities are single-name products, including single-name credit derivatives, for which a liquid two-way market exists (see below), including traded indices on these reference entities.  
- The instrument does not reference an underlying that would be treated as a retail exposure, a residential mortgage exposure, or a commercial mortgage exposure under the standardised approach to credit risk.  
The instrument does not reference a claim on a special purpose entity

If any of these criteria are not met, the position is deemed to be non-qualifying, and then the CS01 should be

calculated with respect to the spread of the instrument rather than the spread of the underlying of the instruments.

A two-way market is deemed to exist where there are independent bona fide offers to buy and sell so that a price reasonably related to the last sales price or current bona fide competitive bid and offer quotations can be determined within one day and settled at such price within a relatively short time conforming to trade custom.

# 25. For Credit Qualifying Base Correlation risk factors, the sensitivity is defined as the BC01.

The BC01 is the change in value for a 1 percentage point increase in the Base Correlation level, that is the sensitivity  $s_{ik}$  defined as

$$
s _ {i k} = V _ {i} \left(B C _ {k} + 1 \%\right) - V _ {i} \left(B C _ {k}\right)
$$

where

$k$  is a given credit index family such as CDX IG or iTraxx Main  
$BC_{k}$  is the Base Correlation curve/surface for index  $k$ , with numerical values such as 0.55.  
1% is one percentage point of correlation, that is 0.01.  
$V_{i}(BC_{k})$  is the value of instrument  $i$  as a function of the Base Correlation for index  $k$ .

# 26. For Equity risk factors, the sensitivity is defined as follows:

The value change of an instrument with respect to a 1 percentage point relative change of the equity price:

$$
s _ {i k} = V _ {i} \left(E Q _ {k} + 1 \% . E Q _ {k}\right) - V _ {i} \left(E Q _ {k}\right)
$$

with

$k$  :a given equity  
-  $EQ_{k}$ : the market value of equity  $k$  
-  $V_{i}$ : the market value of instrument  $i$  as a function of the price of equity  $k$

# 27. For Commodity risk factors, the sensitivity is defined as follows:

The value change of an instrument with respect to a 1 percentage point relative change of the commodity price:

$$
s _ {i k} = V _ {i} \left(\text {CTY} _ {k} + 1 \% \text {CTY} _ {k}\right) - V _ {i} \left(\text {CTY} _ {k}\right)
$$

with

$k$  : a given commodity  
-  $C T Y_{k}$ : the market value of commodity  $k$  
-  $V_{i}$ : the market value of instrument  $i$  as a function of the price of commodity  $k$

# 28. For FX risk factors, the sensitivity is defined as follows:

The value change of an instrument with respect to a 1 percentage point relative change of the

FX rate:

$$
s _ {i k} = V _ {i} \left(F X _ {k} + 1 \%. F X _ {k}\right) - V _ {i} \left(F X _ {k}\right),
$$

with

$k$  : a given currency, other than the calculation currency  
-  $FX_{k}$ : the spot exchange rate between currency  $k$  and the calculation currency, expressed in units of the calculation currency for one unit of currency  $k$ .  
-  $V_{i}$ : the market value of instrument  $i$ , in calculation currency terms, as a function of the exchange rate  $FX_{k}$ .

The FX sensitivity should include the FX translation risk of the instrument's value into the calculation currency. But, the FX sensitivity in the case where  $k$  equals the calculation currency is not included in the calculation.

29. When computing a first order sensitivity for instruments subject to optionality, it is recommended that the volatility under the bump is adjusted per prevailing market practice in each risk class.

# C.3 Definition of "sensitivity" for vega and curvature margin calculation

30. The following paragraphs define the sensitivity  $\partial V_{i} / \partial \sigma$  that should be used as input into the vega and curvature margin calculations in paragraphs 10 and 11. The vega sensitivity to the implied volatility risk factor  $\sigma$  is defined as:

$$
\frac {\partial V _ {i}}{\partial \sigma} = V (\sigma + 1) - V (\sigma),
$$

where:

-  $V(\sigma)$  is the value of the instrument given the implied volatility  $\sigma$  of the risk factor, whilst keeping other inputs, including skew and smile, constant,  
- the implied volatility  $\sigma$  should be the log-normal volatility, except in the case of Interest Rate and Credit risks when it can be the normal volatility or log-normal volatility, or similar, but must match the definition used in paragraph 10 clause (a)  
- for Equity, FX and Commodity instruments, the units of  $\sigma$  must be percentages of log-normal volatility, so that  $20\%$  is represented as 20. A shock to  $\sigma$  of 1 unit therefore represents an increase in volatility of  $1\%$ .  
- for Interest Rate and Credit instruments, the units of  $\sigma$  must match the units of the volatility  $\sigma_{kj}$  used in paragraph 10 clause (a).

31. The central or backward difference methods may also be used, or use a smaller shock size and scale-up:

$$
\frac {\partial V _ {i}}{\partial \sigma} = V (\sigma + 0. 5) - V (\sigma - 0. 5), o r
$$

$$
\frac {\partial V _ {i}}{\partial \sigma} = V (\sigma) - V (\sigma - 1), \mathrm {o r}
$$

$$
\frac {\partial V _ {i}}{\partial \sigma} = \frac {V (\sigma + \epsilon) - V (\sigma)}{\epsilon}, \text {w h e r e} 0 <   | \epsilon | \leq 1.
$$

# D. Interest Rate risk

# D.1 Interest Rate - Risk weights

32. The set of risk-free yield curves within each currency is considered to be a separate bucket.  
33. The risk weights  $RW_{k}$  are set out in the following tables:

(1) There is one table for regular volatility currencies, which are defined to be: the US Dollar (USD), Euro (EUR), British Pound (GBP), Swiss Franc (CHF), Australian Dollar (AUD), New Zealand Dollar (NZD), Canadian Dollar (CAD), Swedish Krona (SEK), Norwegian Krone (NOK), Danish Krona (DKK), Hong Kong Dollar (HKD), South Korean Won (KRW), Singapore Dollar (SGD), and Taiwanese Dollar (TWD).  
(2) There is a second table for low-volatility currencies, which are defined to be the Japanese Yen (JPY) only.  
(3) There is a third table for high-volatility currencies, which are defined to be all other currencies.

Table 1: Risk weights per vertex (regular currencies)  

<table><tr><td>2w</td><td>1m</td><td>3m</td><td>6m</td><td>1yr</td><td>2yr</td><td>3yr</td><td>5yr</td><td>10yr</td><td>15yr</td><td>20yr</td><td>30yr</td></tr><tr><td>109</td><td>105</td><td>90</td><td>71</td><td>66</td><td>66</td><td>64</td><td>60</td><td>60</td><td>61</td><td>61</td><td>67</td></tr></table>

Table 2: Risk weights per vertex (low-volatility currencies)  

<table><tr><td>2w</td><td>1m</td><td>3m</td><td>6m</td><td>1yr</td><td>2yr</td><td>3yr</td><td>5yr</td><td>10yr</td><td>15yr</td><td>20yr</td><td>30yr</td></tr><tr><td>15</td><td>18</td><td>9</td><td>11</td><td>13</td><td>15</td><td>19</td><td>23</td><td>23</td><td>22</td><td>22</td><td>23</td></tr></table>

Table 3: Risk weights per vertex (high-volatility currencies)  

<table><tr><td>2w</td><td>1m</td><td>3m</td><td>6m</td><td>1yr</td><td>2yr</td><td>3yr</td><td>5yr</td><td>10yr</td><td>15yr</td><td>20yr</td><td>30yr</td></tr><tr><td>163</td><td>109</td><td>87</td><td>89</td><td>102</td><td>96</td><td>101</td><td>97</td><td>97</td><td>102</td><td>106</td><td>101</td></tr></table>

The risk weight for any currency's inflation rate is 61.

The risk weight for any currency's cross-currency basis swap spread is 21.

34. The historical volatility ratio,  $HVR$ , for the interest-rate risk class is 0.47.  
35. The vega risk weight, VRW, for the Interest Rate risk class is 0.23.

# D.2 Interest Rate - Correlations

36. The correlation matrix below for risk exposures should be used

Correlations for aggregated weighted sensitivities or risk exposures

<table><tr><td></td><td>2w</td><td>1m</td><td>3m</td><td>6m</td><td>1yr</td><td>2yr</td><td>3yr</td><td>5yr</td><td>10yr</td><td>15yr</td><td>20yr</td><td>30yr</td></tr><tr><td>2w</td><td></td><td>77%</td><td>67%</td><td>59%</td><td>48%</td><td>39%</td><td>34%</td><td>30%</td><td>25%</td><td>23%</td><td>21%</td><td>20%</td></tr><tr><td>1m</td><td>77%</td><td></td><td>84%</td><td>74%</td><td>56%</td><td>43%</td><td>36%</td><td>31%</td><td>26%</td><td>21%</td><td>19%</td><td>19%</td></tr><tr><td>3m</td><td>67%</td><td>84%</td><td></td><td>88%</td><td>69%</td><td>55%</td><td>47%</td><td>40%</td><td>34%</td><td>27%</td><td>25%</td><td>25%</td></tr><tr><td>6m</td><td>59%</td><td>74%</td><td>88%</td><td></td><td>86%</td><td>73%</td><td>65%</td><td>57%</td><td>49%</td><td>40%</td><td>38%</td><td>37%</td></tr><tr><td>1yr</td><td>48%</td><td>56%</td><td>69%</td><td>86%</td><td></td><td>94%</td><td>87%</td><td>79%</td><td>68%</td><td>60%</td><td>57%</td><td>55%</td></tr><tr><td>2yr</td><td>39%</td><td>43%</td><td>55%</td><td>73%</td><td>94%</td><td></td><td>96%</td><td>91%</td><td>80%</td><td>74%</td><td>70%</td><td>69%</td></tr><tr><td>3yr</td><td>34%</td><td>36%</td><td>47%</td><td>65%</td><td>87%</td><td>96%</td><td></td><td>97%</td><td>88%</td><td>81%</td><td>77%</td><td>76%</td></tr><tr><td>5yr</td><td>30%</td><td>31%</td><td>40%</td><td>57%</td><td>79%</td><td>91%</td><td>97%</td><td></td><td>95%</td><td>90%</td><td>86%</td><td>85%</td></tr><tr><td>10yr</td><td>25%</td><td>26%</td><td>34%</td><td>49%</td><td>68%</td><td>80%</td><td>88%</td><td>95%</td><td></td><td>97%</td><td>94%</td><td>94%</td></tr><tr><td>15yr</td><td>23%</td><td>21%</td><td>27%</td><td>40%</td><td>60%</td><td>74%</td><td>81%</td><td>90%</td><td>97%</td><td></td><td>98%</td><td>97%</td></tr><tr><td>20yr</td><td>21%</td><td>19%</td><td>25%</td><td>38%</td><td>57%</td><td>70%</td><td>77%</td><td>86%</td><td>94%</td><td>98%</td><td></td><td>99%</td></tr><tr><td>30yr</td><td>20%</td><td>19%</td><td>25%</td><td>37%</td><td>55%</td><td>69%</td><td>76%</td><td>85%</td><td>94%</td><td>97%</td><td>99%</td><td></td></tr></table>

For sub-curves, the correlation  $\phi_{i,j}$  between any two sub-curves of the same currency is (to one decimal place)  $99.3\%$

For aggregated weighted sensitivities or risk exposures, the correlation between the inflation rate and any yield for the same currency (and the correlation between the inflation volatility and any interest-rate volatility for the same currency) is  $24\%$

For aggregated weighted sensitivities or risk exposures, the correlation between the cross-currency basis swap spread and any yield or inflation rate for the same currency is  $4\%$

37. The parameter  $\gamma_{bc} = 32\%$  should be used for aggregating across different currencies.

# E. Credit Qualifying risk

# E.1 Credit Qualifying - Risk weights

38. Sensitivities or risk exposures to an issuer/seniority should first be assigned to a bucket according to the following table:

<table><tr><td>Bucket number</td><td>Credit quality</td><td>Sector</td></tr><tr><td>1</td><td rowspan="6">Investment grade (IG)</td><td>Sovereigns including central banks</td></tr><tr><td>2</td><td>Financials including government-backed financials</td></tr><tr><td>3</td><td>Basic materials, energy, industrials</td></tr><tr><td>4</td><td>Consumer</td></tr><tr><td>5</td><td>Technology, telecommunications</td></tr><tr><td>6</td><td>Health care, utilities, local government, government-backed corporates (non-financial)</td></tr><tr><td>7</td><td rowspan="6">High yield (HY) &amp; non-rated (NR)</td><td>Sovereigns including central banks</td></tr><tr><td>8</td><td>Financials including government backed financials</td></tr><tr><td>9</td><td>Basic materials, energy, industrials</td></tr><tr><td>10</td><td>Consumer</td></tr><tr><td>11</td><td>Technology, telecommunications</td></tr><tr><td>12</td><td>Health care, utilities, local government, government-backed corporates (non-financial)</td></tr><tr><td colspan="3">Residual</td></tr></table>

Sensitivities must be distinguished depending on the payment currency of the trade (such as Quanto CDS and non-Quanto CDS). No initial netting or aggregation is applied between position sensitivities from different payment currencies (except as then described in paragraph 42).

39. The same risk weight should be used for all vertices (1yr, 2yr, 3yr, 5yr, 10yr), according to bucket, as set out in the following table:

<table><tr><td>Bucket</td><td>Risk weight</td></tr><tr><td>1</td><td>75</td></tr><tr><td>2</td><td>90</td></tr><tr><td>3</td><td>84</td></tr><tr><td>4</td><td>54</td></tr><tr><td>5</td><td>62</td></tr><tr><td>6</td><td>48</td></tr><tr><td>7</td><td>185</td></tr><tr><td>8</td><td>343</td></tr><tr><td>9</td><td>255</td></tr><tr><td>10</td><td>250</td></tr><tr><td>11</td><td>214</td></tr><tr><td>12</td><td>173</td></tr><tr><td>Residual</td><td>343</td></tr></table>

40. The vega risk weight,  $VRW$ , for the Credit risk class is 0.76.

41. The Base Correlation risk weight is 10 for all index families.

# E.2 Credit Qualifying – Correlations

42. The correlation parameters  $\rho_{kl}$  applying to sensitivity or risk exposure pairs within the same bucket are set out in the following table:

<table><tr><td></td><td>Same issuer/seniority, different vertex or currency</td><td>Different issuer/seniority</td></tr><tr><td>Aggregate sensitivities</td><td>93%</td><td>46%</td></tr><tr><td>Residual bucket</td><td>50%</td><td>50%</td></tr></table>

Herein "currency" refers to the payment currency of the sensitivity if there are sensitivities to multiple payment currencies (such as Quanto CDS and non-Quanto CDS), which will not be fully offset.

The correlation parameter  $\rho_{kl}$  applying to Base Correlation risks across different index families is  $29\%$ .

43. The correlation parameters  $\gamma_{bc}$  applying to sensitivity or risk exposure pairs across different non-residual buckets is set out in the following table:

<table><tr><td>Bucket</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td><td>10</td><td>11</td><td>12</td></tr><tr><td>1</td><td></td><td>38%</td><td>38%</td><td>35%</td><td>37%</td><td>34%</td><td>42%</td><td>32%</td><td>34%</td><td>33%</td><td>34%</td><td>33%</td></tr><tr><td>2</td><td>38%</td><td></td><td>48%</td><td>46%</td><td>48%</td><td>46%</td><td>39%</td><td>40%</td><td>41%</td><td>41%</td><td>43%</td><td>40%</td></tr><tr><td>3</td><td>38%</td><td>48%</td><td></td><td>50%</td><td>51%</td><td>50%</td><td>40%</td><td>39%</td><td>45%</td><td>44%</td><td>47%</td><td>42%</td></tr><tr><td>4</td><td>35%</td><td>46%</td><td>50%</td><td></td><td>50%</td><td>50%</td><td>37%</td><td>37%</td><td>41%</td><td>43%</td><td>45%</td><td>40%</td></tr><tr><td>5</td><td>37%</td><td>48%</td><td>51%</td><td>50%</td><td></td><td>50%</td><td>39%</td><td>38%</td><td>43%</td><td>43%</td><td>46%</td><td>42%</td></tr><tr><td>6</td><td>34%</td><td>46%</td><td>50%</td><td>50%</td><td>50%</td><td></td><td>37%</td><td>35%</td><td>39%</td><td>41%</td><td>44%</td><td>41%</td></tr><tr><td>7</td><td>42%</td><td>39%</td><td>40%</td><td>37%</td><td>39%</td><td>37%</td><td></td><td>33%</td><td>37%</td><td>37%</td><td>35%</td><td>35%</td></tr><tr><td>8</td><td>32%</td><td>40%</td><td>39%</td><td>37%</td><td>38%</td><td>35%</td><td>33%</td><td></td><td>36%</td><td>37%</td><td>37%</td><td>36%</td></tr><tr><td>9</td><td>34%</td><td>41%</td><td>45%</td><td>41%</td><td>43%</td><td>39%</td><td>37%</td><td>36%</td><td></td><td>41%</td><td>40%</td><td>38%</td></tr><tr><td>10</td><td>33%</td><td>41%</td><td>44%</td><td>43%</td><td>43%</td><td>41%</td><td>37%</td><td>37%</td><td>41%</td><td></td><td>41%</td><td>39%</td></tr><tr><td>11</td><td>34%</td><td>43%</td><td>47%</td><td>45%</td><td>46%</td><td>44%</td><td>35%</td><td>37%</td><td>40%</td><td>41%</td><td></td><td>40%</td></tr><tr><td>12</td><td>33%</td><td>40%</td><td>42%</td><td>40%</td><td>42%</td><td>41%</td><td>35%</td><td>36%</td><td>38%</td><td>39%</td><td>40%</td><td></td></tr></table>

# F. Credit Non-Qualifying risk

44. Sensitivities to credit spread risk arising from non-qualifying securitisation positions are treated according to the risk weights and correlations specified in the following paragraphs.

# F.1 Credit Non-Qualifying - Risk weights

45. Sensitivities or risk exposures should first be assigned to a bucket according to the following table:

<table><tr><td>Bucket number</td><td>Credit quality</td><td>Sector</td></tr><tr><td>1</td><td>Investment grade (IG)</td><td>RMBS/CMBS</td></tr><tr><td>2</td><td>High yield (HY) &amp; non-rated (NR)</td><td>RMBS/CMBS</td></tr><tr><td colspan="3">Residual</td></tr></table>

If it is not possible to allocate a sensitivity or risk exposure to one of these buckets (for example, because data on categorical variables is not available), then the position must be allocated to the "Residual bucket".

46. The risk weights are set out in the following table:

<table><tr><td>Bucket number</td><td>Risk weight</td></tr><tr><td>1</td><td>280</td></tr><tr><td>2</td><td>1,300</td></tr><tr><td>Residual</td><td>1,300</td></tr></table>

47. The vega risk weight,  $VRW$ , for Credit Non-Qualifying is 0.76.

# F.2 Credit Non-Qualifying – Correlations

48. For the other buckets, the correlation parameters  $\rho_{kl}$  applying to sensitivity or risk exposure pairs within the same bucket are set out in the following table:

<table><tr><td></td><td>Same group name (such as CMBX, ABX)</td><td>Different group name</td></tr><tr><td>Aggregate sensitivities</td><td>83%</td><td>32%</td></tr><tr><td>Residual bucket</td><td>50%</td><td>50%</td></tr></table>

49. The correlation parameters  $\gamma_{bc}$  applying to sensitivity or risk exposure pairs across different buckets is set out in the following table:

<table><tr><td></td><td>Correlation</td></tr><tr><td>Non-residual bucket to non-residual bucket</td><td>43%</td></tr></table>

# G. Equity risk

# G.1 Equity - Risk weights

50. Sensitivities or risk exposures should first be assigned to a bucket according to the buckets defined in the following table:

<table><tr><td>Bucket number</td><td>Size</td><td>Region</td><td>Sector</td></tr><tr><td>1</td><td rowspan="8">Large</td><td rowspan="4">Emerging markets</td><td>Consumer goods and services, transportation and storage, administrative and support service activities, healthcare, utilities</td></tr><tr><td>2</td><td>Telecommunications, industrials</td></tr><tr><td>3</td><td>Basic materials, energy, agriculture, manufacturing, mining and quarrying</td></tr><tr><td>4</td><td>Financials including gov’t-backed financials, real estate activities, technology</td></tr><tr><td>5</td><td rowspan="4">Developed markets</td><td>Consumer goods and services, transportation and storage, administrative and support service activities, healthcare, utilities</td></tr><tr><td>6</td><td>Telecommunications, industrials</td></tr><tr><td>7</td><td>Basic materials, energy, agriculture, manufacturing, mining and quarrying</td></tr><tr><td>8</td><td>Financials including gov’t-backed financials, real estate activities, technology</td></tr><tr><td>9</td><td rowspan="2">Small</td><td>Emerging markets</td><td>All sectors</td></tr><tr><td>10</td><td>Developed markets</td><td>All sectors</td></tr><tr><td>11</td><td>All</td><td>All</td><td>Indexes, Funds, ETFs</td></tr><tr><td>12</td><td>All</td><td>All</td><td>Volatility Indexes</td></tr></table>

51. "Large" is defined as a market capitalisation equal to or greater than USD 2 billion and "small" is defined as a market capitalisation of less than USD 2 billion.  
52. "Market capitalisation" is defined as the sum of the market capitalisations of the same legal entity or group of legal entities across all stock markets globally.  
53. The developed markets are defined as: Canada, US, Mexico, the euro area, the non-euro area western European countries (the United Kingdom, Norway, Sweden, Denmark, and Switzerland), Japan, Oceania (Australia and New Zealand), Singapore and Hong Kong.  
54. The sectors definition is the one generally used in the market. When allocating an equity position to a particular bucket, the bank must prove that the equity issuer's most material activity indeed corresponds to the bucket's definition. Acceptable proofs might be external providers' information, or internal analysis.  
55. For multinational multi-sector equity issuers, the allocation to a particular bucket must be done according to the most material region and sector the issuer operates in.  
56. If it is not possible to allocate a position to one of these buckets (for example, because data on categorical variables is not available) then the position must be allocated to a "Residual bucket". Risk weights should be assigned to each notional position as in the following table:

<table><tr><td>Bucket</td><td>Risk Weight</td></tr><tr><td>1</td><td>30</td></tr><tr><td>2</td><td>33</td></tr><tr><td>3</td><td>36</td></tr><tr><td>4</td><td>29</td></tr><tr><td>5</td><td>26</td></tr><tr><td>6</td><td>25</td></tr><tr><td>7</td><td>34</td></tr><tr><td>8</td><td>28</td></tr><tr><td>9</td><td>36</td></tr><tr><td>10</td><td>50</td></tr><tr><td>11</td><td>19</td></tr><tr><td>12</td><td>19</td></tr><tr><td>Residual</td><td>50</td></tr></table>

57. The historical volatility ratio,  $HVR$ , for the equity risk class is  $60\%$ .  
58. The vega risk weight,  $\text{VRW}$ , for the equity risk class is 0.45 for all buckets except bucket 12 for which the vega risk weight is 0.96.

# G.2 Equity - Correlations

59. The correlation parameters  $\rho_{kl}$  applying to sensitivity or risk exposure pairs within the same bucket are set out in the following table:

<table><tr><td>Bucket number</td><td>Correlation</td></tr><tr><td>1</td><td>18%</td></tr><tr><td>2</td><td>20%</td></tr><tr><td>3</td><td>28%</td></tr><tr><td>4</td><td>24%</td></tr><tr><td>5</td><td>25%</td></tr><tr><td>6</td><td>36%</td></tr><tr><td>7</td><td>35%</td></tr><tr><td>8</td><td>37%</td></tr><tr><td>9</td><td>23%</td></tr><tr><td>10</td><td>27%</td></tr><tr><td>11</td><td>45%</td></tr><tr><td>12</td><td>45%</td></tr><tr><td>Residual</td><td>0%</td></tr></table>

60. The correlation parameters  $\gamma_{bc}$  applying to sensitivity or risk exposure pairs across different non-residual buckets are set out in the following table:

<table><tr><td>Bucket</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td><td>10</td><td>11</td><td>12</td></tr><tr><td>1</td><td></td><td>18%</td><td>19%</td><td>19%</td><td>14%</td><td>16%</td><td>15%</td><td>16%</td><td>18%</td><td>12%</td><td>19%</td><td>19%</td></tr><tr><td>2</td><td>18%</td><td></td><td>22%</td><td>21%</td><td>15%</td><td>18%</td><td>17%</td><td>19%</td><td>20%</td><td>14%</td><td>21%</td><td>21%</td></tr><tr><td>3</td><td>19%</td><td>22%</td><td></td><td>22%</td><td>13%</td><td>16%</td><td>18%</td><td>17%</td><td>22%</td><td>13%</td><td>20%</td><td>20%</td></tr><tr><td>4</td><td>19%</td><td>21%</td><td>22%</td><td></td><td>17%</td><td>22%</td><td>22%</td><td>23%</td><td>22%</td><td>17%</td><td>26%</td><td>26%</td></tr><tr><td>5</td><td>14%</td><td>15%</td><td>13%</td><td>17%</td><td></td><td>29%</td><td>26%</td><td>29%</td><td>14%</td><td>24%</td><td>32%</td><td>32%</td></tr><tr><td>6</td><td>16%</td><td>18%</td><td>16%</td><td>22%</td><td>29%</td><td></td><td>34%</td><td>36%</td><td>17%</td><td>30%</td><td>39%</td><td>39%</td></tr><tr><td>7</td><td>15%</td><td>17%</td><td>18%</td><td>22%</td><td>26%</td><td>34%</td><td></td><td>33%</td><td>16%</td><td>28%</td><td>36%</td><td>36%</td></tr><tr><td>8</td><td>16%</td><td>19%</td><td>17%</td><td>23%</td><td>29%</td><td>36%</td><td>33%</td><td></td><td>17%</td><td>29%</td><td>40%</td><td>40%</td></tr><tr><td>9</td><td>18%</td><td>20%</td><td>22%</td><td>22%</td><td>14%</td><td>17%</td><td>16%</td><td>17%</td><td></td><td>13%</td><td>21%</td><td>21%</td></tr><tr><td>10</td><td>12%</td><td>14%</td><td>13%</td><td>17%</td><td>24%</td><td>30%</td><td>28%</td><td>29%</td><td>13%</td><td></td><td>30%</td><td>30%</td></tr><tr><td>11</td><td>19%</td><td>21%</td><td>20%</td><td>26%</td><td>32%</td><td>39%</td><td>36%</td><td>40%</td><td>21%</td><td>30%</td><td></td><td>45%</td></tr><tr><td>12</td><td>19%</td><td>21%</td><td>20%</td><td>26%</td><td>32%</td><td>39%</td><td>36%</td><td>40%</td><td>21%</td><td>30%</td><td>45%</td><td></td></tr></table>

# H. Commodity risk

# H.1 Commodity - Risk weights

61. The risk weights depend on the commodity type; they are set out in the following table:

<table><tr><td>Bucket</td><td>Commodity</td><td>Risk Weight</td></tr><tr><td>1</td><td>Coal</td><td>48</td></tr><tr><td>2</td><td>Crude</td><td>29</td></tr><tr><td>3</td><td>Light Ends</td><td>33</td></tr><tr><td>4</td><td>Middle Distillates</td><td>25</td></tr><tr><td>5</td><td>Heavy Distillates</td><td>35</td></tr><tr><td>6</td><td>North America Natural Gas</td><td>30</td></tr><tr><td>7</td><td>European Natural Gas</td><td>60</td></tr><tr><td>8</td><td>North American Power</td><td>52</td></tr><tr><td>9</td><td>European Power and Carbon</td><td>68</td></tr><tr><td>10</td><td>Freight</td><td>63</td></tr><tr><td>11</td><td>Base Metals</td><td>21</td></tr><tr><td>12</td><td>Precious Metals</td><td>21</td></tr><tr><td>13</td><td>Grains and Oilseed</td><td>15</td></tr><tr><td>14</td><td>Softs and Other Agriculturals</td><td>16</td></tr><tr><td>15</td><td>Livestock and Dairy</td><td>13</td></tr><tr><td>16</td><td>Other</td><td>68</td></tr><tr><td>17</td><td>Indexes</td><td>17</td></tr></table>

62. The historical volatility ratio,  $HVR$ , for the commodity risk class is 74%

63. The vega risk weight,  $VRW$ , for the commodity risk class is 0.55

# H.2 Commodity - Correlations

64. The correlation parameters  $\rho_{kl}$  applying to sensitivity or risk exposure pairs within the same bucket are set out in the following table:

<table><tr><td>Bucket</td><td>Correlation</td></tr><tr><td>1</td><td>83%</td></tr><tr><td>2</td><td>97%</td></tr><tr><td>3</td><td>93%</td></tr><tr><td>4</td><td>97%</td></tr><tr><td>5</td><td>98%</td></tr><tr><td>6</td><td>90%</td></tr><tr><td>7</td><td>98%</td></tr><tr><td>8</td><td>49%</td></tr><tr><td>9</td><td>80%</td></tr><tr><td>10</td><td>46%</td></tr><tr><td>11</td><td>58%</td></tr><tr><td>12</td><td>53%</td></tr><tr><td>13</td><td>62%</td></tr><tr><td>14</td><td>16%</td></tr><tr><td>15</td><td>18%</td></tr><tr><td>16</td><td>0%</td></tr><tr><td>17</td><td>38%</td></tr></table>

65. The correlation parameters  $\gamma_{bc}$  applying to sensitivity or risk exposure pairs across different buckets are set out in the following table:

<table><tr><td>Buckets</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td><td>10</td><td>11</td><td>12</td><td>13</td><td>14</td><td>15</td><td>16</td><td>17</td></tr><tr><td>1</td><td></td><td>22%</td><td>18%</td><td>21%</td><td>20%</td><td>24%</td><td>49%</td><td>16%</td><td>38%</td><td>14%</td><td>10%</td><td>2%</td><td>12%</td><td>11%</td><td>2%</td><td>0%</td><td>17%</td></tr><tr><td>2</td><td>22%</td><td></td><td>92%</td><td>90%</td><td>88%</td><td>25%</td><td>8%</td><td>19%</td><td>17%</td><td>17%</td><td>42%</td><td>28%</td><td>36%</td><td>27%</td><td>20%</td><td>0%</td><td>64%</td></tr><tr><td>3</td><td>18%</td><td>92%</td><td></td><td>87%</td><td>84%</td><td>16%</td><td>7%</td><td>15%</td><td>10%</td><td>18%</td><td>33%</td><td>22%</td><td>27%</td><td>23%</td><td>16%</td><td>0%</td><td>54%</td></tr><tr><td>4</td><td>21%</td><td>90%</td><td>87%</td><td></td><td>77%</td><td>19%</td><td>11%</td><td>18%</td><td>16%</td><td>14%</td><td>32%</td><td>22%</td><td>28%</td><td>22%</td><td>11%</td><td>0%</td><td>58%</td></tr><tr><td>5</td><td>20%</td><td>88%</td><td>84%</td><td>77%</td><td></td><td>19%</td><td>9%</td><td>12%</td><td>13%</td><td>18%</td><td>42%</td><td>34%</td><td>32%</td><td>29%</td><td>13%</td><td>0%</td><td>59%</td></tr><tr><td>6</td><td>24%</td><td>25%</td><td>16%</td><td>19%</td><td>19%</td><td></td><td>31%</td><td>62%</td><td>23%</td><td>10%</td><td>21%</td><td>5%</td><td>18%</td><td>10%</td><td>8%</td><td>0%</td><td>28%</td></tr><tr><td>7</td><td>49%</td><td>8%</td><td>7%</td><td>11%</td><td>9%</td><td>31%</td><td></td><td>21%</td><td>79%</td><td>17%</td><td>10%</td><td>-8%</td><td>10%</td><td>7%</td><td>-2%</td><td>0%</td><td>13%</td></tr><tr><td>8</td><td>16%</td><td>19%</td><td>15%</td><td>18%</td><td>12%</td><td>62%</td><td>21%</td><td></td><td>16%</td><td>8%</td><td>13%</td><td>-7%</td><td>7%</td><td>5%</td><td>2%</td><td>0%</td><td>19%</td></tr><tr><td>9</td><td>38%</td><td>17%</td><td>10%</td><td>16%</td><td>13%</td><td>23%</td><td>79%</td><td>16%</td><td></td><td>15%</td><td>9%</td><td>-6%</td><td>6%</td><td>6%</td><td>1%</td><td>0%</td><td>16%</td></tr><tr><td>10</td><td>14%</td><td>17%</td><td>18%</td><td>14%</td><td>18%</td><td>10%</td><td>17%</td><td>8%</td><td>15%</td><td></td><td>16%</td><td>9%</td><td>14%</td><td>9%</td><td>3%</td><td>0%</td><td>11%</td></tr><tr><td>11</td><td>10%</td><td>42%</td><td>33%</td><td>32%</td><td>42%</td><td>21%</td><td>10%</td><td>13%</td><td>9%</td><td>16%</td><td></td><td>36%</td><td>30%</td><td>25%</td><td>18%</td><td>0%</td><td>37%</td></tr><tr><td>12</td><td>2%</td><td>28%</td><td>22%</td><td>22%</td><td>34%</td><td>5%</td><td>-8%</td><td>-7%</td><td>-6%</td><td>9%</td><td>36%</td><td></td><td>20%</td><td>18%</td><td>11%</td><td>0%</td><td>26%</td></tr><tr><td>13</td><td>12%</td><td>36%</td><td>27%</td><td>28%</td><td>32%</td><td>18%</td><td>10%</td><td>7%</td><td>6%</td><td>14%</td><td>30%</td><td>20%</td><td></td><td>28%</td><td>19%</td><td>0%</td><td>39%</td></tr><tr><td>14</td><td>11%</td><td>27%</td><td>23%</td><td>22%</td><td>29%</td><td>10%</td><td>7%</td><td>5%</td><td>6%</td><td>9%</td><td>25%</td><td>18%</td><td>28%</td><td></td><td>13%</td><td>0%</td><td>26%</td></tr><tr><td>15</td><td>2%</td><td>20%</td><td>16%</td><td>11%</td><td>13%</td><td>8%</td><td>-2%</td><td>2%</td><td>1%</td><td>3%</td><td>18%</td><td>11%</td><td>19%</td><td>13%</td><td></td><td>0%</td><td>21%</td></tr><tr><td>16</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td></td><td>0%</td></tr><tr><td>17</td><td>17%</td><td>64%</td><td>54%</td><td>58%</td><td>59%</td><td>28%</td><td>13%</td><td>19%</td><td>16%</td><td>11%</td><td>37%</td><td>26%</td><td>39%</td><td>26%</td><td>21%</td><td>0%</td><td></td></tr></table>

# I. Foreign Exchange risk

# I.1 Foreign Exchange - Risk weights

66. All Foreign Exchange sensitivities are considered to be within a single bucket within the FX risk class, so no inter-bucket aggregation is necessary. Note that the cross-bucket Curvature calculations of paragraph 11(d) are still required on the single bucket.  
67. The group of high FX volatility currencies is defined to be: BRL, RUB, TRY.  
68. The group of regular FX volatility currencies is defined to be all other currencies.  
69. The risk weight for a currency depends on the group of the calculation currency, as well as the group of the currency concerned. There should be no FX risk factor for the calculation currency (or equivalently, its risk weight should be set to zero). The risk weight for a given currency and calculation currency is given by the following table where the row corresponds to the FX volatility group of the given currency and the column corresponds to the FX volatility group of the calculation currency:

<table><tr><td>FX Volatility group</td><td>Regular</td><td>High</td></tr><tr><td>Regular</td><td>7.4</td><td>14.7</td></tr><tr><td>High</td><td>14.7</td><td>21.4</td></tr></table>

70. The historical volatility ratio, HVR, for the FX risk class is 0.57.  
71. The vega risk weight, VRW, for FX volatility is 0.48.

# 1.2 Foreign Exchange - Correlations

72. The correlation parameter  $\rho_{kl}$  applying to pairs of FX sensitivities or risk exposures are set out in the following tables. Since each FX risk factor is the exchange rate of a particular currency against the calculation currency, the correlation between two FX risk factors depends on the FX volatility group of the two particular currencies as well as the FX volatility group of the calculation currency. Each table shows the correlation between two FX risk factors, depending on the FX volatility group of each risk factor's currency.

The first table shows the correlations between two FX risk factors if the calculation currency is in the regular FX volatility group:

<table><tr><td>FX Volatility group</td><td>Regular</td><td>High</td></tr><tr><td>Regular</td><td>50%</td><td>25%</td></tr><tr><td>High</td><td>25%</td><td>-5%</td></tr></table>

The second table shows the correlations between two FX risk factors if the calculation currency is in the high FX volatility group:

<table><tr><td>FX Volatility group</td><td>Regular</td><td>High</td></tr><tr><td>Regular</td><td>88%</td><td>72%</td></tr><tr><td>High</td><td>72%</td><td>50%</td></tr></table>

73. For the purposes of correlating pairs of FX volatility and curvature risk factors, the correlation shall be taken to be 0.5.

# J. Concentration Thresholds

The concentration thresholds in this section are defined for the asset-class-specific buckets specified in Sections E, G, and H. For those cases in which the same concentration threshold applies to a related range of buckets, the tables in this section specify the precise range of applicable buckets in the Bucket column and give a narrative description of that group of buckets in the Risk Group column.

# J.1 Interest Rate risk - Delta Concentration Thresholds

74. The delta concentration thresholds for interest rate risk (inclusive of inflation risk) are given by currency group:

<table><tr><td>Currency Risk Group</td><td>Concentration threshold (USD mm/bp)</td></tr><tr><td>High volatility</td><td>30</td></tr><tr><td>Regular volatility, well-traded</td><td>330</td></tr><tr><td>Regular volatility, less well-traded</td><td>130</td></tr><tr><td>Low volatility</td><td>61</td></tr></table>

75. The currency risk groups used in establishing concentration thresholds for Interest Rate Risk are as follows:

(1) High volatility: All other currencies  
(2) Regular volatility, well-traded: USD; EUR; GBP  
(3) Regular volatility, less well-traded: AUD; CAD; CHF; DKK; HKD; KRW; NOK; NZD; SEK; SGD; TWD  
(4) Low volatility: JPY

# J.2 Credit spread risk - Delta Concentration Thresholds

76. The delta concentration thresholds for credit spread risk are given by credit risk group and bucket:

<table><tr><td>Bucket(s)</td><td>Credit Risk Group</td><td>Concentration threshold (USD mm/bp)</td></tr><tr><td colspan="2">Qualifying</td><td></td></tr><tr><td>1, 7</td><td>Sovereigns including central banks</td><td>1.0</td></tr><tr><td>2-6, 8-12</td><td>Corporate entities</td><td>0.17</td></tr><tr><td>Residual</td><td>Not classified</td><td>0.17</td></tr><tr><td colspan="2">Non-Qualifying</td><td></td></tr><tr><td>1</td><td>IG (RMBS and CMBS)</td><td>9.5</td></tr><tr><td>2</td><td>HY/Non-rated (RMBS and CMBS)</td><td>0.50</td></tr><tr><td>Residual</td><td>Not classified</td><td>0.50</td></tr></table>

# J.3 Equity risk - Delta Concentration Thresholds

77. The delta concentration thresholds for equity risk are given by bucket:

<table><tr><td>Bucket(s)</td><td>Equity Risk Group</td><td>Concentration threshold (USD mm/%)</td></tr><tr><td>1-4</td><td>Emerging Markets – Large Cap</td><td>3</td></tr><tr><td>5-8</td><td>Developed Markets – Large Cap</td><td>12</td></tr><tr><td>9</td><td>Emerging Markets – Small Cap</td><td>0.64</td></tr><tr><td>10</td><td>Developed Markets – Small Cap</td><td>0.37</td></tr><tr><td>11-12</td><td>Indexes, Funds, ETFs, Volatility Indexes</td><td>810</td></tr><tr><td>Residual</td><td>Not classified</td><td>0.37</td></tr></table>

# J.4 Commodity risk - Delta Concentration Thresholds

78. The delta concentration thresholds for commodity risk are given by bucket:

<table><tr><td>Bucket</td><td>CT bucket</td><td>Concentration threshold (USD mm/%)</td></tr><tr><td>1</td><td>Coal</td><td>310</td></tr><tr><td>2</td><td>Crude Oil</td><td>2,100</td></tr><tr><td>3-5</td><td>Oil Fractions</td><td>1,700</td></tr><tr><td>6-7</td><td>Natural gas</td><td>2,800</td></tr><tr><td>8-9</td><td>Power</td><td>2,700</td></tr><tr><td>10</td><td>Freight, Dry or Wet</td><td>52</td></tr><tr><td>11</td><td>Base metals</td><td>530</td></tr><tr><td>12</td><td>Precious Metals</td><td>1,300</td></tr><tr><td>13-15</td><td>Agricultural</td><td>100</td></tr><tr><td>16</td><td>Other</td><td>52</td></tr><tr><td>17</td><td>Indices</td><td>4,000</td></tr></table>

# J.5 FX risk - Delta Concentration Thresholds

79. The delta concentration thresholds for FX risk are given by currency risk group:

<table><tr><td>FX Risk Group</td><td>Concentration threshold (USD mm/%)</td></tr><tr><td>Category 1</td><td>3,300</td></tr><tr><td>Category 2</td><td>880</td></tr><tr><td>Category 3</td><td>170</td></tr></table>

80. Currencies were placed in three categories as for delta risk weights, constituted as follows:

Category 1 - Significantly material: USD, EUR, JPY, GBP, AUD, CHF, CAD

Category 2 - Frequently traded: BRL, CNY, HKD, INR, KRW, MXN, NOK, NZD, RUB, SEK, SGD, TRY, ZAR

Category 3 - Others: All other currencies

# J.6 Interest Rate risk - Vega Concentration Thresholds

81. The vega concentration thresholds for Interest Rate risk are:

<table><tr><td>Currency Risk Group</td><td>Concentration threshold (USD mm)</td></tr><tr><td>High volatility</td><td>74</td></tr><tr><td>Regular volatility, well traded</td><td>4,900</td></tr><tr><td>Regular volatility, less well traded</td><td>520</td></tr><tr><td>Low volatility</td><td>970</td></tr></table>

82. The Currency risk groups used in establishing concentration thresholds for Interest Rate risk are identified in paragraph 75 above.

# J.7 Credit spread risk - Vega Concentration Thresholds

83. The vega concentration thresholds for Credit spread risk (including the residual buckets) are:

<table><tr><td>Credit Risk Group</td><td>Concentration threshold (USD mm)</td></tr><tr><td>Qualifying</td><td>360</td></tr><tr><td>Non Qualifying</td><td>70</td></tr></table>

# J.8 Equity risk - Vega Concentration Thresholds

84. The vega concentration thresholds for equity risk are:

<table><tr><td>Bucket</td><td>Equity Risk Group</td><td>Concentration threshold (USD mm)</td></tr><tr><td>1-4</td><td>Emerging Markets – Large Cap</td><td>210</td></tr><tr><td>5-8</td><td>Developed Markets – Large Cap</td><td>1,300</td></tr><tr><td>9</td><td>Emerging Markets – Small Cap</td><td>39</td></tr><tr><td>10</td><td>Developed Markets – Small Cap</td><td>190</td></tr><tr><td>11-12</td><td>Indexes, Funds, ETFs, Volatility Indexes</td><td>6,400</td></tr><tr><td>Residual</td><td>Not classified</td><td>39</td></tr></table>

# J.9 Commodities risk - Vega Concentration Thresholds

85. The vega concentration thresholds for Commodities vega risk are:

<table><tr><td>Bucket</td><td>Commodity Risk Group</td><td>Concentration threshold (USD mm)</td></tr><tr><td>1</td><td>Coal</td><td>390</td></tr><tr><td>2</td><td>Crude Oil</td><td>2,900</td></tr><tr><td>3-5</td><td>Oil fractions</td><td>310</td></tr><tr><td>6-7</td><td>Natural gas</td><td>6,300</td></tr><tr><td>8-9</td><td>Power</td><td>1,200</td></tr><tr><td>10</td><td>Freight, Dry or Wet</td><td>120</td></tr><tr><td>11</td><td>Base metals</td><td>390</td></tr><tr><td>12</td><td>Precious Metals</td><td>1,300</td></tr><tr><td>13-15</td><td>Agricultural</td><td>590</td></tr><tr><td>16</td><td>Other</td><td>69</td></tr><tr><td>17</td><td>Indices</td><td>69</td></tr></table>

# J.10 FX risk - Vega Concentration Thresholds

86. The vega concentration thresholds for FX risk are:

<table><tr><td>FX Risk Group</td><td>Concentration threshold (USD mm)</td></tr><tr><td>Category 1 - Category 1</td><td>2,800</td></tr><tr><td>Category 1 - Category 2</td><td>1,400</td></tr><tr><td>Category 1 - Category 3</td><td>590</td></tr><tr><td>Category 2 - Category 2</td><td>520</td></tr><tr><td>Category 2 - Category 3</td><td>340</td></tr><tr><td>Category 3 - Category 3</td><td>210</td></tr></table>

87. The Currency Categories used in establishing concentration thresholds for FX risk are identified in paragraph 80 above.

# K. Correlation between risk classes within product classes

88. The correlation parameters  $\psi_{rs}$  applying to initial margin risk classes within a single product class are set out in the following table:

<table><tr><td>Risk Class</td><td>Interest Rate</td><td>Credit Qualifying</td><td>Credit Non-Qualifying</td><td>Equity</td><td>Commodity</td><td>FX</td></tr><tr><td>Interest Rate</td><td></td><td>4%</td><td>4%</td><td>7%</td><td>37%</td><td>14%</td></tr><tr><td>Credit Qualifying</td><td>4%</td><td></td><td>54%</td><td>70%</td><td>27%</td><td>37%</td></tr><tr><td>Credit Non-qualifying</td><td>4%</td><td>54%</td><td></td><td>46%</td><td>24%</td><td>15%</td></tr><tr><td>Equity</td><td>7%</td><td>70%</td><td>46%</td><td></td><td>35%</td><td>39%</td></tr><tr><td>Commodity</td><td>37%</td><td>27%</td><td>24%</td><td>35%</td><td></td><td>35%</td></tr><tr><td>FX</td><td>14%</td><td>37%</td><td>15%</td><td>39%</td><td>35%</td><td></td></tr></table>

# L. Additional Initial Margin Formulas

Standardised formulas for calculating Additional Initial Margin are below:

Additional Initial Margin

$$
\begin{array}{l} = A d d O n I M + \left(M S _ {R a t e s F X} - 1\right) S I M M _ {R a t e s F X} + \left(M S _ {C r e d i t} - 1\right) S I M M _ {C r e d i t} \\ + \left(M S _ {E q u i t y} - 1\right) S I M M _ {E q u i t y} + \left(M S _ {C o m m o d i t y} - 1\right) S I M M _ {C o m m o d i t y}. \\ \end{array}
$$

Where AddOn IM is defined as:

$$
A d d O n I M = A d d O n F i x e d + \sum_ {\text {p r o d u c t} p} A d d O n F a c t o r _ {p} N o t i o n a l _ {p},
$$

Where AddOnFixed is a fixed add-on amount, AddOnFactor  $p$  is the add-on factor for each affected product  $p$ , expressed as percentage of the notional (e.g. 5%); and Notional  $p$  is the total notional of the product (sum of the absolute trade notionals). In such use, where a variable notional is involved, the current notional amount should be used.

The four variables  $MS_{RatesFx}$ ,  $MS_{Credit}$ ,  $MS_{Equity}$ ,  $MS_{Commodity}$  are the four "multiplicative scales" for the four product classes (RatesFX, Credit, Equity, Commodity). Their values can be individually specified to be more than 1.0, with 1.0 being the default and minimum value.

See also the SIMM Governance Framework and the associated SIMM Remediation Annex for details of the remediation procedures which can require usage of these additional margin amounts.