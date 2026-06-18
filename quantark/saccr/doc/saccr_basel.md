# Basel Committee on Banking Supervision

![](images/44883c89cbece26166173d4be1e72ef1a947d70fe11747acdb288824297f0d98.jpg)

# The standardised approach for measuring counterparty credit risk exposures

March 2014 (rev. April 2014)

This publication is available on the BIS website (www.bis.org).

© Bank for International Settlements 2014. All rights reserved. Brief excerpts may be reproduced or translated provided the source is stated.

ISBN 978-92-9131-222-1 (print)

ISBN 978-92-9131-223-8 (online)

# Contents

I. Introduction 1

A. Background 1  
B. Introducing the SA-CCR 1  
C. Scope of application 2  
D. Transitional arrangements 3  
E. Examples 3

II. Revisions to Part 2: The First Pillar; Section II: Credit risk – the standardised approach 3  
III. Revisions to Part 2: The First Pillar; Annex 4 Treatment of Counterparty Credit Risk and Cross-Product Netting  
IV. Other revisions to Basel III: A global regulatory framework 21

A. Abbreviations 21  
B. Part 4: Third Pillar; Section II Disclosure requirements 21

Annex 4a Application of the SA-CCR to sample portfolios 22  
Annex 4b Effect of standard margin agreements on the SA-CCR formulation. 31  
Annex 4c Flow chart of steps to calculate [interest rate] add-on 33

# I. Introduction

# A. Background

This document presents the Basel Committee's formulation for its Standardised Approach (SA-CCR) for measuring exposure at default (EAD) for counterparty credit risk (CCR). The SA-CCR will replace both current non-internal models approaches, the Current Exposure Method (CEM) and the Standardised Method (SM).

In formulating the SA-CCR, the Basel Committee's main objectives were to devise an approach that is suitable to be applied to a wide variety of derivatives transactions (margined and unmarked, as well as bilateral and cleared); is capable of being implemented simply and easily; addresses known deficiencies of the CEM and the SM; draws on prudential approaches already available in the Basel framework; minimises discretion used by national authorities and banks; and improves the risk sensitivity of the capital framework without creating undue complexity.

The CEM had been criticised for several limitations, in particular that it did not differentiate between margined and unmargined transactions, that the supervisory add-on factor did not sufficiently capture the level of volatilities as observed over recent stress periods, and the recognition of netting benefits was too simplistic and not reflective of economically meaningful relationships between derivatives positions.

Although being more risk-sensitive than the CEM, the SM was also criticised for several weaknesses. Like the CEM, it did not differentiate between margined and unmargined transactions or sufficiently capture the level of volatilities observed over stress periods in the last five years. In addition, the definition of "hedging set" led to operational complexity resulting in an inability to implement the SM, or implementing it in inconsistent ways. Further, the relationship between current exposure and potential future exposure (PFE) was misrepresented in the SM because only current exposure or PFE was capitalised. Finally, the SM did not provide banks with a true non-internal model alternative for calculating EAD because the SM used internal methods for computing delta-equivalents for non-linear transactions.

# B. Introducing the SA-CCR

The exposures under the SA-CCR consist of two components: replacement cost (RC) and potential future exposure (PFE). Mathematically:

$$
\text {E x p o s u r e} \quad S A = E A D = a l p h a ^ {*} (R C + P F E)
$$

where alpha equals 1.4, which is carried over from the alpha value set by the Basel Committee for the Internal Model Method (IMM). The PFE portion consists of a multiplier that allows for the partial recognition of excess collateral and an aggregate add-on, which is derived from add-ons developed for each asset class (similar to the five asset classes used for the CEM, ie interest rate, foreign exchange, credit, equity and commodity).<sup>1</sup>

The methodology for calculating the add-ons for each asset class hinges on the key concept of a "hedging set". A "hedging set" under the SA-CCR is a set of transactions within a single netting set within which partial or full offsetting is recognised for the purpose of calculating the PFE add-on. The

add-on will vary based on the number of hedging sets that are available within an asset class. These variations are necessary to account for basis risk and differences in correlations within asset classes. The methodologies for calculating the add-ons are summarised below.

- Interest rate derivatives: A hedging set consists of all derivatives that reference interest rates of the same currency such as USD, EUR, JPY, etc. Hedging sets are further divided into maturity categories. Long and short positions in the same hedging set are permitted to fully offset each other within maturity categories; across maturity categories, partial offset is recognised.  
Foreign exchange derivatives: A hedging set consists of derivatives that reference the same foreign exchange currency pair such as USD/Yen, Euro/Yen, or USD/Euro. Long and short positions in the same currency pair are permitted to perfectly offset, but no offset may be recognised across currency pairs.  
- Credit derivatives and equity derivatives: A single hedging set is employed for each asset class. Full offset is recognised for derivatives referencing the same entity (name or index), while partial offset is recognised between derivatives referencing different entities.  
- Commodity derivatives: Four hedging sets are employed for different classes of commodities (one for each of energy, metals, agricultural, and other commodities). Within the same hedging set, full offset is recognised between derivatives referencing the same commodity and partial offset is recognised between derivatives referencing different commodities. No offset is recognised between different hedging sets.

With respect to each asset class, basis transactions and volatility transactions form separate hedging sets in their respective asset classes as described in paragraphs 162 and 163 of the accompanying standards text. These separate hedging sets will be assigned specific supervisory factors as described in those paragraphs and will follow the main hedging set aggregation rules for its relevant asset class.

A basis transaction is a non-foreign-exchange transaction (ie both legs are denominated in the same currency) in which the cash flows of both legs depend on different risk factors from the same asset class. Common examples of basis transactions include interest rate basis swaps (where payments based on two distinct floating interest rates are exchanged) and commodity spread trades (where payments based on prices of two related commodities are exchanged). All basis transactions of a netting set that belong to the same asset class and reference the same pair of risk factors form a single hedging set. For example, all three-month Libor versus six-month Libor swaps in a netting set form a single basis hedging set.

A volatility transaction is one in which the reference asset depends on the volatility (historical or implied) of a risk factor. Common examples of volatility transactions include variance and volatility swaps and options on volatility indices. Volatility transactions form hedging sets according to the rules of their respective asset classes. For example, all equity volatility transactions form a single volatility hedging set.

# C. Scope of application

The SA-CCR will apply to OTC derivatives, exchange-traded derivatives and long settlement transactions.²

# D. Transitional arrangements

The Basel Committee recognises that the SA-CCR introduces a significant change in methodology from the current non-internal model method approaches. Jurisdictions may need time to implement these changes in their respective capital frameworks. In addition, smaller banks may need time to develop operational capabilities in order to employ the SA-CCR. As a result, the SA-CCR will become effective on 1 January 2017.

# E. Examples

Annex 4a sets forth examples of application of the SA-CCR to sample portfolios. Annex 4b sets forth examples of the operation of the SA-CCR in the context of standard margin agreements. Annex 4c sets forth a flow chart of steps for calculating interest-rate add-ons.

# II. Revisions to Part 2: The First Pillar; Section II: Credit risk – the standardised approach

Section D. The standardised approach - credit risk mitigation

Paragraph 84 will be amended by adding the following sentence at the end of the paragraph:

"This paragraph does not apply to posted collateral that is treated under either the SA-CCR (Annex 4, section X) or IMM (Annex 4, section V) calculation methods in the counterparty credit risk framework."

Paragraphs 186, 187 and 187(i) will be deleted in their entirety and replaced with the following:

186. Under the SA-CCR, the calculation of exposure amount will be as follows:

$$
E x p o s u r e a m o u n t = \text {a l p h a} ^ {*} (R C + P F E)
$$

where:

$$
a l p h a \quad = 1. 4,
$$

$RC$  = the replacement cost calculated according to paragraphs 130-145 of Annex 4, and

$PFE =$  the amount for potential future exposure calculated according to paragraphs 146-187 of Annex 4.

187. (deleted)  
187(i). (deleted)

# III. Revisions to Part 2: The First Pillar; Annex 4 Treatment of Counterparty Credit Risk and Cross-Product Netting

# A. Section V – Internal Model Method, Section VI – Standardised Method, and Section VII – Current Exposure Method

To implement the changes to the counterparty credit risk framework accompanying the introduction of the SA-CCR (including the removal of the IMM shortcut method), paragraph  $41^{3}$  of section V will be deleted $^{4}$  as well as sections VI and VII. Sections VI and VII will be replaced in their entirety as follows:

# Section X. Standardised Approach for counterparty credit risk

128. Banks that do not have approval to apply the Internal Model Method (IMM) for the relevant OTC transactions must use the Standardised Approach for counterparty credit risk (SA-CCR). The SA-CCR can be used only for OTC derivatives, exchange-traded derivatives and long settlement transactions; SFTs are subject to the treatments set out under the IMM of this Annex or under Part 2, Section II.D, of this Framework. EAD is to be calculated separately for each netting set. It is determined as follows:

$$
E A D = a l p h a ^ {*} (R C + P F E)
$$

where:

$\begin{array}{rl}{alpha}\\{}&{=1.4,}\end{array}$

$RC =$  the replacement cost calculated according to paragraphs 130-145 of this Annex, and

$PFE =$  the amount for potential future exposure calculated according to paragraphs 146-187 of this Annex.

129. The replacement cost (RC) and the potential future exposure (PFE) components are calculated differently for margined and unmargined netting sets. The EAD for a margined netting set is capped at the EAD of the same netting set calculated on an unmargined basis.

# RC and NICA

130. For unmargined transactions, the RC intends to capture the loss that would occur if a counterparty were to default and were closed out of its transactions immediately. The PFE add-on represents a potential conservative increase in exposure over a one-year time horizon from the present date (ie the calculation date).

131. For margined trades, the RC intends to capture the loss that would occur if a counterparty were to default at the present or at a future time, assuming that the closeout and replacement of transactions occur instantaneously. However, there may be a period (the margin period of risk) between the last exchange of collateral before default and replacement of the trades in the market. The PFE add-on represents the potential change in value of the trades during this time period.

In both cases, the haircut applicable to noncash collateral in the replacement cost formulation represents the potential change in value of the collateral during the appropriate time period (one year for unmarked trades and the margin period of risk for margined trades).

133. Replacement cost is calculated at the netting set level, whereas PFE add-ons are calculated for each asset class within a given netting set and then aggregated (see paragraphs 150-187 below).

134. For capital adequacy purposes, banks may net transactions (eg when determining the RC component of a netting set) subject to novation under which any obligation between a bank and its counterparty to deliver a given currency on a given value date is automatically amalgamated with all other obligations for the same currency and value date, legally substituting one single amount for the previous gross obligations. Banks may also net transactions subject to any legally valid form of bilateral netting not covered in the preceding sentence, including other forms of novation. In every such case where netting is applied, a bank must satisfy its national supervisor that it has:

(i) A netting contract with the counterparty or other agreement which creates a single legal obligation, covering all included transactions, such that the bank would have either a claim to receive or obligation to pay only the net sum of the positive and negative mark-to-market values of included individual transactions in the event a counterparty fails to perform due to any of the following: default, bankruptcy, liquidation or similar circumstances;5

(ii) Written and reasoned legal reviews that, in the event of a legal challenge, the relevant courts and administrative authorities would find the bank's exposure to be such a net amount under:

The law of the jurisdiction in which the counterparty is chartered and, if the foreign branch of a counterparty is involved, then also under the law of the jurisdiction in which the branch is located;

The law that governs the individual transactions; and

The law that governs any contract or agreement necessary to effect the netting.

The national supervisor, after consultation when necessary with other relevant supervisors, must be satisfied that the netting is enforceable under the laws of each of the relevant jurisdictions.<sup>6</sup>

(iii) Procedures in place to ensure that the legal characteristics of netting arrangements are kept under review in light of the possible changes in relevant law.

135. There are two formulations of replacement cost depending on whether the trades with a counterparty are subject to a margin agreement. Where a margin agreement exists, the formulation could apply both to bilateral transactions and central clearing relationships. The formulation also addresses the various arrangements that a bank may have to post and/or receive collateral that may be referred to as initial margin.

# Formulation for unmarked transactions

136. For unmarked transactions (that is, where variation margin (VM) is not exchanged, but collateral other than VM may be present),  $RC$  is defined as the greater of: (i) the current market value of the derivative contracts less net haircut collateral held by the bank (if any), and (ii) zero. This is consistent

with the use of replacement cost as the measure of current exposure, meaning that when the bank owes the counterparty money it has no exposure to the counterparty if it can instantly replace its trades and sell collateral at current market prices. Mathematically:

$$
R C = \max  \{V - C; 0 \}
$$

where  $V$  is the value of the derivative transactions in the netting set and  $C$  is the haircut value of net collateral held, which is calculated in accordance with the NICA methodology defined in paragraph 143 below. For this purpose, the value of non-cash collateral posted by the bank to its counterparty is increased and the value of the non-cash collateral received by the bank from its counterparty is decreased using haircuts (which are the same as those that apply to repo-style transactions) for the time periods described in paragraph 132 above.

137. In the above formulation, it is assumed that the replacement cost representing today's exposure to the counterparty cannot go less than zero. However, banks sometimes hold excess collateral (even in the absence of a margin agreement) or have out-of-the-money trades which can further protect the bank from the increase of the exposure. As discussed in paragraphs 147-149 below, the SA-CCR would allow such over-collateralisation and negative mark-to market value to reduce PFE, but would not affect replacement cost.  
138. Bilateral transactions with a one-way margining agreement in favour of the bank's counterparty (that is, where a bank posts, but does not collect, collateral) must be treated as unimargined transactions.

# Formulation for margined transactions

139. The RC formula for margined transactions builds on the RC formula for unmarked transactions. It also employs concepts used in standard margining agreements, as discussed more fully below.  
140. The RC for margined transactions in the SA-CCR is defined as the greatest exposure that would not trigger a call for VM, taking into account the mechanics of collateral exchanges in marginine agreements. Such mechanics include, for example, "Threshold", "Minimum Transfer Amount" and "Independent Amount" in the standard industry documentation, which are factored into a call for VM. A defined, generic formulation has been created to reflect the variety of marginine approaches used and those being considered by supervisors internationally.

# Incorporating N/CA into replacement cost

141. One objective of the SA-CCR is to more fully reflect the effect of margining agreements and the associated exchange of collateral in the calculation of CCR exposures. The following paragraphs address how the exchange of collateral is incorporated into the SA-CCR.

See Annex 4b for illustrative examples of the effect of standard margin agreements on the SA-CCR formulation.  
For example, the 1992 (Multicurrency-Cross Border) Master Agreement and the 2002 Master Agreement published by the International Swaps & Derivatives Association, Inc. (ISDA Master Agreement). The ISDA Master Agreement includes the ISDA CSA: the 1994 Credit Support Annex (Security Interest – New York Law), or, as applicable, the 1995 Credit Support Annex (Transfer – English Law) and the 1995 Credit Support Deed (Security Interest – English Law).  
For example, in the ISDA Master Agreement, the term "Credit Support Amount", or the overall amount of collateral that must be delivered between the parties, is defined as the greater of the Secured Party's Exposure plus the aggregate of all Independent Amounts applicable to the Pledgor minus all Independent Amounts applicable to the Secured Party, minus the Pledgor's Threshold and zero.

142. To avoid confusion surrounding the use of terms initial margin and independent amount which are used in various contexts and sometimes interchangeably, the term independent collateral amount (ICA) is introduced. ICA represents (i) collateral (other than VM) posted by the counterparty that the bank may seize upon default of the counterparty, the amount of which does not change in response to the value of the transactions it secures and/or (ii) the Independent Amount (IA) parameter as defined in standard industry documentation. ICA can change in response to factors such as the value of the collateral or a change in the number of transactions in the netting set.

143. Because both a bank and its counterparty may be required to post ICA, it is necessary to introduce a companion term, net independent collateral amount (NICA), to describe the amount of collateral that a bank may use to offset its exposure on the default of the counterparty. NICA does not include collateral that a bank has posted to a segregated, bankruptcy remote account, which presumably would be returned upon the bankruptcy of the counterparty. That is, NICA represents any collateral (segregated or unsegregated) posted by the counterparty less the unsegregated collateral posted by the bank. With respect to IA, NICA takes into account the differential of IA required for the bank minus IA required for the counterparty.

144. For margined trades, the replacement cost is:

$$
R C = \max  \left\{V - C; T H + M T A - N / C A; 0 \right\}
$$

where  $V$  and  $C$  are defined as in the unMargined formulation,  $TH$  is the positive threshold before the counterparty must send the bank collateral, and  $MTA$  is the minimum transfer amount applicable to the counterparty.

145.  $TH + MTA - NICA$  represents the largest exposure that would not trigger a VM call and it contains levels of collateral that need always to be maintained. For example, without initial margin or IA, the greatest exposure that would not trigger a variation margin call is the threshold plus any minimum transfer amount. In the adapted formulation, NICA is subtracted from  $TH + MTA$ . This makes the calculation more accurate by fully reflecting both the actual level of exposure that would not trigger a margin call and the effect of collateral held and/or posted by a bank. The calculation is floored at zero, recognising that the bank may hold NICA in excess of  $TH + MTA$ , which could otherwise result in a negative replacement cost.

# PFE add-ons

146. The PFE add-on consists of (i) an aggregate add-on component, which consists of add-ons calculated for each asset class and (ii) a multiplier that allows for the recognition of excess collateral or negative mark-to-market value for the transactions. Mathematically:

$$
P F E = m u l t i p l i e r ^ {*} A d d O n ^ {a g g r e g a t e}
$$

where AddOn<sup>aggregate</sup> is the aggregate add-on component and multiplier is defined as a function of three inputs: V, C and AddOn<sup>aggregate</sup>.

The paragraphs below describe the inputs that enter into the calculation of the add-on formulas in more detail, and set out the formula for each asset class.

Recognition of excess collateral and negative mark-to-market

147. As a general principle, over-collateralisation should reduce capital requirements for counterparty credit risk. In fact, many banks hold excess collateral (ie collateral greater than the net market value of the derivatives contracts) precisely to offset potential increases in exposure represented by the add-on. As discussed in paragraphs 136 and 144, collateral may reduce the replacement cost component of the exposure under the SA-CCR. The PFE component also reflects the risk-reducing property of excess collateral.

148. For prudential reasons, the Basel Committee decided to apply a multiplier to the PFE component that decreases as excess collateral increases, without reaching zero (the multiplier is floored at  $5\%$  of the PFE add-on). When the collateral held is less than the net market value of the derivative contracts ("under-collateralisation"), the current replacement cost is positive and the multiplier is equal to one (ie the PFE component is equal to the full value of the aggregate add-on). Where the collateral held is greater than the net market value of the derivative contracts ("over-collateralisation"), the current replacement cost is zero and the multiplier is less than one (ie the PFE component is less than the full value of the aggregate add-on).  
149. This multiplier will also be activated when the current value of the derivative transactions is negative. This is because out-of-the-money transactions do not currently represent an exposure and have less chance to go in-the-money. Mathematically:

$$
m u l t i p l i e r = \min  \left\{1; F l o o r + (1 - F l o o r) ^ {*} \exp \left(\frac {V - C}{2 ^ {*} (1 - F l o o r) ^ {*} A d d O n ^ {a g g r e g a t e}}\right) \right\}
$$

where  $\exp (\ldots)$  equals to the exponential function,  $Floor$  is  $5\%$ ,  $V$  is the value of the derivative transactions in the netting set, and  $C$  is the haircut value of net collateral held.

# Aggregation across asset classes

150. Diversification benefits across asset classes are not recognised. Instead, the respective add-ons for each asset class are simply aggregated. Mathematically:

$$
A d d O n ^ {a g g r e g a t e} = \sum_ {a} A d d O n ^ {(a)}
$$

where the sum of each asset class add-on is taken.

Allocation of derivative transactions to one or more asset classes

151. The designation of a derivative transaction to an asset class is be made on the basis of its primary risk driver. Most derivative transactions have one primary risk driver, defined by its reference underlying instrument (eg an interest rate curve for an interest rate swap, a reference entity for a credit default swap, a foreign exchange rate for a FX call option, etc). When this primary risk driver is clearly identifiable, the transaction will fall into one of the asset classes described above.  
152. For more complex trades that may have more than one risk driver (eg multi-asset or hybrid derivatives), banks must take sensitivities and volatility of the underlying into account for determining the primary risk driver.

Bank supervisors may also require more complex trades to be allocated to more than one asset class, resulting in the same position being included in multiple classes. In this case, for each asset class to which the position is allocated, banks must determine appropriately the sign and delta adjustment of the relevant risk driver.

# General steps for calculating the add-on

153. For each transaction, the primary risk factor or factors need to be determined and attributed to one or more of the five asset classes: interest rate, foreign exchange, credit, equity or commodity. The add-on for each asset class is calculated using asset-class-specific formulas that represent a stylised Effective EPE calculation under the assumption that all trades in the asset class have zero current mark-to-market value (ie they are at-the-money).  
154. Although the add-on formulas are asset class-specific, they have a number of features in common. To determine the add-on, transactions in each asset class are subject to adjustment in the following general steps:

An adjusted notional amount based on actual notional or price is calculated at the trade level. For interest rate and credit derivatives, this adjusted notional amount also incorporates a supervisory measure of duration;  
- A maturity factor  $M F_{i}^{(type)}$  reflecting the time horizon appropriate for the type of transaction is calculated at the trade level (see paragraph 164 below for details) and is applied to the adjusted notional. Two types of maturity factor are defined, one for margined transactions ( $M F_{i}^{(m\arg\text{ined})}$ )

and one for unmarked transactions  $(MF_{i}^{(\text{unmarg ined})})$ ;

- A supervisory delta adjustment is made to this trade-level adjusted notional amount based on the position (long or short) and whether the trade is an option, CDO tranche or neither, resulting in an effective notional amount;  
A supervisory factor is applied to each effective notional amount to reflect volatility; and  
The trades within each asset class are separated into hedging sets and an aggregation method is applied to aggregate all the trade-level inputs at the hedging set level and finally at the asset-class level. For credit, equity and commodity derivatives, this involves the application of a supervisory correlation parameter to capture important basis risks and diversification.

Each input is described, generally and by asset class, in more detail below.

Period or date parameters:  $M_{i}, E_{i}, S_{i}$  and  $T_{i}$

155. There are four dates that appear in the SA-CCR:

- For all asset classes, the maturity  $M_{i}$  of a contract is the latest date when the contract may still be active. This date appears in the maturity factor defined in paragraph 164 that scales down adjusted notional for unmarked trades for all asset classes. If a derivative contract has another derivative contract as its underlying (for example, a swaption) and may be physically exercised into the underlying contract (ie a bank would assume a position in the underlying contract in the event of exercise), then maturity of the contract is the final settlement date of the underlying derivative contract.  
- For interest rate and credit derivatives, the start date  $S_{i}$  of the time period referenced by an interest rate or credit contract. If the derivative references the value of another interest rate or credit instrument (eg swaption or bond option), the time period must be determined on the basis of the underlying instrument. This date appears in the definition of supervisory duration defined in paragraph 157.  
- For interest rate and credit derivatives, the end date  $E_{i}$  of the time period referenced by an interest rate or credit contract. If the derivative references the value of another interest rate or credit instrument (eg swaption or bond option), the time period must be determined on the basis of the underlying instrument. This date appears in the definition of supervisory duration defined in paragraph 157. In addition, this date specifies the maturity category for an interest rate contract in paragraph 166.  
- For options in all asset classes, the latest contractual exercise date  $T_{i}$  as referenced by the contract. This period shall be used for the determination of the option delta in paragraph 159.

156. Table 1 includes example transactions and provides each transaction's related maturity  $M_{i}$ , start date  $S_{i}$  and end date  $E_{i}$ . In addition, the option delta in paragraph 159 depends on the latest contractual exercise date  $T_{i}$  (not separately shown in the table).

Table 1  

<table><tr><td>Instrument</td><td>Mi</td><td>Si</td><td>Ei</td></tr><tr><td>Interest rate or credit default swap maturing in 10 years</td><td>10 years</td><td>0</td><td>10 years</td></tr><tr><td>10-year interest rate swap, forward starting in 5 years</td><td>15 years</td><td>5 years</td><td>15 years</td></tr><tr><td>Forward rate agreement for time period starting in 6 months and ending in 12 months</td><td>1 year</td><td>0.5 year</td><td>1 year</td></tr><tr><td>Cash-settled European swaption referencing 5-year interest rate swap with exercise date in 6 months</td><td>0.5 year</td><td>0.5 year</td><td>5.5 years</td></tr><tr><td>Physically-settled European swaption referencing 5-year interest rate swap with exercise date in 6 months</td><td>5.5 years</td><td>0.5 year</td><td>5.5 years</td></tr><tr><td>10-year Bermudan swaption with annual exercise dates</td><td>10 years</td><td>1 year</td><td>10 years</td></tr><tr><td>Interest rate cap or floor specified for semi-annual interest rate with maturity 5 years</td><td>5 years</td><td>0</td><td>5 years</td></tr><tr><td>Option on a bond maturing in 5 years with the latest exercise date in 1 year</td><td>1 year</td><td>1 year</td><td>5 years</td></tr><tr><td>3-month Eurodollar futures that matures in 1 year</td><td>1 year</td><td>1 year</td><td>1.25 years</td></tr><tr><td>Futures on 20-year treasury bond that matures in 2 years</td><td>2 years</td><td>2 years</td><td>22 years</td></tr><tr><td>6-month option on 2-year futures on 20-year treasury bond</td><td>2 years</td><td>2 years</td><td>22 years</td></tr></table>

Trade-level adjusted notional (for trade i of asset class a):  $d_{i}^{(a)}$

157. These parameters are defined at the trade level and take into account both the size of a position and its maturity dependency, if any. Specifically, the adjusted notional amounts are calculated as follows:

- For interest rate and credit derivatives, the trade-level adjusted notional is the product of the trade notional amount, converted to the domestic currency, and the supervisory duration  $SD_{i}$  which is given by the following formula:

$$
S D _ {i} = \frac {\exp (- 0 . 0 5 * S _ {i}) - \exp (- 0 . 0 5 * E _ {i})}{0 . 0 5}
$$

where  $S_{i}$  and  $E_{i}$  are the start and end dates, respectively, of the time period referenced by the interest rate or credit derivative (or, where such a derivative references the value of another interest rate or credit instrument, the time period determined on the basis of the underlying instrument), floored by ten business days. If the start date has occurred (eg an ongoing interest rate swap),  $S_{i}$  must be set to zero.

- For foreign exchange derivatives, the adjusted notional is defined as the notional of the foreign currency leg of the contract, converted to the domestic currency. If both legs of a foreign

10

Note there is a distinction between the time period of the underlying transaction and the remaining maturity of the derivative contract. For example, a European interest rate swaption with expiry of 1 year and the term of the underlying swap of 5 years has  $S_{i} = 1$  year and  $E_{i} = 6$  years.

exchange derivative are denominated in currencies other than the domestic currency, the notional amount of each leg is converted to the domestic currency and the leg with the larger domestic currency value is the adjusted notional amount.

- For equity and commodity derivatives, the adjusted notional is defined as the product of the current price of one unit of the stock or commodity (eg a share of equity or barrel of oil) and the number of units referenced by the trade.

158. In many cases the trade notional amount is stated clearly and fixed until maturity. When this is not the case, banks must use the following rules to determine the trade notional amount.

- For transactions with multiple payoffs that are state contingent such as digital options or target redemption forwards, a bank must calculate the trade notional amount for each state and use the largest resulting calculation.  
- Where the notional is a formula of market values, the bank must enter the current market values to determine the trade notional amount.  
- For variable notional swaps such as amortising and accreting swaps, banks must use the average notional over the remaining life of the swap as the trade notional amount.  
- Leveraged swaps must be converted to the notional of the equivalent unleveraged swap, that is, where all rates in a swap are multiplied by a factor, the stated notional must be multiplied by the factor on the interest rates to determine the trade notional amount.  
- For a derivative contract with multiple exchanges of principal, the notional is multiplied by the number of exchanges of principal in the derivative contract to determine the trade notional amount.  
- For a derivative contract that is structured such that on specified dates any outstanding exposure is settled and the terms are reset so that the fair value of the contract is zero, the remaining maturity equals the time until the next reset date.

# Supervisory delta adjustments:  $\delta_{i}$

159. These parameters are also defined at the trade level and are applied to the adjusted notional amounts to reflect the direction of the transaction and its non-linearity. More specifically, the delta adjustments for all derivatives are defined as follows:

<table><tr><td>δi</td><td>Long11 in the primary risk factor</td><td>Short12 in the primary risk factor</td></tr><tr><td>Instruments that are not options or CDO tranches</td><td>+1</td><td>-1</td></tr><tr><td>δi</td><td>Bought</td><td>Sold</td></tr><tr><td>Call Options13</td><td>+Φ( ln(Pi / Ki) + 0.5*σi2*Ti / σi*√Ti)</td><td>-Φ( ln(Pi / Ki) + 0.5*σi2*Ti / σi*√Ti)</td></tr><tr><td>Put Options7</td><td>-Φ(-ln(Pi / Ki) + 0.5*σi2*Ti / σi*√Ti)</td><td>+Φ(-ln(Pi / Ki) + 0.5*σi2*Ti / σi*√Ti)</td></tr><tr><td colspan="3">With the following parameters that banks must determine appropriately: 
Pi: Underlying price (spot, forward, average, etc) 
Ki: Strike price 
Ti: Latest contractual exercise date of the option 
The supervisory volatility Σi of an option is specified on the basis of supervisory factor applicable to the trade (see Table 2 in paragraph 183).</td></tr></table>

<table><tr><td>δi</td><td>Purchased (long protection)</td><td>Sold (short protection)</td></tr><tr><td>CDO tranches</td><td>15/(1+14*A_i)*(1+14*D_i)</td><td>15/(1+14*A_i)*(1+14*D_i)</td></tr><tr><td colspan="3">With the following parameters that banks must determine appropriately: 
Ai: Attachment point of the CDO tranche 
Di: Detachment point of the CDO tranche</td></tr></table>

# Supervisory factors:  $SF_{i}^{(a)}$

160. A factor or factors specific to each asset class is used to convert the effective notional amount into Effective EPE based on the measured volatility of the asset class. Each factor has been calibrated to reflect the Effective EPE of a single at-the-money linear trade of unit notional and one-year maturity. This includes the estimate of realised volatilities assumed by supervisors for each underlying asset class.

# Hedging sets

161. The hedging sets in the different asset classes are defined as follows, except for those described in paragraphs 162 and 163.

- Interest rate derivatives consist of a separate hedging set for each currency;  
FX derivatives consist of a separate hedging set for each currency pair;  
Credit derivatives consist of a single hedging set;  
Equity derivatives consist of a single hedging set;  
Commodity derivatives consist of four hedging sets defined for broad categories of commodity derivatives: energy, metals, agricultural and other commodities.

162. Derivatives that reference the basis between two risk factors and are denominated in a single currency $^{14}$  (basis transactions) must be treated within separate hedging sets within the corresponding asset class. There is a separate hedging set $^{15}$  for each pair of risk factors (ie for each specific basis). Examples of specific bases include three-month Libor versus six-month Libor, three-month Libor versus three-month T-Bill, one-month Libor versus OIS rate, Brent Crude oil versus Henry Hub gas. For hedging sets consisting of basis transactions, the supervisory factor applicable to a given asset class must be multiplied by one-half.  
163. Derivatives that reference the volatility of a risk factor (volatility transactions) must be treated within separate hedging sets within the corresponding asset class. Volatility hedging sets must follow the same hedging set construction outlined in paragraph 161 (for example, all equity volatility transactions form a single hedging set). Examples of volatility transactions include variance and volatility swaps, options on realised or implied volatility. For hedging sets consisting of volatility transactions, the supervisory factor applicable to a given asset class must be multiplied by a factor of five.

# Time Risk Horizons

164. The minimum time risk horizons for the SA-CCR include:

The lesser of one year and remaining maturity of the derivative contract for unimargined transactions, floored at ten business days.16 Therefore, the adjusted notional at the trade level of an unimargined transaction must be mutilplied by:

$$
M F _ {i} ^ {\left(\text {u n m a r g i n e d}\right)} = \sqrt {\frac {\operatorname* {m i n} \left\{M _ {i} ; 1 \text {y e a r} \right\}}{1 \text {y e a r}}}
$$

where  $M_{i}$  is the transaction  $i$  remaining maturity floored by 10 business days.

For margined transactions, the minimum margin period of risk is determined as follows:  
- At least ten business days for non-centrally-cleared derivative transactions subject to daily margin agreements.  
- Five business days for centrally cleared derivative transactions subject to daily margin agreements that clearing members have with their clients.  
- 20 business days for netting sets consisting of 5,000 transactions that are not with a central counterparty.  
- Doubling the margin period of risk for netting sets with outstanding disputes consistent with paragraph 41(ii) of this Annex. $^{17}$

Therefore, the adjusted notional at the trade level of a margined transaction should be multiplied by:

14 Derivatives with two floating legs that are denominated in different currencies (such as cross-currency swaps) are not subject to this treatment; rather, they should be treated as non-basis foreign exchange contracts.  
Within this hedging set, long and short positions are determined with respect to the basis.  
For example, remaining maturity for a one-month option on a 10-year Treasury bond is the one-month to expiration date of the derivative contract. However, the end date of the transaction is the 10-year remaining maturity on the Treasury bond.  
See paragraphs 41(i), 41(ii) and 111, which were introduced via Basel III and the capital requirements for bank exposures to central counterparties, for circumstances requiring an extended margin period of risk.

$$
M F _ {i} ^ {(m a r g i n e d)} = \frac {3}{2} \sqrt {\frac {M P O R _ {i}}{1 y e a r}}
$$

where  $MPOR_{i}$  is the margin period of risk appropriate for the margin agreement containing the transaction  $i$ .

Supervisory correlation parameters:  $\rho_{i}^{(a)}$

165. These parameters only apply to the PFE add-on calculation for equity, credit and commodity derivatives. For these asset classes, the supervisory correlation parameters are derived from a single-factor model and specify the weight between systematic and idiosyncratic components. This weight determines the degree of offset between individual trades, recognising that imperfect hedges provide some, but not perfect, offset. Supervisory correlation parameters do not apply to interest rate and foreign exchange derivatives.

# Add-on for interest rate derivatives

166. The add-on for interest rate derivatives captures the risk of interest rate derivatives of different maturities being imperfectly correlated. To address this risk, the SA-CCR divides interest rate derivatives into maturity categories (also referred to as "buckets") based on the end date (as described in paragraphs 155 and 157) of the transactions. The three relevant maturity categories are: less than one year, between one and five years and more than five years. The SA-CCR allows full recognition of offsetting positions within a maturity category. Across maturity categories, the SA-CCR recognises partial offset.  
167. The add-on for interest rate derivatives is the sum of the add-ons for each hedging set of interest rates derivatives transacted with a counterparty in a netting set. The add-on for a hedging set of interest rate derivatives is calculated in two steps.  
168. In the first step, the effective notional  $D_{jk}^{(IR)}$  is calculated for time bucket  $k$  of hedging set (ie currency)  $j$  according to:

$$
D _ {j k} ^ {(I R)} = \sum_ {i \in \{C c y _ {j}, M B _ {k} \}} \delta_ {i} ^ {\star} d _ {i} ^ {(I R)} ^ {\star} M F _ {i} ^ {(t y p e)}
$$

where notation  $i \in \{Ccy_j, MB_k\}$  refers to trades of currency  $j$  that belong to maturity bucket  $k$ . That is, the effective notional for each time bucket and currency is the sum of the trade-level adjusted notional amounts (cf. paragraphs 157-158) multiplied by the supervisory delta adjustments (cf. paragraph 159) and the maturity factor (cf. paragraph 164).

169. In the second step, aggregation across maturity buckets for each hedging set is calculated according to the following formula: $^{18}$

$$
E f f e c t i v e N o t i o n a l _ {j} ^ {(I R)} = \left[ \left(D _ {j 1} ^ {(I R)}\right) ^ {2} + \left(D _ {j 2} ^ {(I R)}\right) ^ {2} + \left(D _ {j 3} ^ {(I R)}\right) ^ {2} + 1. 4 * D _ {j 1} ^ {(I R)} * D _ {j 2} ^ {(I R)} + 1. 4 * D _ {j 2} ^ {(I R)} * D _ {j 3} ^ {(I R)} + 0. 6 * D _ {j 1} ^ {(I R)} * D _ {j 3} ^ {(I R)} \right] ^ {\frac {1}{2}}
$$

The hedging set level add-on is calculated as the product of the effective notional and the interest rate supervisory factor:

$$
A d d O n _ {j} ^ {(I R)} = S F _ {j} ^ {(I R)} * E f f e c t i v e N o t i o n a l _ {j} ^ {(I R)}
$$

Aggregation across hedging sets is performed via simple summation:

$$
A d d O n ^ {(I R)} = \sum_ {j} A d d O n _ {j} ^ {(I R)}
$$

Add-on for foreign exchange derivatives

170. The add-on formula for foreign exchange derivatives shares many similarities with the add-on formula for interest rates. Similar to interest rate derivatives, the effective notional of a hedging set is defined as the sum of all the trade-level adjusted notional amounts multiplied by their supervisory delta. The add-on for a hedging set is the product of:

The absolute value of its effective notional amount; and  
The supervisory factor (same for all FX hedging sets).

171. In the case of foreign exchange derivatives, the adjusted notional amount is maturity-independent and given by the notional of the foreign currency leg of the contract, converted to the domestic currency. Mathematically:

$$
A d d O n ^ {(F X)} = \sum_ {j} A d d O n _ {H S _ {j}} ^ {(F X)}
$$

where the sum is taken over all the hedging sets  $HS_{j}$  included in the netting set. The add-on and the effective notional of the hedging set  $HS_{j}$  are respectively given by:

$$
A d d O n _ {H S _ {j}} ^ {(F X)} = S F _ {j} ^ {(F X)} * \left| E f f e c t i v e N o t i o n a l _ {j} ^ {(F X)} \right|
$$

$$
\text {E f f e c t i v e N o t i o n a l} _ {j} ^ {(F X)} = \sum_ {i \in H S _ {j}} \delta_ {i} * d _ {i} ^ {(F X)} * M F _ {i} ^ {(t y p e)}
$$

where  $i \in HS_{j}$  refers to trades of hedging set  $HS_{j}$ . That is, the effective notional for each currency pair is the sum of the trade-level adjusted notional amounts (cf. paragraphs 157-158) multiplied by the supervisory delta adjustments (cf. paragraph 159) and the maturity factor (cf. paragraph 164).

Add-on for credit derivatives

172. There are two levels of offsetting benefits for credit derivatives. First, all credit derivatives referencing the same entity (either a single entity or an index) are allowed to offset each other fully to form an entity-level effective notional amount:

$$
E f f e c t i v e N o t i o n a l _ {k} ^ {(C r e d i t)} = \sum_ {i \in E n t i t y _ {k}} \delta_ {i} * d _ {i} ^ {(C r e d i t)} * M F _ {i} ^ {(t y p e)}
$$

where  $i \in \text{Entity}_k$  refers to trades of entity  $k$ . That is, the effective notional for each entity is the sum of the trade-level adjusted notional amounts (cf. paragraphs 157-158) multiplied by the supervisory delta adjustments (cf. paragraph 159) and the maturity factor (cf. paragraph 164).

The add-on for all the positions referencing this entity is defined as the product of its effective notional amount and the supervisory factor  $SF_{k}^{(\text{Credit})}$ , ie:

$$
A d d O r \left(E n t i t y _ {k}\right) = S F _ {k} ^ {\left(C r e d i t\right)} * E f f e c t i v e N o t i o n a l _ {k} ^ {\left(C r e d i t\right)}
$$

For single name entities,  $SF_{k}^{(\text{Credit})}$  is determined by the reference name's credit rating. For index entities,  $SF_{k}^{(\text{Credit})}$  is determined by whether the index is investment grade or speculative grade.

Second, all the entity-level add-ons are grouped within a single hedging set (except for basis and volatility transactions) in which full offsetting between two different entity-level add-ons is not permitted. Instead, a single-factor model has been used to allow partial offsetting between the entity-level add-ons by dividing the risk of the credit derivatives asset class into a systematic component and an idiosyncratic component.

173. The entity-level add-ons are allowed to offset each other fully in the systematic component; whereas, there is no offsetting benefit in the idiosyncratic component. These two components are weighted by a correlation factor which determines the degree of offsetting/hedging benefit within the credit derivatives asset class. The higher the correlation factor, the higher the importance of the systemic component, hence the higher the degree of offsetting benefits. Derivatives referencing credit indices are treated as though they were referencing single names, but with a higher correlation factor applied. Mathematically:

$$
A d d O n ^ {(C r e d i t)} = \left[ \left(\sum_ {k} \rho_ {k} ^ {(C r e d i t)} * A d d O n (E n t i t y _ {k})\right) ^ {2} + \sum_ {k} \left(1 - \left(\rho_ {k} ^ {(C r e d i t)}\right) ^ {2}\right) * \left(A d d O n (E n t i t y _ {k})\right) ^ {2} \right] ^ {\frac {1}{2}}
$$

where  $\rho_{k}^{(Credit)}$  is the appropriate correlation factor corresponding to the Entity  $k$ .

It should be noted that a higher or lower correlation does not necessarily mean a higher or lower capital charge. For portfolios consisting of long and short credit positions, a high correlation factor would reduce the charge. For portfolios consisting exclusively of long positions (or short positions), a higher correlation factor would increase the charge. If most of the risk consists of systematic risk, then individual reference entities would be highly correlated and long and short positions should offset each other. If, however, most of the risk is idiosyncratic to a reference entity, then individual long and short positions would not be effective hedges for each other.  
175. The use of a single hedging set for credit derivatives implies that credit derivatives from different industries and regions are equally able to offset the systematic component of an exposure, although they would not be able to offset the idiosyncratic portion. This approach recognises that meaningful distinctions between industries and/or regions are complex and difficult to analyse for global conglomerates.

# Add-on for equity derivatives

176. The add-on formula for equity derivatives shares many similarities with the add-on formula for credit derivatives. The approach also uses a single factor model to divide the risk into a systematic component and an idiosyncratic component for each reference entity (a single entity or an index). Derivatives referencing equity indices are treated as though they were referencing single entities, but with a higher correlation factor used for the systematic component. Offsetting is allowed only for the systematic components of the entity-level add-ons, while full offsetting of transactions within the same reference entity is permitted. The entity-level add-ons are proportional to the product of two items: the effective notional amount of the entity (similar to credit derivatives) and the supervisory factor appropriate to the entity.

177. The calibration of the supervisory factors for equity derivatives relies on estimates of the market volatility of equity indices, with the application of a conservative beta factor<sup>19</sup> to translate this estimate into an estimate of individual volatilities. Banks are not permitted to make any modelling assumptions in the calculation of the PFE add-ons, including estimating individual volatilities or taking publicly available estimates of beta. This is a pragmatic approach to ensure a consistent implementation across jurisdictions but also to keep the add-on calculation relatively simple and prudent. Therefore, only two values of supervisory factors have been defined for equity derivatives, one for single entities and one for indices.

In summary, the formula is as follows:

$$
A d d O n ^ {(E q u i t y)} = \left[ \left(\sum_ {k} \rho_ {k} ^ {(E q u i t y)} * A d d O n (E n t i t y _ {k})\right) ^ {2} + \sum_ {k} \left(1 - \left(\rho_ {k} ^ {(E q u i t y)}\right) ^ {2}\right) * \left(A d d O n (E n t i t y _ {k})\right) ^ {2} \right] ^ {\frac {1}{2}}
$$

where  $\rho_{k}^{(\text{Equity})}$  is the appropriate correlation factor corresponding to the entity  $k$ . The add-on for all the positions referencing entity  $k$  and its effective notional are given by:

$$
A d d O n \left(E n t i t y _ {k}\right) = S F _ {k} ^ {(E q u i t y)} * E f f e c t i v e N o t i o n a l _ {k} ^ {(E q u i t y)}
$$

and

$$
E f f e c t i v e N o t i o n a l _ {k} ^ {(E q u i t y)} = \sum_ {i \in E n t i t y _ {k}} \delta_ {i} * d _ {i} ^ {(E q u i t y)} * M F _ {i} ^ {(t y p e)}
$$

where  $i \in Entity_k$  refers to trades of entity  $k$ . That is, the effective notional for each entity is the sum of the trade-level adjusted notional amounts (cf. paragraphs 157-158) multiplied by the supervisory delta adjustments (cf. paragraph 159) and the maturity factor (cf. paragraph 164).

Add-on for commodity derivatives

178. The add-on for the asset class is given by:

$$
A d d O n ^ {(\text {C o m})} = \sum_ {j} A d d O n _ {H S _ {j}} ^ {(\text {C o m})}
$$

where the sum is taken over all hedging sets.

179. Within each hedging set, a single factor model is used to divide the risk of the same type of commodities into a systematic component and an idiosyncratic component, consistent with the approach taken for credit and equity derivatives. Full offsetting/hedging benefits is allowed between all derivative transactions referencing the same type of commodity, forming a commodity type-level effective notional. Partial offsetting/hedging benefits is allowed within each hedging set between the same type of commodities (supervisory correlation factors are defined for each) while no offsetting/hedging benefits is permitted between hedging sets. In summary, we have:

$$
A d d O n _ {H S _ {j}} ^ {(C o m)} = \left[ \left(\rho_ {j} ^ {(C o m)} * \sum_ {k} A d d O n (T y p e _ {k} ^ {j})\right) ^ {2} + \left(1 - \left(\rho_ {j} ^ {(C o m)}\right) ^ {2}\right) * \sum_ {k} \left(A d d O n (T y p e _ {k} ^ {j})\right) ^ {2} \right] ^ {\frac {1}{2}}
$$

where  $\rho_{j}^{(Com)}$  is the appropriate correlation factor corresponding to the hedging set  $j$ . The addition and the effective notional of the commodity type  $k$  are respectively given by:

$$
A d d O n \left(T y p e _ {k} ^ {j}\right) = S F _ {T y p e _ {k} ^ {j}} ^ {(C o m)} * E f f e c t i v e N o t i o n a l _ {k} ^ {(C o m)}
$$

and

$$
E f f e c t i v e N o t i o n a l _ {k} ^ {(C o m)} = \sum_ {i \in T y p e _ {k} ^ {j}} \delta_ {i} * d _ {i} ^ {(C o m)} * M F _ {i} ^ {(t y p e)}
$$

where  $i \in Type_k^j$  refers to trades of commodity type  $k$  in hedging set  $j$ . That is, the effective notional for each commodity type is the sum of the trade-level adjusted notional amounts (cf. paragraph 157-158) multiplied by the supervisory delta adjustments (cf. paragraph 159) and the maturity factor (cf. paragraph 164).

180. This approach assumes that the four broad categories of commodity derivatives cannot be used to hedge one another (eg a forward contract on crude oil cannot hedge a forward contract on corn). However, within each category, the different commodity types are more likely to demonstrate some stable, meaningful joint dynamics.  
181. Defining individual commodity types is operationally difficult. In fact, it is impossible to fully specify all relevant distinctions between commodity types so that all basis risk is captured. For example crude oil could be a commodity type within the energy hedging set, but in certain cases this definition could omit a substantial basis risk between different types of crude oil (West Texas Intermediate, Brent, Saudi Light, etc).  
182. Commodity type hedging sets have been defined without regard to characteristics such as location and quality. For example, the energy hedging set contains commodity types such as crude oil, electricity, natural gas and coal. However, national supervisors may require banks to use more refined definitions of commodities when they are significantly exposed to the basis risk of different products within those commodity types.  
183. Table 2 includes the supervisory factors, correlations and supervisory option volatility add-ons for each asset class and subclass.

# Summary table of supervisory parameters

Table 2  

<table><tr><td>Asset Class</td><td>Subclass</td><td></td><td>Supervisory factor</td><td>Correlation</td><td>Supervisory option volatility</td></tr><tr><td>Interest rate</td><td></td><td></td><td>0.50%</td><td>N/A</td><td>50%</td></tr><tr><td>Foreign exchange</td><td></td><td></td><td>4.0%</td><td>N/A</td><td>15%</td></tr><tr><td rowspan="7">Credit, Single Name</td><td>AAA</td><td></td><td>0.38%</td><td>50%</td><td>100%</td></tr><tr><td>AA</td><td></td><td>0.38%</td><td>50%</td><td>100%</td></tr><tr><td>A</td><td></td><td>0.42%</td><td>50%</td><td>100%</td></tr><tr><td>BBB</td><td></td><td>0.54%</td><td>50%</td><td>100%</td></tr><tr><td>BB</td><td></td><td>1.06%</td><td>50%</td><td>100%</td></tr><tr><td>B</td><td></td><td>1.6%</td><td>50%</td><td>100%</td></tr><tr><td>CCC</td><td></td><td>6.0%</td><td>50%</td><td>100%</td></tr><tr><td>Credit, Index</td><td>IG</td><td></td><td>0.38%</td><td>80%</td><td>80%</td></tr><tr><td></td><td>SG</td><td></td><td>1.06%</td><td>80%</td><td>80%</td></tr><tr><td>Equity, Single Name</td><td></td><td></td><td>32%</td><td>50%</td><td>120%</td></tr><tr><td>Equity, Index</td><td></td><td></td><td>20%</td><td>80%</td><td>75%</td></tr><tr><td rowspan="5">Commodity</td><td>Electricity</td><td></td><td>40%</td><td>40%</td><td>150%</td></tr><tr><td>Oil/Gas</td><td></td><td>18%</td><td>40%</td><td>70%</td></tr><tr><td>Metals</td><td></td><td>18%</td><td>40%</td><td>70%</td></tr><tr><td>Agricultural</td><td></td><td>18%</td><td>40%</td><td>70%</td></tr><tr><td>Other</td><td></td><td>18%</td><td>40%</td><td>70%</td></tr></table>

184. For a basis transaction hedging set, the supervisory factor applicable to its relevant asset class must be multiplied by one-half. For a volatility transaction hedging set, the supervisory factor applicable to its relevant asset class must be multiplied by a factor of five.

Treatment of multiple margin agreements and multiple netting sets

185. If multiple margin agreements apply to a single netting set, the netting set must be divided into sub-netting sets that align with their respective margin agreement. This treatment applies to both RC and PFE components.  
186. If a single margin agreement applies to several netting sets, replacement cost at any time is determined by the sum of netting set unmargined exposures minus the collateral available at that time (including both VM and N/CA). Since it is problematic to allocate the common collateral to individual netting sets, RC for the entire margin agreement is:

$$
R C _ {M A} = \max  \left\{\sum_ {N S \in M A} \max  \left\{V _ {N S}; 0 \right\} - C _ {M A}; 0 \right\}
$$

where the summation  $\mathsf{NS} \in \mathsf{MA}$  is across the netting sets covered by the margin agreement (hence the notation),  $V_{NS}$  is the current mark-to-market value of the netting set  $NS$  and  $C_{MA}$  is the cash equivalent value of all currently available collateral under the margin agreement.

187. Where a single margin agreement applies to several netting sets as described in paragraph 186, collateral will be exchanged based on mark-to-market values that are netted across all transactions covered under the margin agreement, irrespective of netting sets. That is, collateral exchanged on a net basis may not be sufficient to cover PFE.

In this situation, therefore, the PFE add-on must be calculated according to the unimargined methodology. Netting set-level PFEs are then aggregated. Mathematically:

$$
P F E _ {\text {M A}} = \sum_ {\mathcal {N S} \in \mathcal {M A}} P F E _ {\mathcal {N S}} ^ {(\text {u n m a r g i n e d})}
$$

where  $PFE_{NS}^{(unmargined)}$  is the PFE add-on for the netting set  $NS$  calculated according to the unmargined requirements.

Section VII (deleted)

Paragraphs 91-96(vi) (deleted)

A. References to the SM, CEM, and IMM shortcut method

(a) Introduction

The last sentence of paragraph 1 will be amended by replacing the words "the standardised method or the current exposure method" with the words "the Standardised Approach for counterparty credit risk" and retaining the language in the remainder of the paragraph.

(b) Section I - Definitions

Paragraph 2.C. (definitions of Netting sets, hedging sets, and related terms) will be amended as follows: "Hedging Set is a set of transactions within a single netting set within which full or partial offsetting is recognised for the purpose of calculating the PFE add-on of the Standardised Approach for counterparty credit risk."

(c) Section IV - Approval to adopt an internal modelling method to estimate EAD

Paragraph 21 will be amended by replacing the words "the standardised method or the current exposure method" with the words "the Standardised Approach for counterparty credit risk" and retaining the language in the remainder of the paragraph.

Paragraph 22 will be amended by replacing the words "either the standardised method or the current exposure method" with the words "the Standardised Approach for counterparty credit risk". The rest of the paragraph will be deleted.

Paragraph 23 will be amended by replacing the words "any of the three" with the words "either of the" and retaining the language in the remainder of the paragraph.

Paragraph 24 will be amended by replacing the words "either the current exposure or standardised methods" with the words "the Standardised Approach for counterparty credit risk" and retaining the language in the remainder of the paragraph.

(d) Section VIII - Treatment of mark-to market counterparty risk losses (CVA capital charge)

Paragraph 98 will be amended by removing the words "For banks using the short cut method (paragraph 41 of Annex 4) for margined trades, the paragraph 99 should be applied."

Paragraph 99 will be amended by removing the subparagraph that begins with the words "Banks using the short cut method for collateralised OTC derivatives..." and by replacing the words "CEM (Current Exposure Method) or SM (Standardised Method)" in the first and second lines of the following subparagraph with the words "the Standardised Approach for counterparty credit risk (SA-CCR)"), and by replacing the words "CEM or SM" with the words "the SA-CCR" in the eighth line of the same paragraph. The language in the remainder of the subparagraph will be retained.

Paragraph 104 will be amended by replacing the words "IMM, SM or CEM" with the words "IMM or SA-CCR" in the third bullet following the formula, and retaining the language in the remainder of the paragraph.

Paragraph 105 will be amended by replacing the words "CEM or SM, respectively" with the words "the SA-CCR" in the first paragraph, and by replacing the words "CEM or SM" in subparagraph C.i. with the word "SA-CCR". The language in the remainder of the paragraph will be retained.

(e) Section IX - Central counterparties

Paragraph 113 will be amended by replacing the words "either the CEM or the Standardised Method" with "the SA-CCR" and retaining the language in the remainder of the paragraph.

Paragraph 123 will be amended by replacing the word "CEM" in the first bullet following the formula with the words "the SA-CCR", and retaining the language in the remainder of the paragraph.

# IV. Other revisions to Basel III: A global regulatory framework

# A. Abbreviations

The "CEM" and "SM" entries will be deleted and a new entry, "SA-CCR - Standardised Approach for counterparty credit risk", will be added in its correct alphabetical position.

# B. Part 4: Third Pillar; Section II Disclosure requirements

Table 8 (General disclosure for exposures related to counterparty credit risk) will be amended by replacing the words "IMM, SM or CEM" with "IMM or SA-CCR", and retaining the language in the remainder of the table.

# Annex 4a

# Application of the SA-CCR to sample portfolios<sup>20</sup>

# Example 1

Netting set 1 consists of three interest rates derivatives: two fixed versus floating interest rate swaps and one purchased physically-settled European swaption. The table below summarises the relevant contractual terms of the three derivatives.

<table><tr><td>Trade
#</td><td>Nature</td><td>Residual maturity</td><td>Base currency</td><td>Notional 
(thousands)</td><td>Pay Leg (*)</td><td>Receive Leg (*)</td><td>Market value 
(thousands)</td></tr><tr><td>1</td><td>Interest rate swap</td><td>10 years</td><td>USD</td><td>10,000</td><td>Fixed</td><td>Floating</td><td>30</td></tr><tr><td>2</td><td>Interest rate swap</td><td>4 years</td><td>USD</td><td>10,000</td><td>Floating</td><td>Fixed</td><td>-20</td></tr><tr><td>3</td><td>European swaption</td><td>1 into 10 years</td><td>EUR</td><td>5,000</td><td>Floating</td><td>Fixed</td><td>50</td></tr></table>

(*) For the swaption, the legs are those of the underlying swap.

All notional amounts and market values in the table are given in USD.

The netting set is not subject to a margin agreement and there is no exchange of collateral (independent amount/initial margin) at inception. According to the SA-CCR formula, the EAD for unmargined netting sets is given by:

$$
E A D = a l p h a ^ {*} (R C + m u l t i p l i e r ^ {*} A d d O n ^ {a g g r e g a t e})
$$

The replacement cost is calculated at the netting set level as a simple algebraic sum (floored at zero) of the derivatives' market values at the reference date. Thus, using the market values indicated in the table (expressed in thousands):

$$
R C = \max  \{V - C; 0 \} = \max  \{3 0 - 2 0 + 5 0; 0 \} = 6 0
$$

Since  $V-C$  is positive (equal to  $V$ , ie 60,000), the value of the multiplier is 1, as explained in the paragraphs 148-149 of Annex 4.

All the transactions in the netting set belong to the interest rate asset class. For the calculation of the interest rate add-on, the three trades must be assigned to a hedging set (based on the currency) and to a maturity bucket (based on the end date of the transaction). In this example, the netting set is comprised of two hedging sets, since the trades refer to interest rates denominated in two different currencies (USD and EUR). Within hedging set "USD", trade 1 falls into the third maturity bucket ( $>$ 5 years) and trade 2 falls into the second maturity bucket (1-5 years). Trade 3 falls into the third maturity bucket ( $>$ 5 years) of hedging set "EUR".

For each IR trade  $i$ , the adjusted notional is calculated according to:

$$
d _ {i} ^ {(I R)} = \text {T r a d e N o t i o n a l} * \frac {\exp (- 0 . 0 5 * S _ {i}) - \exp (- 0 . 0 5 * E _ {i})}{0 . 0 5}
$$

where the second factor in the product is the supervisory duration (SD).  $S_{i}$  and  $E_{i}$  represent the start date and end date, respectively, of the time period referenced by the interest rate transactions, as defined in accordance with paragraphs 155 and 157 of Annex 4.

<table><tr><td>Trade
#</td><td>Hedging set</td><td>Time 
bucket</td><td>Notional 
(thousands)</td><td>Si</td><td>Ei</td><td>SDi</td><td>Adjusted 
notional 
(thousands)</td><td>Supervisory 
delta</td></tr><tr><td>1</td><td>USD</td><td>3</td><td>10,000</td><td>0</td><td>10</td><td>7.87</td><td>78,694</td><td>1</td></tr><tr><td>2</td><td>USD</td><td>2</td><td>10,000</td><td>0</td><td>4</td><td>3.63</td><td>36,254</td><td>-1</td></tr><tr><td>3</td><td>EUR</td><td>3</td><td>5,000</td><td>1</td><td>11</td><td>7.49</td><td>37,428</td><td>-0.27</td></tr></table>

A supervisory delta is assigned to each trade in accordance with paragraph 159 of Annex 4. In particular, trade 1 is long in the primary risk factor (the reference floating rate) and is not an option so the supervisory delta is equal to 1. Trade 2 is short in the primary risk factor and is not an option; thus, the supervisory delta is equal to -1. Trade 3 is an option to enter into an interest rate swap that is short in the primary risk factor and therefore is treated as a bought put option. As such, the supervisory delta is determined by applying the relevant formula in paragraph 159, using  $50\%$  as the supervisory option volatility and 1 (year) as the option exercise date. In particular, assuming that the underlying price (the appropriate forward swap rate) is  $6\%$  and the strike price (the swaption's fixed rate) is  $5\%$ , the supervisory delta is:

$$
\delta_ {i} = - \Phi \left(- \frac {\ln (0 . 0 6 / 0 . 0 5) + 0 . 5 * (0 . 5) ^ {2} * 1}{0 . 5 * \sqrt {1}}\right) = - 0. 2 7
$$

The effective notional of each maturity bucket of each hedging set is calculated according to:

$$
D _ {j k} ^ {(I R)} = \sum_ {i \in \{C c y _ {j}, M B _ {k} \}} \delta_ {i} * d _ {i} ^ {(I R)} * M F _ {i} ^ {(t y p e)}
$$

$\mathsf{MF}_i$  is 1 for all the trades (since they are unimargined and have remaining maturities in excess of one year) in the example and  $\delta_{i}$  is the supervisory delta. In particular:

Hedging set USD, time bucket 2:  $D_{\text{USD},2}^{(IR)} = -1 * 36,254 = -36,254$

Hedging set USD, time bucket 3:  $D_{USD,2}^{(IR)} = 1 * 78,694 = 78,694$

Hedging set EUR, time bucket 3:  $D_{EUR,3}^{(IR)} = -0.27 * 37,428 = -10,083$

Then, aggregation of effective notionals across time buckets inside the same hedging set is performed on the basis of the following formula:

$$
E f f e c t i v e N o t i o n a l _ {j} ^ {(I R)} = \left[ \left(D _ {j 1} ^ {(I R)}\right) ^ {2} + \left(D _ {j 2} ^ {(I R)}\right) ^ {2} + \left(D _ {j 3} ^ {(I R)}\right) ^ {2} + 1. 4 * D _ {j 1} ^ {(I R)} * D _ {j 2} ^ {(I R)} + 1. 4 * D _ {j 2} ^ {(I R)} * D _ {j 3} ^ {(I R)} + 0. 6 * D _ {j 1} ^ {(I R)} * D _ {j 3} ^ {(I R)} \right] ^ {\frac {1}{2}}
$$

Thus, the effective notional amount for hedging set "USD" is given by:

$$
\text {E f f e c t i v e N o t i o n a l} _ {\text {U S D}} ^ {(\text {I R})} = \left[ (- 3 6, 2 5 4) ^ {2} + 7 8, 6 9 4 ^ {2} + 1. 4 * (- 3 6, 2 5 4) * 7 8, 6 9 4 \right] ^ {\frac {1}{2}} = 5 9, 2 7 0
$$

Since hedging set "EUR" is made of only one maturity bucket, its effective notional is:

$$
\text {E f f e c t i v e N o t i o n a l} _ {\text {E U R}} ^ {(\text {I R})} = \left[ (- 1 0, 0 8 3) ^ {2} \right] ^ {\frac {1}{2}} = 1 0, 0 8 3
$$

The effective notional amounts should be multiplied by the SF (that for interest rates is equal to  $0.5\%$ ) and summed up across hedging sets:

$$
A d D O n ^ {I R} = 0.5 \% * 59,270 + 0.5 \% * 10,083 = 347
$$

For this netting set the interest rate add-on is also the aggregate add-on because there are no derivatives belonging to other asset classes. Finally, the SA-CCR exposure is calculated by adding up the RC component and PFE component and multiplying the result by 1.4:

$$
E A D = 1. 4 ^ {*} (6 0 + 1 ^ {*} 3 4 7) = 5 6 9
$$

where a value of 1 is used for the multiplier.

# Example 2

Netting set 2 consists of three credit derivatives: one long single-name CDS written on Firm A (rated AA), one short single-name CDS written on Firm B (rated BBB), and one long CDS index (investment grade). The table below summarises the relevant contractual terms of the three derivatives.

<table><tr><td>Trade #</td><td>Nature</td><td>Reference entity / index name</td><td>Rating reference entity</td><td>Residual maturity</td><td>Base currency</td><td>Notional (thousands)</td><td>Position</td><td>Market value (thousands)</td></tr><tr><td>1</td><td>Single-name CDS</td><td>Firm A</td><td>AA</td><td>3 years</td><td>USD</td><td>10,000</td><td>Protectio n buyer</td><td>20</td></tr><tr><td>2</td><td>Single-name CDS</td><td>Firm B</td><td>BBB</td><td>6 years</td><td>EUR</td><td>10,000</td><td>Protectio n seller</td><td>-40</td></tr><tr><td>3</td><td>CDS index</td><td>CDX.IG 5y</td><td>Investment grade</td><td>5 years</td><td>USD</td><td>10,000</td><td>Protectio n buyer</td><td>0</td></tr></table>

All notional amounts and market values in the table are in USD. As in the previous example, the netting set is not subject to a margin agreement and there is no exchange of collateral (independent amount/initial margin) at inception. The EAD formulation for unmargined netting sets is:

$$
E A D = a l p h a ^ {*} (R C + m u l t i p l i e r * A d d O n ^ {a g g r e g a t e})
$$

The replacement cost is:

$$
R C = \max  \{V - C; 0 \} = \max  \{2 0 - 4 0 + 0; 0 \} = 0
$$

Since in this example V-C is negative (equal to V, ie -20), the multiplier will be activated (ie it will be less than 1). Before calculating its value, the aggregate add-on needs to be determined.

In order to calculate the aggregate add-on, first, the adjusted notional of each trade must be calculated by multiplying the notional amount with the supervisory duration, where the latter is determined based on the start date  $S_{i}$  and the end date  $E_{i}$  in accordance with the formula in paragraph 157 of Annex 4. The results are shown in the table below.

<table><tr><td>Trade
#</td><td>Notional 
(thousands)</td><td>Si</td><td>Ei</td><td>SDi</td><td>Adjusted 
notional (thousands)</td><td>Supervisory 
delta</td></tr><tr><td>1</td><td>10,000</td><td>0</td><td>3</td><td>2.79</td><td>27,858</td><td>1</td></tr><tr><td>2</td><td>10,000</td><td>0</td><td>6</td><td>5.18</td><td>51,836</td><td>-1</td></tr><tr><td>3</td><td>10,000</td><td>0</td><td>5</td><td>4.42</td><td>44,240</td><td>1</td></tr></table>

The appropriate supervisory delta must be assigned to each trade: in particular, since trade 1 and trade 3 are long in the primary risk factor (CDS spread), their delta is 1; on the contrary, the supervisory delta for trade 2 is -1.

Since all derivatives refer to different entities (single names/indices), it is not necessary to aggregate the trades at the entity level. Thus, the entity-level effective notional is equal to the adjusted notional times the supervisory delta (the maturity factor is 1 for all three derivatives). A supervisory factor is assigned to each single-name entity based on the rating of the reference entity (0.38% for AA-rated firms and 0.54% for BBB-rated firms). For CDS indices, the SF is assigned according to whether the index is investment or speculative grade; in this example, its value is 0.38% since the index is investment grade. Thus, the entity level add-ons are the following:

$$
\operatorname {Addon}(\text{FirmA}) = 0.38\% *27,858 = 106
$$

$$
\operatorname {Addon}(\text{FirmB}) = 0.54\% *(-51,836) = -280
$$

$$
\text{Addon} (\text{CDX.IG}) = 0.38 \% * 44,240 = 168
$$

Once the entity-level add-ons are calculated, the following formula can be applied:

$$
A d d O n ^ {(C r e d i t)} = \left[ \underbrace {\left(\sum_ {k} \rho_ {k} ^ {(C r e d i t)} * A d d O n (E n t i t y _ {k})\right) ^ {2}} _ {\text {s y s t e m a t i c c o m p o n e n t}} + \underbrace {\sum_ {k} \left(1 - \left(\rho_ {k} ^ {(C r e d i t)}\right) ^ {2}\right) * \left(A d d O n (E n t i t y _ {k})\right) ^ {2}} _ {\text {i d i o s y n c r a t i c c o m p o n e n t}} \right] ^ {\frac {1}{2}}
$$

Where the correlation parameter  $\rho_{k}^{(\text{Credit})}$  is equal to 0.5 for the single-name entities (Firm A and Firm B) and 0.8 for the index (CDX.IG).

The following table shows a simple way to calculate of the systematic and idiosyncratic components in the formula.

<table><tr><td>Reference Entity</td><td>Entity-level add-on</td><td>Correlation parameter (r)</td><td>Entity-level add-on times r</td><td>(Entity-level add-on)2</td><td>1-r2</td><td>(Entity-level add-on)2 times (1-r2)</td></tr><tr><td>Firm A</td><td>106</td><td>0.5</td><td>52.9</td><td>11,207</td><td>0.75</td><td>8,405</td></tr><tr><td>Firm B</td><td>-280</td><td>0.5</td><td>-140</td><td>78,353</td><td>0.75</td><td>58,765</td></tr><tr><td>CDX.IG</td><td>168</td><td>0.8</td><td>134.5</td><td>28,261</td><td>0.36</td><td>10,174</td></tr><tr><td rowspan="2">sum = 
(∑m)2 =</td><td colspan="2"></td><td>47.5</td><td colspan="2"></td><td rowspan="2">77,344</td></tr><tr><td colspan="2"></td><td>2,253</td><td colspan="2"></td></tr></table>

According to the calculations in the table, the systematic component is 2,253, while the idiosyncratic component is 77,344.

Thus,

$$
A d d O n ^ {(C r e d i t)} = [ 2, 2 5 3 + 7 7, 3 4 4 ] ^ {\frac {1}{2}} = 2 8 2
$$

The value of the multiplier can now be calculated as:

$$
m u l t i p l i e r = \min  \left\{1; 0. 0 5 + 0. 9 5 * \exp \left(\frac {- 2 0}{2 * 0 . 9 5 * 2 8 2}\right) \right\} = 0. 9 6 5
$$

Finally, aggregating the replacement cost and the PFE component and multiplying the result by the alpha factor of 1.4, the exposure is:

$$
E A D = 1. 4 ^ {*} (0 + 0. 9 6 5 * 2 8 2) = 3 8 1.
$$

# Example 3

Netting set 3 consists of three commodity forward contracts:  

<table><tr><td>Trade
#</td><td>Nature</td><td>Underlying</td><td>Direction</td><td>Residual maturity</td><td>Notional</td><td>Market value</td></tr><tr><td>1</td><td>Forward</td><td>(WTI) Crude Oil</td><td>Long</td><td>9 months</td><td>10,000</td><td>-50</td></tr><tr><td>2</td><td>Forward</td><td>(Brent) Crude Oil</td><td>Short</td><td>2 years</td><td>20,000</td><td>-30</td></tr><tr><td>3</td><td>Forward</td><td>Silver</td><td>Long</td><td>5 years</td><td>10,000</td><td>100</td></tr></table>

There is no margin agreement and no collateral. The replacement cost is given by:

$$
R C = \max  \{V - C; 0 \} = \max  \{1 0 0 - 3 0 - 5 0; 0 \} = 2 0
$$

Because the replacement cost is positive and there is no exchange of collateral (so the bank has not received excess collateral), the multiplier is equal to 1.

To calculate the add-on, the trades need to be classified into hedging sets (energy, metals, agricultural and other) and, within each hedging set, into commodity types. In this case:

<table><tr><td>Hedging Set</td><td>Commodity Type</td><td>Trades</td></tr><tr><td rowspan="4">Energy</td><td>Crude Oil</td><td>1 and 2</td></tr><tr><td>Natural Gas</td><td>None</td></tr><tr><td>Coal</td><td>None</td></tr><tr><td>Electricity</td><td>None</td></tr><tr><td rowspan="3">Metals</td><td>Silver</td><td>3</td></tr><tr><td>Gold</td><td>None</td></tr><tr><td>…</td><td>…</td></tr><tr><td rowspan="2">Agricultural</td><td>…</td><td>…</td></tr><tr><td>…</td><td>…</td></tr><tr><td>Other</td><td>…</td><td>…</td></tr></table>

For purposes of this calculation, the bank can ignore the basis difference between the WTI and Brent forward contracts since they belong to the same commodity type, "Crude Oil" (unless the national supervisor requires the bank to use a more refined definition of commodity types). Therefore, these contracts can be aggregated into a single effective notional, taking into account each trade's supervisory delta and maturity factor. In particular, the supervisory delta is 1 for trade 1 (long position) and -1 for trade 2 (short position). Since the remaining maturity of trade 1 is nine months (thus, shorter than 1 year) and the trade is unimargined, its maturity factor is

$$
M F _ {t r a d e 1} = \sqrt {9 / 1 2}
$$

The maturity factor is 1 for trade 2 (remaining maturity greater than 1 year and unmargined trade). Thus, the effective notional for commodity type "Crude Oil" is

$$
E f f e c t i v e N o t i o n a l _ {\text {C r u d e O i l}} = 1 ^ {*} 1 0, 0 0 0 ^ {*} \sqrt {9 / 1 2} + (- 1) ^ {*} 2 0, 0 0 0 ^ {*} 1 = - 1 1, 3 4 0.
$$

where the supervisory delta has been assigned to each trade (+1 for long and -1 for short). The effective notional amount must be multiplied by the supervisory factor for Oil/Gas (18%) to obtain the add-on for the Crude Oil commodity type:

$$
\mathrm{AddOn}\left(\text {Type} _ {\text {CrudeOil}} ^ {\text {Energy}}\right) = 18 \% * (-11,340) = -2,041.
$$

The next step, in theory, is to calculate the add-on for the hedging set "Energy" according to the formula:

$$
A d d O n _ {E n e r g y} ^ {(C o m)} = \left[ \underbrace {\left(\rho_ {E n e r g y} ^ {(C o m)} * \sum_ {k} A d d O n (T y p e _ {k} ^ {E n e r g y})\right) ^ {2}} _ {s y s t e m a t i c c o m p o n e n t} + \underbrace {\left(1 - \left(\rho_ {E n e r g y} ^ {(C o m)}\right) ^ {2}\right) * \sum_ {k} \left(A d d O n (T y p e _ {k} ^ {E n e r g y})\right) ^ {2}} _ {i d i s o n c r a t i c c o m p o n e n t} \right] ^ {\frac {1}{2}}
$$

However, in our example, only one commodity type within the "Energy" hedging set is populated (ie all other commodity types have a zero add-on). Therefore,

$$
A d d O n _ {E n e r g y} ^ {(C o m)} = \left[ (0. 4 * (- 2, 0 4 1)) ^ {2} + (1 - (0. 4) ^ {2}) * (- 2, 0 4 1) ^ {2} \right] ^ {\frac {1}{2}} = 2, 0 4 1.
$$

This calculation shows that, when there is only one commodity type within a hedging set, the hedging-set add-on is equal (in absolute value) to the commodity-type add-on.

Similarly, for commodity type "Silver" in the "Metals" hedging set, we have

$$
\text {E f f e c t i v e N o t i o n a l} _ {\text {S i l v e r}} = 1 * 1 0, 0 0 0 * 1 = 1 0, 0 0 0
$$

since the supervisory delta and maturity factor for trade 3 are both equal to 1. Furthermore, since the "Metals" hedging set includes only the "Silver" commodity type in this example:

$$
\mathrm{AddOn}_{\text{Metals}}^{\text{(Com)}} = \mathrm{AddOn}\left(\text{Type}_{\text{Silver}}^{\text{Metals}}\right) = 18\% *10,000 = 1,800.
$$

The aggregate add-on for the commodity derivative asset class is:

$$
\mathrm {A d d O n} ^ {(\mathrm {C o m})} = \mathrm {A d d O n} _ {\text {E n e r g y}} ^ {(\mathrm {C o m})} + \mathrm {A d d O n} _ {\text {M e t a l s}} ^ {(\mathrm {C o m})} = 2, 0 4 1 + 1, 8 0 0 = 3, 8 4 1.
$$

Finally, the exposure is:

$$
\mathrm {E A D} = 1. 4 ^ {*} (2 0 + 1 ^ {*} 3, 8 4 1) = 5, 4 0 6.
$$

# Example 4

Netting set 4 consists of the combined trades of Examples 1 and 2. There is no margin agreement and no collateral.

The replacement cost of the combined netting set is:

$$
R C = \max  \{V - C; 0 \} = \max  \{3 0 - 2 0 + 5 0 + 2 0 - 4 0 + 0; 0 \} = 4 0
$$

The add-on for the combined netting set is the sum of add-ons for each asset class. In this case, there are two asset classes, interest rates and credit:

$$
A d d O n ^ {a g g r e g a t e} = A d d O n ^ {(R)} + A d d O n ^ {(C r e d i t)} = 3 4 7 + 2 8 2 = 6 2 9
$$

where the add-ons for interest rate and credit derivatives have been copied from Examples 1 and 2. Because the netting set has a positive replacement cost and there is no exchange of collateral (so the bank has not received excess collateral), the multiplier is equal to 1. Finally, the exposure is:

$$
E A D = 1. 4 ^ {*} (4 0 + 1 ^ {*} 6 2 9) = 9 3 6.
$$

# Example 5

Netting set 5 consists of the combined trades of Examples 1 and 3. However, instead of being unimargined (as assumed in those examples), the trades are subject to a margin agreement with the following specifications:

<table><tr><td>Margin frequency</td><td>Threshold</td><td>Minimum Transfer Amount (thousands)</td><td>Independent Amount (thousands)</td><td>Net collateral currently held by the bank (thousands)</td></tr><tr><td>Weekly</td><td>0</td><td>5</td><td>150</td><td>200</td></tr></table>

The above table depicts a situation in which the bank received from the counterparty a net independent amount of 150 (taking into account the net amount of initial margin posted by the counterparty and any unsegregated initial margin posted by the bank). The total net collateral currently held by the bank is 200, which includes 50 for variation margin received and 150 for the net independent amount.

First, we determine the replacement cost. The net collateral currently held is 200 and the NICA is equal to the independent amount (that is, 150). The current market value of the netting set is:

$$
\mathrm {V} = 3 0 - 2 0 + 5 0 - 5 0 - 3 0 + 1 0 0 = 8 0
$$

Therefore:

$$
R C = \max  (V - C; T H + M T A - N I C A; 0) = \max  (8 0 - 2 0 0; 0 + 5 - 1 5 0; 0) = 0
$$

Second, it is necessary to recalculate the interest rate and commodity add-ons, based on the value of the maturity factor for margined transactions, which depends on the margin period of risk. For daily re-margining, the margin period of risk (MPOR) would be 10 days. In accordance with paragraph 41(iii) of Annex 4, for re-margining with a periodicity of N days, the MPOR is equal to ten days plus N days minus one day. Thus, for weekly re-margining (every five business days),  $\text{MPOR} = 10 + 5 - 1 = 14$ . Hence, the re-scaled maturity factor for the trades in the netting set is:21

$$
\mathrm {M F} _ {\mathrm {i}} ^ {(\text {M a r g i n e d})} = \frac {3}{2} \sqrt {\frac {\text {M P O R}}{1 \text {y e a r}}} = 1. 5 * \sqrt {1 4 / 2 5 0}.
$$

Repeating the calculation of Example 1 with the new value of the maturity factor, we get:

Hedging set USD, time bucket 2:  $\mathbf{D}_{\mathrm{USD},2} = (-1)^{*}36,254^{*}\left(1.5^{*}\sqrt{14 / 250}\right) = -12,869$

Hedging set USD, time bucket 3:  $\mathrm{D}_{\mathrm{USD},3} = 1^{*}78,694^{*}\left(1.5^{*}\sqrt{14 / 250}\right) = 27,934$

Hedging set EUR, time bucket 3:  $D_{\text{EUR},3} = (-0.27)^{*}37,428^{*}\left(1.5^{*}\sqrt{14 / 250}\right) = -3,579$

The effective notional amount for hedging sets "USD" and "EUR" are given by:

$$
\text {E f f e c t i v e N o t i o n a l} _ {\mathrm {U S D}} ^ {(\mathrm {I R})} = \left[ (- 1 2, 8 6 9) ^ {2} + (2 7, 9 3 4) ^ {2} + 1. 4 * (- 1 2, 8 6 9) * 2 7, 9 3 4 \right] ^ {\frac {1}{2}} = 2 1, 0 3 9
$$

$$
\text {E f f e c t i v e N o t i o n a l} _ {\text {E U R}} ^ {(\text {I R})} = \left[ (- 3, 5 7 9) ^ {2} \right] ^ {\frac {1}{2}} = 3, 5 7 9
$$

The effective notional amounts can be multiplied by the SF (that for interest rates is equal to  $0.5\%$ ) and summed up across hedging sets:

$$
A d d O n ^ {(I R)} = 0.5 \% * 21,039 + 0.5 \% * 3,579 = 123
$$

Repeating the calculation of Example 3 with the new value of the maturity factor, we get for hedging set "Energy":

$$
\text {E f f e c t i v e N o t i o n a l} _ {\text {C r u d e O i l}} ^ {\text {E n e r g y}} = 1 * 1 0, 0 0 0 * \left(1. 5 * \sqrt {1 4 / 2 5 0}\right) + (- 1) * 2 0, 0 0 0 * \left(1. 5 * \sqrt {1 4 / 2 5 0}\right) = - 3, 5 5 0
$$

$$
\mathrm{AddOn}\left(\text {Type} _ {\text {CrudeOil}} ^ {\text {Energy}}\right) = 18 \% * (-3,550) = -639
$$

$$
\mathrm {A d d O n} _ {\text {E n e r g y}} ^ {\left(\mathrm {C o m}\right)} = \left[ (0. 4 * (- 6 3 9)) ^ {2} + (1 - (0. 4) ^ {2}) * (- 6 3 9) ^ {2} \right] ^ {\frac {1}{2}} = 6 3 9
$$

Similarly, for hedging set "Metals", we have

$$
E f f e c t i v e N o t i o n a l _ {\text {S i l v e r}} = 1 * 1 0, 0 0 0 * \left(1. 5 * \sqrt {1 4 / 2 5 0}\right) = 3, 5 5 0
$$

$$
\mathrm{AddOn}_{\text{Metals}}^{\mathrm{(Com)}} = \mathrm{AddOn}\left(\mathrm{Type}_{\text{Silver}}^{\mathrm{Metals}}\right) = 18\% *3,550 = 639.
$$

The overall add-on for the commodity derivative asset class is:

$$
\mathrm {A d d O n} ^ {(\mathrm {C o m})} = \mathrm {A d d O n} _ {\text {E n e r g y}} ^ {(\mathrm {C o m})} + \mathrm {A d d O n} _ {\text {M e t a l s}} ^ {(\mathrm {C o m})} = 6 3 9 + 6 3 9 = 1, 2 7 8.
$$

Since there are two asset classes (interest rate and commodity), the aggregate add-in is given by:

$$
A d d O n ^ {a g g r e g a t e} = A d d O n ^ {(I R)} + A d d O n ^ {(C o m)} = 1 2 3 + 1, 2 7 8 = 1, 4 0 1
$$

Third, we calculate the multiplier as a function of over-collateralisation and the new add-on:

$$
m u l t i p l i e r = \min  \left(1; 0. 0 5 + 0. 9 5 ^ {*} \exp \left(\frac {8 0 - 2 0 0}{2 ^ {*} 0 . 9 5 ^ {*} 1 , 4 0 1}\right)\right) = 0. 9 5 8
$$

Finally, the exposure is:

$$
E A D = 1. 4 ^ {*} (0 + 0. 9 5 8 ^ {*} 1, 4 0 1) = 1, 8 7 9
$$

# Annex 4b

# Effect of standard margin agreements on the SA-CCR formulation

The following examples illustrate the operation of the SA-CCR in the context of standard margin agreements. In particular, they relate to the formulation of replacement cost for margined trades, as depicted in paragraph 144 of Annex 4.

$$
R C = \max  \{V - C; T H + M T A - N I C A; 0 \}
$$

# Example 1

1. The bank currently has met all past variation margin (VM) calls so that the value of trades with its counterparty (€80 million) is offset by cumulative VM in the form of cash collateral received. There is a small "Minimum Transfer Amount" (MTA) of €1 million and a €0 "Threshold" (TH). Furthermore, an "Independent Amount" (IA) of €10 million is agreed in favour of the bank and none in favour of its counterparty. This leads to a credit support amount of €90 million, which is assumed to have been fully received as of the reporting date.

2. In this example, the first term in the replacement cost (RC) formula (V-C) is zero, since the value of the trades is offset by collateral received; €80 million – €90 million = negative €10. The second term (TH + MTA - NICA) of the replacement cost formula is negative €9 million (€0 TH + €1 million MTA – €10 million of net independent collateral amount held). The last term in the RC formula is always zero, which ensures that replacement cost is not negative. The greatest of the three terms (-€10 million, -€9 million, 0) is zero, so the replacement cost is zero. This is due to the large amount of collateral posted by the bank's counterparty.

# Example  $2^{22}$

3. The counterparty has met all VM calls but the bank has some residual exposure due to the MTA of €1 million in its master agreement, and has a €0 TH. The value of the bank's trades with the counterparty is €80 million and the bank holds €79.5 million in VM in the form of cash collateral. The bank holds in addition €10 million in independent collateral (here being an initial margin independent of VM, the latter of which is driven by mark-to-market (MtM) changes) from the counterparty and the counterparty holds €10 million in independent collateral from the bank (which is held by the counterparty in a non-segregated manner).

4. In this case, the first term of the replacement cost (V-C) is €0.5 million (€80 million - €79.5 million - €10 million + €10 million), the second term (TH+MTA-NICA) is €1 million (€0 TH + €1 million MTA - €10 million ICA held + €10 million ICA posted). The third term is zero. The greatest of these three terms (€0.5 million, €1 million, 0) is €1 million, which represents the largest exposure before collateral must be exchanged.

# Bank as a clearing member

5. The case of central clearing can be viewed from a number of perspectives. One example in which the replacement cost formula for margined trades can be applied is when the bank is a clearing member and is calculating replacement cost for its own trades with a central counterparty (CCP). In this case, the MTA and TH are generally zero. VM is usually exchanged at least daily and ICA in the form of a performance bond or initial margin is held by the CCP.

# Example 3

6. The bank, in its capacity as clearing member of a CCP, has posted VM to the CCP in an amount equal to the value of the trades it has with the CCP. The bank has posted cash as initial margin and the CCP holds the initial margin in a bankruptcy remote fashion. Assume that the value of trades with the CCP are negative €50 million, the bank has posted €50 million in VM and €10 million in IM to the CCP.  
7. In this case, the first term (V-C) is €0 ([-€50 million - (-€50 million)] - €0), ie the already posted VM reduces the V to zero. The second term (TH+MTA-NICA) is €0 (€0 + €0 - €0) since MTA and TH equal €0, and the IM held by the CCP is bankruptcy remote and does not affect NICA. Therefore, the replacement cost is €0.

# Example 4

8. Example 4 is the same as the Example 3, except that the IM posted to the CCP is not bankruptcy remote. In this case, the first term (V-C) of the replacement cost is €10 million ([-€50 million - (-€50 million)] - [-€10 million]), the value of the second term (TH+MTA-NICA) is €10 million (€0+€0 - [-€10 million]), and the third term is zero. The greatest of these three terms (€10 million, €10 million, €0) is €10 million, representing the IM posted to the CCP which would be lost upon default of the CCP, including bankruptcy.

# Example 5

# Maintenance Margin Agreement

9. Some margin agreements specify that a counterparty (in this case, a bank) must maintain a level of collateral that is a fixed percentage of the MtM of the transactions in a netting set. For this type of margining agreement, ICA is the percentage of MtM that the counterparty must maintain above the net MtM of the transactions. For example, suppose the agreement states that a counterparty must maintain a collateral balance of at least 140% of the MtM of its transactions. Furthermore, suppose there is no TH and no MTA. ICA is the amount of collateral that is required to be posted to the bank by the counterparty. The MTM of the derivative transactions is €50. The counterparty posts €80 in cash collateral. ICA in this case is the amount that the counterparty is required to post above the MTM (140% * €50 - €50 = €20). Replacement cost is determined by the greater of the MtM minus the collateral (€50 - €80 = -€30), MTA+TH-NICA (€0+€0-€20 = -€20), and zero, thus the replacement cost is zero.

# Annex 4c

Flow chart of steps to calculate [interest rate] add-on

![](images/e0da7eadb4d0f7eff0a581859b4196c976d47037ce4a2505cb007a4d2b14edbc.jpg)