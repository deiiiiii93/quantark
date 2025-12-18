![](images/f095e40a58b23d4029879519e143a245feadabcaaff49dcb461e52a42dd8b88e.jpg)

# Quantitative Strategies Research Notes

November 1994

# Valuing Convertible Bonds as Derivatives

$\text{©} 1994$  Goldman, Sachs & Co. All rights reserved.

This material is for your private information, and we are not soliciting any action based upon it. This report is not to be construed as an offer to sell or the solicitation of an offer to buy any security in any jurisdiction where such an offer or solicitation would be illegal. The material is based upon information that we consider reliable, but we do not represent that it is accurate or complete, and it should not be relied upon as such. Opinions expressed are our current opinions as of the date appearing on this material only. While we endeavour to update on a reasonable basis the information discussed in this material, there may be regulatory, compliance, or other reasons that prevent us from doing so. We and our affiliates, officers, directors, partners, and employees, including persons involved in the preparation or issuance of this material may, from time to time, have long or short positions in, and buy or sell, the securities, or derivatives (including options) thereof, of companies mentioned herein.

This material has been issued by Goldman, Sachs & Co. and/or one of its affiliates and has been approved by Goldman Sachs International, a member of The Securities and Futures Authority, in connection with its distribution in the United Kingdom and by Goldman, Sachs Canada in connection with its distribution in Canada. This material is distributed in Hong Kong by Goldman Sachs (Asia) Limited; and in Japan by Goldman Sachs (Japan) Ltd. This material is not for distribution in the United Kingdom to private customers, as that term is defined under the rules of The Securities and Futures Authority; and any investments, including any convertible bonds or derivatives, mentioned in this material will not be made available by us to any such private customer. Neither Goldman, Sachs & Co. nor its representative in Seoul, Korea, is licensed to engage in the securities business in the Republic of Korea. Goldman Sachs International and its non-U.S. affiliates may, to the extent permitted under applicable law, have acted upon or used this research, to the extent it relates to non-U.S. issuers' prior to or immediately following its publication. Foreign-currency-denominated securities are subject to fluctuations in exchange rates that could have an adverse effect on the value or price of, or income derived from, the investment. In addition, investors in securities such as ADRs, the values of which are influenced by foreign currencies, effectively assume currency risk.

Further information on any of the securities mentioned in this material may be obtained upon request, and for this purpose persons in Italy should contact Goldman Sachs S.I.M. S.p.A. in Milan, or at its London branch office at 133 Fleet Street, and persons in Hong Kong should contact Goldman Sachs (Asia) Limited at 3 Garden Road. Unless governing law permits otherwise, you must contact a Goldman Sachs entity in your home jurisdiction if you want to use our services in effecting a transaction in the securities mentioned in this material.

# SUMMARY

Convertible bonds are derivative securities; they contain options on the underlying common stock and the straight debt of their issuer. In this paper, we describe a binomial one-factor model for calculating their theoretical value. The model assumes that all uncertainty in the future value of a convertible bond stems from the volatility of the underlying stock price. It uses a credit-adjusted discount rate when calculating the present value of future cash flows, in order to account for the credit sensitivity of a convertible. A computer implementation of the model, called Hydra, is available from the Quantitative Strategies Group.

Indrajit Bardhan (212) 902-5274

Alex Bergier [212] 902-2944

Emanuel Derman (212) 902-0129

Cemal Dosembet (212) 902-8856

Iraj Kani (212) 902-3561

Editorial: Barbara Dunn

The model described here is a descendant of an earlier model developed by Fischer Black and H.S. Huang, as described in the paper A Valuation Model for Options, Warrants and Convertibles (Goldman, Sachs & Co., 1988). We alone are responsible for the ways in which this model differs from the Black-Huang version. We are grateful to Fischer Black, Charles Eve, Michael Hintze, David Lockwood, Evan Misshula, Allan Teh, Gary Williams and Masayoshi Yoshihara for many conversations about the options features of convertibles. We also thank Piotr Karasinski who worked together with us on this model.

# Table of Contents

Introduction 1

Convertibles as Derivatives 1

What Influences A Convertible's Value? 4

The Terms 4

The Market Variables 5

How Convertibles Behave. 7

The Model 13

The Binomial Tree 13

The Credit-Adjusted Discount Rate 16

Model Summary 17

Some Subtleties 18

An Example 20

Using The Model To Hedge 24

The Call Adjustment Factor 25

Appendix: The Mathematics Of The Model 26

# INTRODUCTION

A convertible bond is a corporate debt security that can be converted into the issuer's common stock. The convertible bond owner receives periodic coupon payments from the issuer. In addition, at any time prior to maturity, the owner has the right to convert - that is, to exchange the security for a predetermined number of shares of the common stock. An owner who has not converted by maturity receives the full principal.

Most convertible bonds are subordinated debt of the issuer. In the event of bankruptcy, the claims of other bondholders take priority over convertible bondholders, who themselves have priority over owners of the preferred and common stock.

# Convertibles as Derivatives

A convertible is a hybrid security, part debt and part equity. It is a derivative security whose value is derived from the value of the debt and equity on which it ultimately depends. Our main focus in this paper is to develop an options-like model with which one can derive the current theoretical value of the convertible from the current values of its underlying straight debt and equity.

In order to highlight their derivative aspects, let's consider a default-free European-style convertible that can be exchanged for stock only at maturity. You can think of the convertible as the following portfolio of securities:

a straight bond; and  
a call that allows you to exchange the bond for equity.

Alternatively, you can think of the same convertible as

equity;  
- a put to exchange the equity for a straight bond; and  
- a swap to maturity that gives you the bond's coupons in exchange for the equity's dividends.

These complementary views of convertibles<sup>1</sup> allow them to appeal to both issuers and investors with differing preferences for risk.

From a fixed-income point of view, it is convenient to think of the convertible as a straight bond plus an equity call. A company issuing convertibles is giving away call options on its equity in exchange for a lower cost of debt funding. In periods of rising stock prices or of high volatility, the reduction in debt payments may be substantial.

From an equity standpoint, it may be more attractive to view the convertible as a portfolio of equity, a swap of stock dividends for coupons, and a put. A convertible investor is purchasing not just an equity stake, but also a put for downside protection, as well as coupons that are usually greater in value than the equity dividends.

The above two views of convertibles as portfolios that include derivatives describe bare-bones, European-style convertibles. In practice, convertibles usually allow American-style conversion, and often have additional call and put provisions. The calls allow the issuer to buy back the security at a predetermined price, and consequently limit the investor's returns when interest rates fall or stock prices rise. The puts allow the investor to return the convertible to the issuer for cash, and so provide additional downside protection.

Issuers have several reasons to use convertible financing. By issuing convertibles they can lower their cost of debt funding compared to straight debt alone. Lower-credit companies who may not be able to access the straight debt market can often still issue convertible debt. Companies who anticipate equity appreciation can use convertibles to defer equity financing to a time when growth has been achieved.

Investors find several features of convertibles appealing. They offer greater stability of income than common stock. They provide a yield that is often higher than the dividend yield of common stock. Finally, because they are often theoretically underpriced, they may provide a cheap source of common stock volatility.

There are a number of structural variations on the convertibles theme. Convertible preferred stock is preferred stock that can be converted to some number of shares of common stock of the company. Convertible preferred stocks usually have longer maturities than convertible bonds, and are issued as corporate equity. A less common security is an exchangeable convertible, which gives investors the right to convert into shares of stock of another company.

# WHAT INFLUENCES A CONVERTIBLE'S VALUE?

The Terms

In this section we describe the features of a convertible that affect its theoretical value.

The bond indenture details the specific terms of the convertible bond issue. Generally, they include the following quantities and features:

- Principal: The face value of the convertible bond, usually the amount for which the bond can be redeemed at maturity<sup>2</sup>.  
- Coupon: The annual interest rate as a percentage of principal paid to the owner.  
- Coupon frequency: The number of coupon payments per year, typically two for bonds and four for preferreds in the U.S.  
- Conversion ratio: The number of shares of the underlying stock for which the convertible bond can be exchanged. This ratio is usually established at issue, and changed only to account for stock dividends or splits of the underlying shares, so as to preserve the total equity value for which the convertible can be exchanged.  
- Conversion price: The price paid for each underlying share on conversion, assuming the bond principal is used to pay for the shares received. So, conversion price = principal ÷ conversion ratio.  
- Parity: The market value of the underlying shares; that is parity = conversion ratio × current stock price.  
- First conversion date: The first date after issue at which the bond can be converted into stock. Sometimes, there is a lock-out period after issue during which conversion is not allowed.  
- Call provisions: A call provision gives the issuer the right to buy back the bond. The price at which the issuer may buy the convertible is known as the call price and is specified in the call schedule, which gives the call price at each future time. While call prices for coupon bonds generally decrease in steps until maturity, zero-coupon bonds have a call price that accretes at the call accretion rate. Typically, bonds are call-protected for a certain number of years; they become callable only after a certain date. A call provision can be regarded as a call option that

has been sold by the investor to the issuer; it reduces the value of the bond when compared with a similar non-callable convertible bond.

Stock performance call provisions: These are call provisions subject to the constraint that the issuer can exercise the call only if the underlying stock price rises above a certain level, the provisional call level. A stock performance call reduces the value of the bond by forcing the investor to convert to stock and give up the remaining value of the option. The provisional call level is usually quoted as a percentage of the conversion price, typically around  $150\%$ .  
- Put provisions: These allow the bondholder to put the bond to the issuer for specific cash amounts (the put prices) on specific dates prior to maturity. This put schedule provides extra downside protection for the convertible bondholder and so adds to the value of the bond. A put provision can be regarded as a put option that has been sold to the investor by the issuer, and so increases the value of the bond when compared with a similar non-putable convertible bond. Zero-coupon bonds have put prices that grow in time at an accretion rate, usually the same as the call accretion rate. On the final date the bond may be redeemed for the principal.

# The Market Variables

The value of a convertible bond depends on the following market variables:

- Current common stock price.  
Stock price volatility: This affects the likelihood of future conversion.  
Dividend yield on the common stock.  
- Riskless rate: This is the rate at which an investor can borrow or lend money with no risk. Traders often approximate this rate by their cost of funds, quoted as a spread over LIBOR.  
Stock loan rate: This is the interest rate earned on funds received from shorting stock. It is usually less than the cost of funds because of a rebate fee that has to be paid to the stock lender.

- Issuer's credit spread: Corporations may default on the payments of interest and/or principal of their bonds. This lowers corporate bond values compared to those of Treasury bonds of the same coupon and maturity, and increases their quoted yield over similar Treasuries by an amount known as the credit spread. The credit spread provides information about the likelihood of default on payments of a convertible bond's coupons and principal, and how this possibility of default affects the value of the convertible. We explain this further in The Credit-Adjusted Discount Rate section on page 16.

# HOW CONVERTIBLES BEHAVE

In this section we use our model to illustrate how the theoretical value of a simple convertible depends upon some market parameters whose base values are listed in Table 1.

TABLE 1.A Simple Convertible Bond  

<table><tr><td>Principal:</td><td>$100</td></tr><tr><td>Coupon:</td><td>6% per yeara</td></tr><tr><td>Maturity:</td><td>2, 5 or 10 years</td></tr><tr><td>Calls:</td><td>none</td></tr><tr><td>Puts:</td><td>none</td></tr><tr><td>Volatility:</td><td>20% per year</td></tr><tr><td>Riskless rate:</td><td>5% per year</td></tr><tr><td>Stock loan rate:</td><td>5% per year</td></tr><tr><td>Dividend yield:</td><td>2% per year</td></tr><tr><td>Credit spread:</td><td>100 basis points</td></tr></table>

a. Bond coupon, stock dividend yield and all interest rates are annually compounded.

Figure 1(a) on page 9 shows that the convertible bond's theoretical value increases with the parity of the underlying stock. The value of a straight bond with a coupon of  $6\%$  in a  $5\%$  interest-rate environment and a credit spread of 100 basis points is  $\$ 100$ . At low market levels with parity of, say,  $\$ 50$ , the convertible is worth little more than this straight bond. As parity increases and conversion becomes more likely, the convertible value rises to reflect the increased value of the conversion privilege. In this graph we show the variation for convertible bonds of maturities 2, 5 and 10 years. At very high market levels the convertible continues to be worth more than parity, because the investor can postpone conversion until maturity without much risk in order to keep collecting the bond coupons. At all market levels, a longer-maturity convertible is worth more.

Figure 1(b) shows the increase in theoretical value with stock volatility. At higher volatilities the convertible value increases, because the value of the option to exchange the straight bond for stock is greater. For all maturities shown, the value of the convertible bond grows approximately linearly with common stock

volatility. This occurs because the conversion option is near-the-money.

Figure 2(a) on page 10 shows how the theoretical value of the convertible declines as interest rates increase. We assume here that both the stock loan rate and the riskless rate rise together when interest rates rise, while the credit spread remains constant. The rate of decline in value when interest rates rise is greater for longer maturity convertibles. Nevertheless, the rate of decline is less for a convertible than for an equivalent straight bond, because of the cushioning effect of the conversion option, which increases in value as interest rates rise.

Figure 2(b) shows how the theoretical value of the convertible declines with increasing credit spread. For fixed values of the stock loan rate and the riskless rate, an increase in the credit spread lowers the present value of any future coupons and principal paid to a convertible investor. Convertible bonds with longer maturity are more sensitive to changes in credit spread.

Figure 3(a) on page 11 illustrates how absolute calls reduce the theoretical value of a convertible. Lower call levels give the issuer a greater probability of being able to force early conversion, thereby diminishing more of the time value of the conversion option, so bonds with lower call levels are worth less. The reduction in value due to calls is more pronounced at high parity levels. Similar results hold for provisional calls.

Figure 3(b) shows the added theoretical value the convertible owner obtains from a put provision. A bond with a higher put level has greater value because of the additional protection puts provide in declining markets. The contribution to the theoretical value from puts is greater at lower parity levels.

Figure 4(a) on page 12 illustrates the effect of calls on the convertible's value over a range of interest rates. The decline in value owing to calls is more pronounced at low interest rates, where the fixed-income value of the convertible is greater.

Figure 4(b) illustrates the effect of puts on the convertible's value over the same range of interest rates. The increase in value owing to puts is more pronounced at high interest rates, where the fixed income value of the convertible is lower.

![](images/d89249c0196143f3c78b8ea94338929f1274c41fde7001a0c9acc894a4cb55bf.jpg)  
FIGURE 1. Variation of the theoretical value of the convertible bond described in Table 1 with (a) parity and (b) volatility.  
(a)

![](images/3b517bd3c8df2b5e52c1197232218db69a6977dd50dc9ef33a957c4c8cd41bef.jpg)  
(b)

![](images/c55b4b2796ed8aba32746dd1620b663f7721734ee941ea303b97e484b67a5241.jpg)  
FIGURE 2. Variation of the theoretical value of the convertible bond described in Table 1 with (a) interest rate and (b) credit spread.  
(a)

![](images/713fdd8a815ae81b1ff55dd200ee0799fd53a08ccdd376fba8c3a057e04124cf.jpg)  
(b)

![](images/c1cf67ab7cd7b24ffad0d3f89b44ebb570e63546ae9b788ebf2d3a844e946654.jpg)  
FIGURE 3. Variation with parity of the theoretical value of the convertible bond described in Table 1 in the presence of (a) absolute call provisions and (b) put provisions. The convertible shown has a maturity of ten years.  
(a)

![](images/d1407ddafae36347ceab575124d51b53edf36f7b6992038791620879bfa7fa6f.jpg)  
(b)

![](images/59933bf93f5fbb2f9251c50854c35a92258cbc149e156096a5623280cbc8eb0a.jpg)  
FIGURE 4. Variation with interest rates of the theoretical value of the convertible bond described in Table 1 in the presence of (a) absolute call provisions and (b) put provisions. The convertible shown has a maturity of ten years.  
(a)

![](images/0a5bc25e2beb298e3b8c5ec83ea3bbcbb65e4980c22a689fbfea6796dbc2bbc3.jpg)  
(b)

# THE MODEL

The owner of a convertible bond has the right to receive future coupon and principal payments, and the option to forgo these payments in exchange for stock, subject to certain call and put provisions as outlined above. We can use the theory of options to value and hedge convertibles.

There are several sources of uncertainty that affect the value of convertibles. In this paper, we simplify things by assuming that the only source of uncertainty is the future price of the underlying stock. We assume that everything else (interest rates and the volatility of the stock, for example) is known with certainty. Stock price uncertainty does seem to capture the major source of options value in most convertibles<sup>3</sup>.

More specifically, we make the following assumptions:

1. The distribution of future stock prices is lognormal with known volatility.  
2. All future interest rates -- the riskless rate, the stock loan rate and the issuer's credit spread -- are known with certainty.  
3. All the information we need about default risk is contained in the credit spread for the issuer's straight bonds. (We will later explain how we use the credit spread in calculating present values of future convertible cash flows.)

With these assumptions, the valuation methods developed by Black and Scholes for ordinary options apply here as well. You can hedge a convertible bond by shorting the underlying stock to create an instantaneously riskless hedge. You can value the bond by calculating its expected value over all future stock price scenarios, provided they are consistent with the known forward prices of the stock and its volatility.

# The Binomial Tree

We follow the Cox-Ross-Rubinstein method of formulating the Black-Scholes equation. We start by building a binomial tree of stock prices in a "risk-neutral" world, where any security has an

expected total return equal to the riskless rate less any rebates for borrowing securities. Each node in the tree represents a possible stock price at a specific time. Figure 4 shows one period of the stock tree extending over one short valuation time step of duration t. The whole tree, of which Figure 4 is merely a part, starts on the valuation date and ends at the maturity of the convertible.

![](images/c59dd20231e11539e4255b69b7dd38d2c16c09c0a76bae037d050d615454bc09.jpg)  
FIGURE 4. One-period stock tree.

The stock starts out at price S. After time t elapses, the stock can move to either Su or Sd with equal probability. The difference between Su and Sd is determined by the volatility of the stock. The mean of Su and Sd is the forward price of the stock after time t.

Once we have the tree of future stock prices, we can build a corresponding tree of future convertible bond prices. We can calculate the value of the convertible at each convertible tree node by starting at maturity, where its value is known with certainty, and then moving backwards in time down the tree, period by period, to calculate the value at earlier nodes. Figure 5 shows the corresponding one-period convertible tree.

![](images/9b2b55997b5573369288577939025714909a38456c71021d228a804adabad069.jpg)  
FIGURE 5. One-period convertible tree.

Let  $V$  be the value of the convertible bond at the start of the period. We can find  $V$  by comparing the choices available to the issuer and the investor, assuming that each behaves in a rational manner, knowing that the only possible convertible values one period in the future are  $V_{u}$  and  $V_{d}$ .

The holding value  $H$  of the convertible at the start of the period in a binomial model is the expected present value of  $V_{u}$  and  $V_{d}$  plus the present value of any convertible coupons paid during this period.  $H$  is the expected value the investor can realize by waiting for one further time period without converting, assuming no provisions are applicable during that time. We list below how to calculate the value  $V$  of the convertible at the current node for all combinations of provisions that may be in effect:

1. No active call or put provisions: The investor can either hold the bond for one more period or convert it to stock. Therefore, he will choose to make  $V$  the maximum of the holding value  $H$  and parity.  
2. The convertible can be put at price  $P$ . The investor can hold, convert, or put the bond for cash equal in value to  $P$  plus accrued interest. He will choose to make  $V$  the maximum of holding value  $H$ , parity, and the put value.  
3. The convertible is both callable at price  $C$  and putable at price  $P$ . The issuer will call the bond when the call value (defined as  $C$  plus the accrued interest) is less than the holding value  $H$ . If the bond is called, the investor can still choose whether to put

the bond for the put value, convert it to stock, or accept the issuer's call. V is the maximum of three values: parity, put value, and the minimum of holding value and call value<sup>7</sup>.

Note that put and call provisions allow the investor to receive accrued interest. An investor who converts will forfeit the accrued interest.

The Credit-Adjusted Discount Rate

The holding value  $H$  of the convertible at a node in the tree is the sum of the present values of the coupons paid over the next period, plus the expected present value of the convertible at the two nodes at the end of the period. What discount rate should we use to calculate these present values?

For an ordinary option that exercises into stock, the appropriate discount rate is the riskless rate  $r$ , because hedging the option with stock results in a riskless investment over a short time period. But convertible bonds pay coupons and return principal, which, unlike stock, are both subject to default. So, the riskless rate is not entirely appropriate for discounting these payoffs.

First let's look at the two extremes of high and low stock prices. If, at the next node, the stock is far above the conversion price, the conversion option will be deep-in-the-money and is certain to be exercised. Then the appropriate discount rate is the riskless rate  $r$ , because the investor is certain to obtain stock with no default risk.

Alternatively, if the stock price at the next node is far below the conversion price, the conversion option will be deep-out-of-themoney and will certainly not be exercised. In that case, the investor owns a corporate-grade fixed-income instrument that will continue to pay coupons and principal. The appropriate discount rate,  $d$ , is the "risky rate" obtained by adding the issuer's credit spread to the riskless rate.

So, at high stock prices, where eventual conversion is virtually guaranteed, the appropriate discount rate is  $r$ ; at low stock prices, where eventual conversion is overwhelmingly unlikely, the appropriate rate is  $d$ . At intermediate stock prices, we intend to use the credit-adjusted discount rate,  $y$ , which we define in the next paragraph, and which always lies between  $r$  and  $d$ .

Let  $p$  be the probability at a given node that the convertible will convert to stock in the future. Then  $(1 - p)$  is the probability that it will remain a fixed-income bond. You can think of the convertible at any node as being a weighted mixture of default-free stock and a default-prone corporate bond, with  $p$  specifying the weighting factor. We want the credit-adjusted discount rate to reflect the proportion of riskless and risky assets contained in the convertible. So, we define  $y$  as the weighted mixture of the riskless and risky rates, where the weighting factor is  $p -$  that is,  $y = p \times r + (1 - p) \times d$ .

The value of  $p$  is easy to find on a binomial tree; we explain how below. This credit-adjusted rate  $y$  is equal to  $r$  when  $p = l$  and conversion is certain. It is equal to  $d$  when  $p = 0$  and conversion is impossible. The effect of using  $y$  for discounting is to assign a credit spread to the convertible that moves smoothly between zero and the issuer's credit spread, depending upon how likely it is that the convertible ultimately converts.

# Model Summary

Here is a short summary of the steps we follow to construct our binomial model for valuing convertible bonds.

1. Build a Cox-Ross-Rubinstein stock price tree that extends from the valuation date to the maturity of the convertible. In building the tree, ensure that at each time level the stock has the appropriate assumed volatility, and that the average stock price matches the stock's forward price.  
2. At maturity, compute the value of the convertible bond as the greater of its fixed-income redemption value and its conversion value. Define the probability of conversion to be one at nodes where it pays to convert, and zero otherwise.  
3. Move backwards in time down the tree, one level at a time. At each node within each level, define the conversion probability as the average of the probabilities at the two connected future nodes. Compute the credit-adjusted discount rate at each node

using this conversion probability. Then compute the holding value at each node as the sum of the cash flows occurring over the next period and the expected bond values of the two nodes one period in the future, discounted at the credit-adjusted discount rate.

4. Compute the actual convertible value by comparing the holding value at the node to the values of the call, put and conversion provisions.  
5. If the value of the convertible at any node results from the bond being put, set the conversion probability at that node to zero, because its value is fully subject to default. If the value at the node results from conversion, set the node's conversion probability to one.

# Some Subtleties

Convertibles are complex securities. They incorporate within one instrument many features at the boundary of current options theory: how to value long-term options, how to model the yield curve and how to model credit risk. Our model is an attempt to incorporate some of the convertible security's most important properties. Several consequences may surprise you, and so are worth keeping in mind when you use it:

- A convertible with parity much greater than its face value, and therefore certain to convert at some time in the future, has a credit-adjusted discount rate equal to the riskless rate. In this case, our model discounts the coupons paid until conversion at the riskless rate, as though they have no default risk. It is possible to develop variants of our model in which these coupons are nevertheless discounted at the "risky" rate.  
- An at-the-money convertible from an issuer with a sizeable credit spread can decrease in value when volatility increases. In options parlance, it can have negative vega. This nonintuitive result occurs because an increase in volatility has two effects. First, it makes the expected payoff from conversion greater. Second, it increases the probability of default, which lowers the convertible's value. Sometimes the latter effect outweighs the former.

Looking deeper, there are further complexities in the modeling of convertibles which we briefly mention.

- A convertible bond is a derivative claim on the company's stock. The stock itself can be regarded as a derivative claim on a company's assets<sup>4</sup>. From this point of view, the convertible is really a compound derivative claim on the assets, and it is natural to assume that the distribution of future asset values is lognormal. Then, the distribution of future stock prices can not be lognormal, contrary to the assumptions of our model.  
Corporations may default on the future payments of interest and return of principal of their bonds. Therefore, the value of a convertible depends upon the current and future creditworthiness of the issuer. So, you can think of a convertible as a credit derivative. The value of this derivative depends upon the volatility of the changes in the credit spread. In the present model we have assumed the credit spread remains constant throughout the convertible's lifetime.

It is quite feasible to extend or modify our model to account for these features. But the resultant model would take longer to run on a computer, and it would require more input data and assumptions about a company's financial structure and credit risk. The model we present in this paper disregards these complexities in the interest of having a relatively straightforward model that can be used with easily available security prices, interest rates and credit ratings.

# AN EXAMPLE

We now illustrate how to use the model by valuing a hypothetical convertible bond, described in Table 2. We choose a large credit spread of 500 basis points to make its effect on the credit-adjusted discount rate easily observable. In order to display the full tree from valuation date to maturity, we choose the time steps between periods to be one year in length. Although this makes the distribution of stock prices unrealistically coarse-grained, it does make drawing the tree easy. The computer version of the model uses much smaller steps to get greater numerical accuracy.

TABLE 2. A Hypothetical Convertible Bond  

<table><tr><td>Principal:</td><td>$100</td></tr><tr><td>Coupon:</td><td>10% per yeara</td></tr><tr><td>Maturity:</td><td>5 years</td></tr><tr><td>Conversion ratio:</td><td>1</td></tr><tr><td>Calls:</td><td>$115 in year 2, declining by $5 every year to maturity</td></tr><tr><td>Puts:</td><td>put at $120 in year 3</td></tr><tr><td>Current stock price:</td><td>$100</td></tr><tr><td>Stock dividend rate:</td><td>0%</td></tr><tr><td>Volatility:</td><td>10% per year</td></tr><tr><td>Riskless rate:</td><td>5% per year</td></tr><tr><td>Stock loan rate:</td><td>5% per year</td></tr><tr><td>Credit spread:</td><td>500 basis points</td></tr></table>

a. Bond coupon, stock dividend yield and all interest rates are annually compounded.

Figure 6 shows the stock and convertible trees with one-year time steps. At each node on the stock tree we show the stock price. At each node on the convertible tree we show the theoretical convertible value, together with a letter code to indicate the action that has been taken at that node. Table 3 explains the letter codes. At each node, we also show the conversion probability used to calculate the credit-adjusted discount rate.

# TABLE 3. Letter Codes for Figure 6

X: investor converts to stock  
P: investor exerci es put  
C: Issuer exercises call  
H: investor holds convertible for one more period  
R: Issuer redeemems convertible at maturity

Let's look at some specific nodes and show how we obtain the values associated with them on the trees. At the start, the stock has a value of 100. After one year, it moves up to 115.47 or down to 94.53. These are the moves that correspond to a one-year return volatility of 10% and an average price of $105, exactly the forward price of the stock after one year.

Now look at the convertible bond tree. At maturity in year 5, the convertible bond pays out 110 (principal plus accrued interest) if the investor does not convert into stock. Therefore, at each node, the bond is worth the maximum of the stock price at the corresponding node on the stock tree and 110. At those nodes where the maximum is the stock price, the conversion probability is 1.0, the node's letter code is X to indicate conversion, and the credit-adjusted discount rate is the riskless rate (5%). At those nodes where the maximum is 110, the conversion probability is 0.0, the letter code is R for redemption, and the credit-adjusted rate is the riskless rate (5%) plus the credit spread (500 b.p.), or 10%.

Now look at the stock node with the lowest price (79.87) in year 4. The corresponding convertible node in year 4 can evolve into up- and down-nodes at maturity that are each worth 110 and carry code R with a conversion probability of 0.0. The credit-adjusted discount rate at each of these nodes is  $10\%$ . Therefore, the expected present values over these two nodes is  $(0.5)(110 / 1.1) + (0.5)(110 / 1.1) = 100$ . You get the holding value of the convertible at the lowest stock node in year 4 by adding the coupon worth 10 that is paid at the start of the year, to give a holding value of 110.

![](images/5b0fb125a749470acf726c0174aab182b33dec2a419e3c086d994c1e08558098.jpg)  
FIGURE 6. Binomial Trees for Valuing the Convertible Bond

![](images/ed2e204066eb1c695762bb80cfadad8c5043fcc531402e2718a28db1ecb72000.jpg)

![](images/a1dee652e4394afd414ecc5bb4f760f9532508b8f366d9b58207151c4e75f925.jpg)

Since the call value (including accrued interest) in year 4 is 115, the issuer will not call the bond. There is no applicable put in year 4. Converting to stock yields a value of 79.87. Therefore, the investor maximizes his value by holding the convertible, and its theoretical value at this node is 110, with letter code H. The conversion probability at this node, at which the bond is neither converted, put nor called, is the average of its values at the up- and down-nodes, namely 1.00. The corresponding credit-adjusted discount rate is  $10\%$ .

As a final example, look at the stock node with price 103.19 in year 3. On the convertible tree, the holding value at the corresponding node is given by the sum of (1) the coupon paid at the start of year 3, and (2) the expected value of the convertible at the two connected up- and down-nodes in year 4, each discounted at the credit-adjusted rate. The up-node in year 4 has a value of 119.15, with a conversion probability of 1.0. The corresponding credit-adjusted rate for discounting the up-value is  $5\%$ . The present value of 119.15 at this rate is 113.48. Similarly, the down-node's conversion probability is 0.5. Therefore, the credit-adjusted rate at the up-node is  $(0.5)(5\%) + (0.5)(10\%) = 7.50\%$ . The present value of 113.64 at this rate is 105.71. The expected present value of the up- and down-nodes is 109.60. The holding value includes a coupon worth 10, to give a total of 119.60. However, the investor has the right to put the bond for the put price ( $120) plus accrued interest ($ 10), a total of $130. Since this is worth more than holding the bond for one more year, the investor will exercise the put, and the value of the convertible at this node is $130. The conversion probability here is reset to 0.0 because the convertible was put, and the credit-adjusted rate is  $10.00\%$ .

We can repeat this procedure at all nodes in the tree, working from maturity to the present, to compute the value of the convertible in year 0 when the stock price is 100.

# USING THE MODEL TO HEDGE

A convertibles trader finances the purchase of the convertible by borrowing at his cost of funds.

To hedge the convertible's stock price risk, he maintains a short position of  $\Delta$  shares of the underlying stock, where  $\Delta$  is the model's value for the sensitivity of the convertible's value to stock price changes. The funds from the short stock position are maintained in an escrow account where they earn the stock loan rate. The trader now owns a portfolio consisting of a long position in the convertible bond, and a short position in stock. In theory, this portfolio is hedged against small stock price moves. Any income generated by convertible coupons and the short stock position after paying for the convertible's financing is used to reduce the amount of borrowed capital.

The trader can also use the model to hedge against changes in interest rates by computing the value of the convertible's sensitivity to interest-rate changes. There are three independent types of interest rates in the model: the riskless rate, the stock loan rate, and the credit spread.

The riskless rate and the stock loan rate usually move more or less in tandem. They can both be hedged using Treasury bonds of maturity comparable to that of the convertible. For callable convertibles, it makes more sense to use Treasuries with maturities near the first call date.

The issuer's credit spread can vary independently of Treasury rates. If there are outstanding straight bonds of the issuer with similar terms, they will have sensitivity to both the Treasury rates and the credit spread. A trader can then combine them with Treasury bonds to offset the risks due to changes in both types of rates. Unfortunately, such straight bonds rarely exist, because companies usually issue convertibles as a substitute for straight debt. In that case, hedging exposure to credit risk is difficult.

# THE CALL ADJUSTMENT FACTOR

Issuers usually call a convertible bond to force the investor to convert. Most call provisions give the investor 30 days to decide whether to convert after a call. If the stock price were to fall appreciably during that time, investors might decline to convert and the issuer would have to redeem the bond for cash. Therefore, issuers tend to call the bond only when the stock price trades at a cushion far enough above the conversion price or call level such that a move below it is very unlikely over the next 30 days.

Our model contains a call adjustment factor to handle this. Users can specify an empirical factor that determines the size of the cushion below which the issuer will not call the bond, even though he is legally entitled to do so.

Using the call adjustment factor increases the theoretical value of the convertible bond for the investor, since it delays the point at which issuers exercise their option to call the bond.

APPENDIX: THE MATHEMATICS OF THE MODEL

This section contains a brief mathematical account of the model.

The underlying stock price satisfies the stochastic differential equation

$$
\frac {d S}{S} = \left(l _ {t} - d _ {t}\right) d t + \sigma_ {t} d W \tag {EQ1}
$$

Here  $l_{t}$  is the instantaneous stock loan rate at time  $t$ ,  $d_{t}$  is the instantaneous stock dividend yield, and  $\sigma_{t}$  is the instantaneous volatility.  $dW$  is a random shock that represents the uncertainty in stock returns over the infinitesimal time  $dt$ , and is distributed normally with mean 0 and variance  $dt$ . All rates are based on continuous compounding.

The binomial tree is a discrete-time version of this process, with time steps of  $\Delta t$ ; it converges to it in the limit of zero time between tree levels. We choose to have equal probabilities 0.5 of moving up or down at any node. The formulas for generating up and down values from a given stock price  $S$  are

$$
S _ {u} = S \exp \left(\left(l _ {t} - d _ {t} - \frac {1}{2} \sigma_ {t} ^ {2}\right) \Delta t + \sigma_ {t} \sqrt {\Delta t}\right) \tag {EQ2}
$$

$$
S _ {u} = S \exp \left(\left(l _ {t} - d _ {t} - \frac {1}{2} \sigma_ {t} ^ {2}\right) \Delta t - \sigma_ {t} \sqrt {\Delta t}\right)
$$

With these values, the expected stock price during time  $\Delta t$  grows at the rate  $l_{t} - d_{t}$ .

The value of the convertible bond at any node on the convertible tree is given by

$$
V = \max  (n S, P + a, \min  (H, C + a)) \tag {EQ3}
$$

where  $n$  is the conversion ratio,  $S$  is the stock price at the node,  $P$  is the put value,  $C$  is the call value,  $a$  is the accrued interest and  $H$  is the holding value at the node. This holding value is computed from the values  $V_{u}$  and  $V_{d}$  one period later as

$$
H = (0. 5) \left(\frac {V _ {u}}{1 + y _ {u} \Delta t} + \frac {V _ {d}}{1 + y _ {d} \Delta t}\right) \tag {EQ4}
$$

Finally, we explain how to calculate the credit-adjusted discount rate  $y$  at a node. First, the probability of conversion  $p$  at the node is determined in the following manner:

If the convertible is put or redeemed,  $p = 0$ .

If it converts to stock,  $p = 1$ .

If it is neither put nor converted,  $p = 0.5(p_u + p_d)$ , where  $p_u$  and  $p_d$  are the conversion probabilities one period later.

The credit-adjusted discount rate at the node is defined by the mixing equation:

$$
y = p \times r + (1 - p) \times d \tag {EQ5}
$$

where  $r$  is the riskless rate and  $d$  is the risky rate that includes the credit spread.