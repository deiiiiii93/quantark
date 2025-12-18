# OVCV Model Description

David Frank

Quantitative Research and Development, Equities Team

March 27, 2009

# Introduction

In this document, we describe the Jump-Diffusion model used for convertible bonds in the function OVCV. We first describe the process followed by the stock price under this model. We then derive the partial differential equation (PDE) whose solution gives the convertible bond price as a function of stock price and time under this model.

# The Stock Process Under the Jump-Diffusion Model

We model convertible bond prices using a Jump-Diffusion model. In this section, we describe the stock dynamics followed under this model.

The convertible bond is priced using a one factor model. We assume the stock price follows the usual Black-Scholes, lognormal stock process with time-dependent rates and volatilities, with the addition of an independent Poisson process to model defaults. The following list describes notation for use in this and the following section:

$B_{t}$  Bond price, including accrued interest (dirty price)

$F$  Face value (par value) of the bond

$S_{t}$  Stock price at time  $t$

Time-dependent interest rate

q Time-dependent dividend rate

$R$  The recovery rate

K Time-dependent conversion ratio

D The value of the convertible bond after default, including recovery

$h$  Time-dependent hazard rate

$\eta$  Fractional loss in the stock price on default (set to .40 in OVCV)

$W_{t}$  A standard Brownian motion

$U_{t}$  Aoisson process, independent of  $W_{t}$

The lognormal stock process can be described as:

$$
d S _ {t} = \left[ r (t) - q (t) \right] S _ {t} d t + \sigma (t) S _ {t} d W _ {t}
$$

The addition to these dynamics of a jump in the stock price on default leads to the following dynamics:

$$
d S _ {t} = \left[ r (t) - q (t) + \eta h (t) \right] S _ {t} ^ {-} d t + \sigma (t) S _ {t} ^ {-} d W _ {t} - \eta S _ {t} ^ {-} d U _ {t} \tag {1}
$$

where  $U$  is an independent Poisson process used to model defaults, with  $dU_{t} = 1$

with probability  $h(t)dt$ , and 0 otherwise. The notation  $S_{t}^{-}$  is used to denote the stock price immediately before any jump at time  $t$ . The parameter  $h(t)$  is known as the hazard rate. The hazard rate function is calibrated to Credit Default Swap data if such data is available for the bond issuer. On default, the stock price is assumed to jump downward by exactly the fraction  $\eta$  of its pre-default value. In the OVCV model,  $\eta$  is set to .40.

As with the Black-Scholes model for stock options, the model described above leads via the usual arbitrage arguments to a PDE for the convert price. The actual solution method used is to solve that PDE over a grid in the two dimensions of stock price and time, with boundary conditions appropriate to the convertible bond's conversion, call, and put provisions. We derive the PDE in the next section.

# Derivation of the Convertible Bond PDE

In this section, we derive the partial differential equation which holds for the price of a convertible bond with default risk under our model. Henceforth we employ subscripts on the variable  $B$  to denote partial derivatives of the convertible bond price.

First, consider the case of the convertible bond without default, that is, where  $h(t)$  is zero in equation (1). If we form a portfolio II consisting of one convertible bond and  $-\beta$  shares of the stock, then by Ito's Formula we arrive at the following PDE for changes in the value of this portfolio:

$$
d \Pi = \left[ B _ {T} + \frac {1}{2} \sigma (t) ^ {2} S ^ {2} B _ {S S} \right] d t + B _ {S} d S - \beta (d S + q (t) S d t)
$$

Using the standard Black-Scholes argument, we can eliminate risk from the portfolio by choosing  $\beta = B_{S}$ , in which case the portfolio must grow at the risk free rate. We can thus derive a PDE for the bond price under the no-default assumption.

With the addition of the risk of default, we arrive at the Jump-Diffusion model. We assume that the probability of default in the interval  $[t, t + dt]$  is  $h(t)dt$ , and that after default, the bond value falls to some value  $D$ , a function of  $R$  and other factors (we provide the exact form of  $D$  later). Then, assuming default risk is fully diversifiable, the risk neutral default probabilities and real world probabilities will be equal, and there is no excess expected return above the risk free rate earned for holding credit risk. We can then form a portfolio  $\Pi$  as above but now including one risky convertible bond and  $-\beta$  shares of the stock. The change in value of this portfolio is given by

$$
\begin{array}{l} {d \Pi} = {[ 1 - d U _ {t} ] \left[ \left(B _ {T} + \frac {1}{2} \sigma (t) ^ {2} S ^ {2} B _ {S S}\right) d t + B _ {S} d S \right]} \\ - \quad [ 1 - d U _ {t} ] \left[ \beta d S + \beta q (t) S d t \right] \\ + \quad d U _ {t} \quad [ (D - B) + \eta S \beta ] \\ \end{array}
$$

where the first line contains terms that represent the change in the value of the bond if there is no default during the period  $dt$ , the second line has terms that represent the change in value of the short stock position if there is no default, and the third line is the change in value of the bond and short stock when there is a default.

If we now eliminate the stock risk from the portfolio by again choosing  $\beta = B_{S}$ , and take expectations with respect to the risk neutral measure we find

$$
\begin{array}{l} E [ d \Pi ] = [ 1 - h (t) d t ] \left[ \left(B _ {T} + \frac {1}{2} \sigma (t) ^ {2} S ^ {2} B _ {S S}\right) d t + B _ {S} d S \right] \\ - \quad [ 1 - h (t) d t ] \left[ B _ {S} d S + B _ {S} q (t) S d t \right] \\ + \quad h (t) d t \quad [ (D - B) + \eta S B _ {S} ] \\ \end{array}
$$

Now by eliminating terms of order higher than  $dt$  and by dropping the canceling  $dS$  terms, the equation reduces to

$$
E [ d \Pi ] = \left[ B _ {T} + \frac {1}{2} \sigma (t) ^ {2} S ^ {2} B _ {S S} \right] d t - B _ {S} q (t) S d t + h (t) \left[ D - B + \eta S B _ {S} \right] d t \tag {2}
$$

The assumed diversifiability of credit risk implies that the expected return on the portfolio is again the risk free rate:

$$
r (t) \Pi d t = E [ d \Pi ], \quad \mathrm {w h e r e} \quad \Pi = B - B _ {S} S
$$

This last equation combined with (2) gives us the PDE which we solve to price the convertible bond under Jump-Diffusion. Direct substitution into (2) gives:

$$
r (t) \left(B - B _ {S} S\right) d t = \left[ B _ {T} + \frac {1}{2} \sigma (t) ^ {2} S ^ {2} B _ {S S} \right] d t - B _ {S} q (t) S d t + h (t) \left[ D - B + \eta S B _ {S} \right] d t
$$

By dividing out  $dt$  and simplifying, we find

$$
\left[ r (t) + h (t) \right] B = B _ {T} + \frac {1}{2} \sigma (t) ^ {2} S ^ {2} B _ {S S} + \left[ r (t) - q (t) + \eta h (t) \right] B _ {S} S + h (t) D
$$

Assuming the value on default  $D$  is the maximum of the recovery value on the bond, and the remaining post-default conversion value, this leads to the final PDE:

$$
\left[ r (t) + h (t) \right] B = B _ {T} + \frac {1}{2} \sigma (t) ^ {2} S ^ {2} B _ {S S} + \left[ r (t) - q (t) + \eta h (t) \right] B _ {S} S + h (t) \max \left[ R F, K (t) S (1 - \eta) \right]
$$

We solve the above equation with further modifications to handle discrete dividends and coupon payments. Further, the convertible bond may have time-varying put, call, and conversion features. These are modeled as constraints which are enforced when the various features are in effect.