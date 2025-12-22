"""
Created on Tue May 24 2023

@author: yaofuxin
"""
import numpy as np
from scipy.stats import norm
from EuropeanOption import EuropeanVanillaOption
from DigitalOption import DigitalOption
from OneTouch import OneTouch


class KnockOutOption:
    # signle barrier knock out option
    def __init__(
        self,
        initial,
        rrf,
        div,
        vol,
        K,
        H,
        spot,
        cal_days,
        bus_days,
        disc_mode,
        N,
        rebate,
        is_call,
        up_or_do,
        tenor=0,
        obs_type="Daily",
        participation_rate=1.0,
        bus_days_in_year=245,
        pay_type="Expiry",
    ) -> None:
        self.initial = initial
        self.rrf = rrf
        self.div = div
        self.b = self.rrf - self.div
        self.vol = vol
        self.K = K
        self.H = H
        self.shifted_barrier = self.H
        self.spot = spot
        self.cal_days = cal_days
        self.bus_days = bus_days
        self.T = cal_days / 365
        self.disc_mode = disc_mode
        self.bus_days_in_year = bus_days_in_year
        self.tau = self.bus_days / self.bus_days_in_year
        self.df = (
            np.exp(-1.0 * self.rrf * self.tau)
            if self.disc_mode == "BD"
            else np.exp(-1.0 * self.rrf * self.T)
        )
        self.N = N
        self.rebate = rebate
        self.is_call = is_call
        self.up_or_do = up_or_do
        self.obs_type = obs_type
        self.participation_rate = participation_rate
        self.pay_type = pay_type
        self.tenor = cal_days if tenor <= 0 else tenor
        self.value = 0.0
        self.delta = 0.0
        self.delta_cash = 0.0
        self.gamma = 0.0
        self.gamma_cash = 0.0
        self.vega = 0.0
        self.theta = 0.0
        self.rho = 0.0
        self.rhoQ = 0.0

    def __A__(self, phi):
        return phi * self.spot * np.exp((self.b - self.rrf) * self.tau) * norm.cdf(
            phi * self.x1, loc=0, scale=1
        ) - phi * self.K * np.exp(-1.0 * self.rrf * self.tau) * norm.cdf(
            phi * self.x1 - phi * self.vol * np.sqrt(self.tau), loc=0, scale=1
        )

    def __B__(self, phi):
        return phi * self.spot * np.exp((self.b - self.rrf) * self.tau) * norm.cdf(
            phi * self.x2, loc=0, scale=1
        ) - phi * self.K * np.exp(-1.0 * self.rrf * self.tau) * norm.cdf(
            phi * self.x2 - phi * self.vol * np.sqrt(self.tau), loc=0, scale=1
        )

    def __C__(self, phi, yita):
        return phi * self.spot * np.exp((self.b - self.rrf) * self.tau) * np.power(
            self.shifted_barrier / self.spot, 2 * (self.mu + 1)
        ) * norm.cdf(yita * self.y1, loc=0, scale=1) - phi * self.K * np.exp(
            -1.0 * self.rrf * self.tau
        ) * np.power(
            self.shifted_barrier / self.spot, 2 * self.mu
        ) * norm.cdf(
            yita * self.y1 - yita * self.vol * np.sqrt(self.tau), loc=0, scale=1
        )

    def __D__(self, phi, yita):
        return phi * self.spot * np.exp((self.b - self.rrf) * self.tau) * np.power(
            self.shifted_barrier / self.spot, 2 * (self.mu + 1)
        ) * norm.cdf(yita * self.y2, loc=0, scale=1) - phi * self.K * np.exp(
            -1.0 * self.rrf * self.tau
        ) * np.power(
            self.shifted_barrier / self.spot, 2 * self.mu
        ) * norm.cdf(
            yita * self.y2 - yita * self.vol * np.sqrt(self.tau), loc=0, scale=1
        )

    def __cal_factors__(self):
        self.mu = (self.b - self.vol**2 / 2) / self.vol**2
        self.lamda = np.sqrt(self.mu**2 + 2 * self.rrf / self.vol**2)
        self.x1 = np.log(self.spot / self.K) / self.vol / np.sqrt(self.tau) + (
            1 + self.mu
        ) * self.vol * np.sqrt(self.tau)
        self.x2 = np.log(self.spot / self.shifted_barrier) / self.vol / np.sqrt(
            self.tau
        ) + (1 + self.mu) * self.vol * np.sqrt(self.tau)
        self.y1 = np.log(
            self.shifted_barrier**2 / self.spot / self.K
        ) / self.vol / np.sqrt(self.tau) + (1 + self.mu) * self.vol * np.sqrt(self.tau)
        self.y2 = np.log(self.shifted_barrier / self.spot) / self.vol / np.sqrt(
            self.tau
        ) + (1 + self.mu) * self.vol * np.sqrt(self.tau)
        self.z = np.log(self.shifted_barrier / self.spot) / self.vol / np.sqrt(
            self.tau
        ) + self.lamda * self.vol * np.sqrt(self.tau)

    def __barrier_shift__(self):
        beta = 0.5825971579390107
        dt = 1.0 / self.bus_days_in_year
        if self.obs_type.upper() == "DAILY":
            # if self.is_call and self.up_or_do.upper() == "UP" and self.K < self.H:
            if self.up_or_do.upper() == "UP":
                self.shifted_barrier = self.H * np.exp(beta * self.vol * np.sqrt(dt))
            if (
                self.up_or_do.upper() == "DO"
                or self.up_or_do.upper() == "DOWN"
                # and not self.is_call
                # and self.K > self.H
            ):
                self.shifted_barrier = self.H * np.exp(
                    -1.0 * beta * self.vol * np.sqrt(dt)
                )

    def price_bsm(self):
        if self.obs_type.upper() == "DAILY":
            self.__barrier_shift__()
            self.__cal_factors__()
            if self.is_call and self.up_or_do.upper() == "UP":
                if self.H > self.K:
                    self.value = (
                        (
                            self.__A__(1.0)
                            - self.__B__(1.0)
                            + self.__C__(1.0, -1.0)
                            - self.__D__(1.0, -1.0)
                        )
                        * self.N
                        / 365
                        * self.tenor
                    )
                else:
                    self.value = 0.0

            if self.is_call and (
                self.up_or_do.upper() == "DO" or self.up_or_do.upper() == "DOWN"
            ):
                if self.H < self.K:
                    self.value = (
                        (self.__A__(1.0) - self.__C__(1.0, 1.0))
                        * self.N
                        / 365
                        * self.tenor
                    )
                else:
                    self.value = (
                        (self.__B__(1.0) - self.__D__(1.0, 1.0))
                        * self.N
                        / 365
                        * self.tenor
                    )

            if not self.is_call and self.up_or_do.upper() == "UP":
                if self.H > self.K:
                    self.value = (
                        (self.__A__(-1.0) - self.__C__(-1.0, -1.0))
                        * self.N
                        / 365
                        * self.tenor
                    )
                else:
                    self.value = (
                        (self.__B__(-1.0) - self.__D__(-1.0, -1.0))
                        * self.N
                        / 365
                        * self.tenor
                    )

            if not self.is_call and (
                self.up_or_do.upper() == "DO" or self.up_or_do.upper() == "DOWN"
            ):
                if self.H < self.K:
                    self.value = (
                        (
                            self.__A__(-1.0)
                            - self.__B__(-1.0)
                            + self.__C__(-1.0, 1.0)
                            - self.__D__(-1.0, 1.0)
                        )
                        * self.N
                        / 365
                        * self.tenor
                    )
                else:
                    self.value = 0.0

            self.value = self.value * self.participation_rate

            if np.abs(self.rebate) > 0:
                ot_call = True if self.up_or_do.upper() == "UP" else False
                ot = OneTouch(
                    self.initial,
                    self.rrf,
                    self.div,
                    self.vol,
                    self.H,
                    self.spot,
                    self.cal_days,
                    self.bus_days,
                    self.disc_mode,
                    self.rebate,
                    self.N,
                    ot_call,
                    self.tenor,
                    self.obs_type,
                    pay_type=self.pay_type,
                    bus_days_in_year=self.bus_days_in_year,
                )
                ot.price_bsm()
                self.value += ot.value

        if self.obs_type.upper() == "EXPIRY":
            if self.is_call and self.up_or_do.upper() == "UP":
                if self.H > self.K:
                    eu_1 = EuropeanVanillaOption(
                        self.rrf,
                        self.div,
                        self.vol,
                        self.K,
                        self.spot,
                        self.cal_days,
                        self.bus_days,
                        self.disc_mode,
                        self.N,
                        True,
                        self.tenor,
                        bus_days_in_year=self.bus_days_in_year,
                    )
                    eu_2 = EuropeanVanillaOption(
                        self.rrf,
                        self.div,
                        self.vol,
                        self.H,
                        self.spot,
                        self.cal_days,
                        self.bus_days,
                        self.disc_mode,
                        self.N,
                        True,
                        self.tenor,
                        bus_days_in_year=self.bus_days_in_year,
                    )
                    di_pay_off = (
                        ((self.H - self.K) / self.initial - self.rebate)
                        / 365
                        * self.tenor
                        * self.initial
                        / self.participation_rate
                    )
                    di_1 = DigitalOption(
                        self.rrf,
                        self.div,
                        self.vol,
                        self.H,
                        self.spot,
                        self.cal_days,
                        self.bus_days,
                        self.disc_mode,
                        di_pay_off,
                        self.N,
                        True,
                        bus_days_in_year=self.bus_days_in_year,
                    )
                    eu_1.price_bsm()
                    eu_2.price_bsm()
                    di_1.price_bsm()
                    self.value = (
                        eu_1.value - eu_2.value - di_1.value
                    ) * self.participation_rate
                else:
                    di_pay_off = self.rebate / 365 * self.tenor * self.initial
                    di_1 = DigitalOption(
                        self.rrf,
                        self.div,
                        self.vol,
                        self.H,
                        self.spot,
                        self.cal_days,
                        self.bus_days,
                        self.disc_mode,
                        di_pay_off,
                        self.N,
                        True,
                        bus_days_in_year=self.bus_days_in_year,
                    )
                    di_1.price_bsm()
                    self.value = di_1.value
            if self.is_call and (
                self.up_or_do.upper() == "DO" or self.up_or_do.upper() == "DOWN"
            ):
                if self.H > self.K:
                    eu_1 = EuropeanVanillaOption(
                        self.rrf,
                        self.div,
                        self.vol,
                        self.H,
                        self.spot,
                        self.cal_days,
                        self.bus_days,
                        self.disc_mode,
                        self.N,
                        True,
                        self.tenor,
                        bus_days_in_year=self.bus_days_in_year,
                    )
                    di_pay_off_1 = (
                        ((self.H - self.K) / self.initial)
                        / 365
                        * self.tenor
                        * self.initial
                    )
                    di_1 = DigitalOption(
                        self.rrf,
                        self.div,
                        self.vol,
                        self.H,
                        self.spot,
                        self.cal_days,
                        self.bus_days,
                        self.disc_mode,
                        di_pay_off_1,
                        self.N,
                        True,
                        bus_days_in_year=self.bus_days_in_year,
                    )
                    di_pay_off_2 = self.rebate / 365 * self.tenor * self.initial
                    di_2 = DigitalOption(
                        self.rrf,
                        self.div,
                        self.vol,
                        self.H,
                        self.spot,
                        self.cal_days,
                        self.bus_days,
                        self.disc_mode,
                        di_pay_off_2,
                        self.N,
                        False,
                        bus_days_in_year=self.bus_days_in_year,
                    )
                    eu_1.price_bsm()
                    di_1.price_bsm()
                    di_2.price_bsm()
                    self.value = (
                        eu_1.value + di_1.value
                    ) * self.participation_rate + di_2.value
                else:
                    eu_1 = EuropeanVanillaOption(
                        self.rrf,
                        self.div,
                        self.vol,
                        self.K,
                        self.spot,
                        self.cal_days,
                        self.bus_days,
                        self.disc_mode,
                        self.N,
                        True,
                        self.tenor,
                        bus_days_in_year=self.bus_days_in_year,
                    )
                    di_pay_off = self.rebate / 365 * self.tenor * self.initial
                    di_1 = DigitalOption(
                        self.rrf,
                        self.div,
                        self.vol,
                        self.H,
                        self.spot,
                        self.cal_days,
                        self.bus_days,
                        self.disc_mode,
                        di_pay_off,
                        self.N,
                        False,
                        bus_days_in_year=self.bus_days_in_year,
                    )
                    eu_1.price_bsm()
                    di_1.price_bsm()
                    self.value = eu_1.value * self.participation_rate + di_1.value
            if not self.is_call and self.up_or_do.upper() == "UP":
                if self.H < self.K:
                    eu_1 = EuropeanVanillaOption(
                        self.rrf,
                        self.div,
                        self.vol,
                        self.H,
                        self.spot,
                        self.cal_days,
                        self.bus_days,
                        self.disc_mode,
                        self.N,
                        False,
                        self.tenor,
                        bus_days_in_year=self.bus_days_in_year,
                    )
                    di_pay_off_1 = (
                        ((self.K - self.H) / self.initial)
                        / 365
                        * self.tenor
                        * self.initial
                    )
                    di_1 = DigitalOption(
                        self.rrf,
                        self.div,
                        self.vol,
                        self.H,
                        self.spot,
                        self.cal_days,
                        self.bus_days,
                        self.disc_mode,
                        di_pay_off_1,
                        self.N,
                        False,
                        bus_days_in_year=self.bus_days_in_year,
                    )
                    di_pay_off_2 = self.rebate / 365 * self.tenor * self.initial
                    di_2 = DigitalOption(
                        self.rrf,
                        self.div,
                        self.vol,
                        self.H,
                        self.spot,
                        self.cal_days,
                        self.bus_days,
                        self.disc_mode,
                        di_pay_off_2,
                        self.N,
                        True,
                        bus_days_in_year=self.bus_days_in_year,
                    )
                    eu_1.price_bsm()
                    di_1.price_bsm()
                    di_2.price_bsm()
                    self.value = (
                        eu_1.value + di_1.value
                    ) * self.participation_rate + di_2.value
                else:
                    eu_1 = EuropeanVanillaOption(
                        self.rrf,
                        self.div,
                        self.vol,
                        self.K,
                        self.spot,
                        self.cal_days,
                        self.bus_days,
                        self.disc_mode,
                        self.N,
                        False,
                        self.tenor,
                        bus_days_in_year=self.bus_days_in_year,
                    )
                    di_pay_off = self.rebate / 365 * self.tenor * self.initial
                    di_1 = DigitalOption(
                        self.rrf,
                        self.div,
                        self.vol,
                        self.H,
                        self.spot,
                        self.cal_days,
                        self.bus_days,
                        self.disc_mode,
                        di_pay_off,
                        self.N,
                        True,
                        bus_days_in_year=self.bus_days_in_year,
                    )
                    eu_1.price_bsm()
                    di_1.price_bsm()
                    self.value = eu_1.value * self.participation_rate + di_1.value
            if not self.is_call and (
                self.up_or_do.upper() == "DO" or self.up_or_do.upper() == "DOWN"
            ):
                if self.H < self.K:
                    eu_1 = EuropeanVanillaOption(
                        self.rrf,
                        self.div,
                        self.vol,
                        self.K,
                        self.spot,
                        self.cal_days,
                        self.bus_days,
                        self.disc_mode,
                        self.N,
                        False,
                        self.tenor,
                        bus_days_in_year=self.bus_days_in_year,
                    )
                    eu_2 = EuropeanVanillaOption(
                        self.rrf,
                        self.div,
                        self.vol,
                        self.H,
                        self.spot,
                        self.cal_days,
                        self.bus_days,
                        self.disc_mode,
                        self.N,
                        False,
                        self.tenor,
                        bus_days_in_year=self.bus_days_in_year,
                    )
                    di_pay_off = (
                        ((self.K - self.H) / self.initial - self.rebate)
                        / 365
                        * self.tenor
                        * self.initial
                        / self.participation_rate
                    )
                    di_1 = DigitalOption(
                        self.rrf,
                        self.div,
                        self.vol,
                        self.H,
                        self.spot,
                        self.cal_days,
                        self.bus_days,
                        self.disc_mode,
                        di_pay_off,
                        self.N,
                        False,
                        bus_days_in_year=self.bus_days_in_year,
                    )
                    eu_1.price_bsm()
                    eu_2.price_bsm()
                    di_1.price_bsm()
                    self.value = (
                        eu_1.value - eu_2.value - di_1.value
                    ) * self.participation_rate
                else:
                    di_pay_off = self.initial * self.rebate / 365 * self.tenor
                    di_1 = DigitalOption(
                        self.rrf,
                        self.div,
                        self.vol,
                        self.H,
                        self.spot,
                        self.cal_days,
                        self.bus_days,
                        self.disc_mode,
                        di_pay_off,
                        self.N,
                        False,
                        bus_days_in_year=self.bus_days_in_year,
                    )
                    di_1.price_bsm()
                    self.value = di_1.value
