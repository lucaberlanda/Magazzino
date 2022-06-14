import pandas as pd
from scipy.optimize import newton


class Bond:
    def __init__(self, face_value, coupon, years, price, ytm):

        self.face_value = face_value
        self.coupon = coupon
        self.years = years
        self.price = price
        self.ytm = ytm

    def discounted_cashflow(self, init_value=0):
        variables = {'coupon': init_value if self.coupon is None else self.coupon,
                     'ytm': init_value if self.ytm is None else self.ytm,
                     'price': init_value if self.price is None else self.price,
                     'years': self.years,
                     'face_value': init_value if self.face_value is None else self.face_value}

        cfs = {cnt: (variables['coupon'] * variables['face_value']) / (1 + variables['ytm']) ** cnt for cnt in
               range(1, variables['years'] + 1)}

        # add the notional
        cfs[variables['years']] = cfs[variables['years']] + variables['face_value'] / (1 + variables['ytm']) ** \
                                  variables['years']

        return variables, cfs

    def wrapper(self, init_value):
        variables, cfs = self.discounted_cashflow(init_value=init_value)
        to_minimize = variables['price'] - sum(cfs.values())
        return to_minimize

    def get_duration(self):
        bond_vars, cash_flows = self.discounted_cashflow()
        df = pd.Series(cash_flows).reset_index()
        df.columns = ['years', 'cash_flow']
        macaulay_duration = ((df.years * df.cash_flow) / df.cash_flow.sum()).sum()
        modified_duration = macaulay_duration / (1 + bond_vars['ytm'])
        return macaulay_duration, modified_duration

    def get_metrics(self):
        # todo init value as f(missing_value)
        init_value = 0.0
        res = newton(
            self.wrapper,
            init_value,
            tol=1e-11)

        if self.price is None:
            self.price = res

        if self.ytm is None:
            self.ytm = res

        if self.coupon is None:
            self.coupon = res

        return res


if __name__ == '__main__':

    bond = Bond(face_value=100,
                coupon=0.03,
                years=10,
                price=None,
                ytm=0.02)


    bond.get_metrics()
    mac_dur, mod_dur = bond.get_duration()

    print(mac_dur)
    print(mod_dur)
