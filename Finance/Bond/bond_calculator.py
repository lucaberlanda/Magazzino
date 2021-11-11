from scipy.optimize import newton


class Bond:
    def __init__(self, face_value, coupon, years, price, ytm):
        self.face_value = face_value
        self.coupon = coupon
        self.years = years
        self.price = price
        self.ytm = ytm

    def cashflow(self, solve_for):
        variables = {'coupon': solve_for if self.coupon is None else self.coupon,
                     'ytm': solve_for if self.ytm is None else self.ytm,
                     'price': solve_for if self.price is None else self.price,
                     'years': self.years}

        cfs = [(variables['coupon'] * self.face_value) / (1 + variables['ytm']) ** cnt for cnt in
               range(1, variables['years'] + 1)]

        notional = self.face_value / (1 + variables['ytm']) ** variables['years']
        cfs.append(notional)
        to_minimize = variables['price'] - sum(cfs)
        return to_minimize

    def compute_yield(self):
        res = newton(
            self.cashflow,
            0.03,
            tol=1e-11)

        return res


bond = Bond(face_value=100, coupon=0.03, years=10, price=200, ytm=None)
print(bond.compute_yield())
