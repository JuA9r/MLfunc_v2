import numpy as np
import mpmath as mp
from scipy.integrate import quad
import matplotlib.pyplot as plt
from dataclasses import dataclass
import os


mp.mp.dps = 30


@dataclass
class H_FunctionParameter:
    value: complex
    weight: float


class H_Function:
    def __init__(
            self, int_m: int, int_n: int,
            a_parameter: list[H_FunctionParameter],
            b_parameter: list[H_FunctionParameter]
    ) -> None:
        self.__int_m = int_m
        self.__int_n = int_n
        self.__a = a_parameter
        self.__b = b_parameter

    def evaluate(self, z: float, nu: float, x: float) -> float:
        if hasattr(mp, 'fox_h'):
            a_tuples = [(p.value, p.weight) for p in self.__a]
            upper = [a_tuples[:self.__int_n], a_tuples[self.__int_n:]]
            b_tuples = [(p.value, p.weight) for p in self.__b]
            lower = [b_tuples[:self.__int_m], b_tuples[self.__int_m:]]
            try:
                val = mp.foxh(upper, lower, z)
                return float(mp.re(val)) / (nu * x)
            except ValueError:
                return self.__fallback_integral(x, nu)
        else:
            return self.__fallback_integral(x, nu)

    @staticmethod
    def __fallback_integral(x: float, nu: float) -> float:
        integrand = lambda k: np.cos(k * x) * np.cos(k ** (nu / 2)) * np.exp(-0.001 * k)
        res, _ = quad(integrand, 0, 1000, limit=1000)
        return res / np.pi


def main():
    nu_list = [1.2, 1.5, 1.8, 2.0]
    print(f"Using mpmath version: {mp.__version__}")

    x_vals = np.linspace(0.1, 1.2, 150)
    fig = plt.figure(figsize=(10, 6))

    for nu in nu_list:
        a_params = [
            H_FunctionParameter(0, 1 / nu),
            H_FunctionParameter(0, 1),
            H_FunctionParameter(0, 0.5)
        ]
        b_params = [
            H_FunctionParameter(0, 1 / nu),
            H_FunctionParameter(0, 2 / nu),
            H_FunctionParameter(0, 0.5)
        ]

        h_func = H_Function(1, 2, a_params, b_params)

        y_vals = []
        for x in x_vals:
            val = h_func.evaluate(1 / x, nu, x)
            y_vals.append(val)

        line_style = '-' if nu == 2.0 else '-'
        plt.plot(x_vals, y_vals, label=f'$\\nu={nu}$', linewidth=2, linestyle=line_style)

    plt.axhline(0.0, color='black', lw=0.5)
    plt.title("Green Function $K_{\\nu, 2}(x)$ for various $\\nu$")
    plt.xlabel("$x$")
    plt.ylabel("$K_{\\nu, 2}(x)$")
    plt.grid(True, linestyle='-', alpha=0.7)
    plt.legend()

    save_filename = "ImageFile/frac_wave_green_func.png"

    plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    print(f"Saved graph to '{save_filename}'")
    print(f"Absolute path: {os.path.abspath(save_filename)}")

    plt.show()


if __name__ == "__main__":
    main()