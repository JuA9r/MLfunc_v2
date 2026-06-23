import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma


class MittagLefflerUtil:
    """
    A utility class for calculating the Mittag-Leffler function E_{alpha, beta}(z).
    """

    @staticmethod
    def compute_MittagLeffler(alpha, beta, z, n_terms=50) -> np.ndarray:
        """
        E_{alpha, beta}(z) = sum_{k=0}^{inf}z^k / Gamma(alpha*k+beta) based calculation.
        """
        result = np.zeros_like(z, dtype=np.complex128)

        for k in range(n_terms):
            term = z ** k / gamma(alpha * k + beta)
            result += term

        return np.real(result)


class FractionalHarmonicOscillator:
    """
    Class representing fractional harmonic oscillators.
    """

    def __init__(self, alpha, omega=1.0, x_0=1.0, y_0=1.0) -> None:
        self.alpha = alpha
        self.omega = omega
        self.x_0 = x_0
        self.y_0 = y_0

    def calculate_function(self, t) -> np.ndarray:
        z = -1 * (self.omega * t) ** (2 * self.alpha)

        term_1 = np.zeros_like(t, dtype=float)
        term_2 = np.zeros_like(t, dtype=float)

        if self.x_0 != 0:
            ans1 = MittagLefflerUtil.compute_MittagLeffler(2 * self.alpha, 1, z)
            term_1 = ans1 * self.x_0

        if self.y_0 != 0:
            ans2 = MittagLefflerUtil.compute_MittagLeffler(2 * self.alpha, self.alpha + 1, z)
            term_2 = ans2 * self.y_0 * (t ** self.alpha)

        return term_1 + term_2


# ----- Main execution unit -----

def main():
    alphas = []
    print("4つの1以上2未満の少数値を入力してください: ")
    for i in range(4):
        while True:
            try:
                num = float(input(f"input No.{i+1}: "))
                if num < 0:
                    print("1以上の少数値を入力してください")
                    continue
                alphas.append(num)
                break

            except ValueError:
                print("有効な浮動少数を入力してください")

    plt.rcParams['mathtext.fontset'] = 'cm'
    t = np.linspace(0, 8, 100)

    cases = [
        {"title": "Case 1: $x(0)=1, y(0)=0$", "x_0": 1.0, "y_0": 0.0, "filename": "case1.png"},
        {"title": "Case 2: $x(0)=0, y(0)=1$", "x_0": 0.0, "y_0": 1.0, "filename": "case2.png"}
    ]

    for case in cases:
        plt.figure(figsize=(12, 8))

        for a in alphas:
            oscillator = FractionalHarmonicOscillator(
                alpha=a, omega=1.00, x_0=case["x_0"], y_0=case["y_0"]
            )
            x_t = oscillator.calculate_function(t)

            label_str = r'$\alpha=%.3f$' % a
            if a == 1.0:
                label_str += ' (Standard HO)'
            plt.plot(t, x_t, label=label_str)

        # グラフの装飾と保存

        plt.title(case["title"])
        plt.xlabel('Time $t$')
        plt.ylabel('$x(t)$')
        plt.ylim(-20.5, 20.5)
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.savefig(case["filename"], dpi=300)
        print(f"Saved {case['filename']}")

        plt.show()
        plt.close()


if __name__ == "__main__":
    main()