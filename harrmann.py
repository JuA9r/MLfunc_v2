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

        term_1 = 0
        if self.x_0 != 0:
            ans1 = MittagLefflerUtil.compute_MittagLeffler(2 * self.alpha, 1, z)
            term_1 = ans1 * self.x_0

        term2 = 0
        if self.y_0 != 0:
            ans2 = MittagLefflerUtil.compute_MittagLeffler(2 * self.alpha, self.alpha + 1, z)
            term2 = ans2 * self.y_0 * (t ** self.alpha)

        return term_1 + term2


# ----- Main execution unit -----
def main():
    plt.rcParams['mathtext.fontset'] = 'cm'
    t = np.linspace(0, 8, 100)
    alphas = [1.2, 1.5, 1.8, 2.0]

    # --- Case 1 ---
    plt.figure(figsize=(12, 8))

    for a in alphas:
        oscillator = FractionalHarmonicOscillator(alpha=a, omega=1.00, x_0=1.00, y_0=0.0)
        x_t = oscillator.calculate_function(t)

        label_str = r'$\alpha=%.1f$' % a
        if a == 1.0: label_str += '(Standard HO)'
        plt.plot(t, x_t, label=label_str)

    plt.title('Case 1: $x(0)=1, y(0)=0$')
    plt.xlabel('Time $t$')
    plt.ylabel('$x(t)$')
    plt.ylim(-20.5, 20.5)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # save case1 graph
    plt.savefig('case1.png', dpi=300)
    print("Saved case1.png")

    plt.show()
    plt.close()

    # --- Case 2  ---
    plt.figure(figsize=(12, 8))

    for a in alphas:
        oscillator = FractionalHarmonicOscillator(alpha=a, omega=1.00, x_0=0.0, y_0=1.00)
        x_t = oscillator.calculate_function(t)

        label_str = r'$\alpha=%.1f$' % a
        plt.plot(t, x_t, label=label_str)

    plt.title(r'Case 2: $x(0)=0, y(0)=1$')
    plt.xlabel('Time $t$')
    plt.ylabel('$x(t)$')
    plt.ylim(-20.5, 20.5)
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.savefig('case2.png', dpi=300)
    print("Saved case2.png")

    plt.show()
    plt.close()


if __name__ == "__main__":
    main()