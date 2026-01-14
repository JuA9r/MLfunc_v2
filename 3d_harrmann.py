import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import gamma
import warnings

warnings.filterwarnings('ignore')


class MittagLefflerUtil:
    @staticmethod
    def compute_MittagLeffler(alpha, beta, z, n_terms=80) -> np.ndarray:
        result = np.zeros_like(z, dtype=np.complex128)
        for k in range(n_terms):
            try:
                term = (z ** k) / gamma(alpha * k + beta)
                result += term
            except ArithmeticError:
                break
        return np.real(result)


class FractionalHarmonicOscillator:
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


def main():
    t_vals = np.linspace(0, 8, 100)
    alpha_vals = np.linspace(1.1, 2.0, 30)

    T, A = np.meshgrid(t_vals, alpha_vals)
    Z = np.zeros_like(T)

    print("Calculation 3D data...")

    for i, a in enumerate(alpha_vals):
        oscillator = FractionalHarmonicOscillator(alpha=a, omega=1.0, x_0=1.0, y_0=0.0)
        Z[i, :] = oscillator.calculate_function(t_vals)

    # --- 3Dプロットの描画 ---
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(T, A, Z, cmap='viridis', edgecolor='none', alpha=0.9)

    ax.set_title(r'3D Visualization: Case 1 ($x_0=1, y_0=0$)', fontsize=15)
    ax.set_xlabel('Time $t$', fontsize=12)
    ax.set_ylabel(r'Fractional Order $\alpha$', fontsize=12)
    ax.set_zlabel(r'Displacement $x(t)$', fontsize=12)

    ax.view_init(elev=30, azim=225)

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='$x(t)$ amplitude')

    plt.savefig('case1_3d.png', dpi=300)
    print("Saved case1_3d.png")

    plt.show()


if __name__ == "__main__":
    main()