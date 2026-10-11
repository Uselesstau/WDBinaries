import numpy as np
from scipy.optimize import minimize_scalar
from astropy.cosmology import Planck18
import astropy.units as u


class ClusterWeightCalculator:
    def __init__(self, z_sun=0.014, avg_stellar_mass=0.6):
        """
        Initializes the calculator using standard cosmological assumptions.
        z_sun: Solar metallicity.
        avg_stellar_mass: Conversion factor to estimate total mass M_0 from N stars.
        """
        self.z_sun = z_sun
        self.avg_stellar_mass = avg_stellar_mass
        self.m_star = 10 ** 6.26

    def _mass_weight(self, N_10_5):
        M_0 = N_10_5 * 1e5 * self.avg_stellar_mass
        weight = (M_0 ** -2) * np.exp(-M_0 / self.m_star)
        return weight

    def _radius_weight(self, rv):
        log_rv = np.log10(rv / 1.0)
        weight = np.exp(- (log_rv ** 2) / (2 * 0.3 ** 2))
        return weight

    def _formation_rate_density(self, z):
        z_p, a_z, b_z = 2.2, 2.6, 3.6
        term = (1 + z) / (1 + z_p)
        f_cl = 0.28
        rate = (f_cl * term ** a_z) / (1 + term ** (a_z + b_z))
        return rate

    def _metallicity_weight(self, Z, z):
        zeta = np.log10(Z / self.z_sun)
        mu_zeta = 0.153 - 0.074 * (z ** 1.34)
        sigma_zeta = 0.5
        weight = np.exp(- ((zeta - mu_zeta) ** 2) / (2 * sigma_zeta ** 2))
        return weight

    def calculate_model(self, rv, Z, N):
        """
        Calculates the optimal birth redshift and independent statistical weight.
        """

        def neg_prob(z):
            if z < 0 or z > 20:
                return 0.0
            return -(self._formation_rate_density(z) * self._metallicity_weight(Z, z))

        res = minimize_scalar(neg_prob, bounds=(0, 20), method='bounded')
        best_z = res.x
        birth_time_myr = Planck18.age(best_z).to(u.Myr).value

        # Calculate the independent combined weight
        w_mass = self._mass_weight(N)
        w_radius = self._radius_weight(rv)
        w_z = -res.fun

        # This is the standalone weight for this specific configuration
        independent_weight = w_mass * w_radius * w_z

        return best_z, birth_time_myr, independent_weight


def inspect_model_weights(model_stats, top_n=22):
    """
    Prints a formatted table of the calculated weights and birth times for validation.
    Sorts the models from the highest weight (most realistic) to the lowest weight.
    """
    print("\n=== Model Weight Validation ===")
    print(
        f"{'Model ID':<10} | {'rv (pc)':<8} | {'Z':<8} | {'N (x10^5)':<10} | {'Birth z':<8} | {'Birth (Myr)':<12} | {'Weight'}")
    print("-" * 90)

    # Sort the dictionary by weight in descending order
    sorted_models = sorted(model_stats.items(), key=lambda item: item[1]['weight'], reverse=True)

    count = 0
    for mid, stats in sorted_models:
        # Format the variables for clean printing
        rv = stats['rv']
        Z = stats['Z']
        N = stats['N']
        bz = stats['birth_z']
        b_time = stats['birth_time_myr']
        weight = stats['weight']

        print(f"{mid:<10.0f} | {rv:<8.2f} | {Z:<8.4f} | {N:<10.2f} | {bz:<8.2f} | {b_time:<12.1f} | {weight:.5e}")

        count += 1
        if count >= top_n:
            break

    print("-" * 90)
    print(f"Showing Top {top_n} out of {len(model_stats)} models.\n")


def assign_model_weights_and_times(data):
    """
    Extracts unique models and calculates their independent birth times and weights.
    """
    print("\nCalculating cosmological birth times and independent weights for models...")

    unique_models = {}
    for row in data:
        mid = row[0]
        if mid not in unique_models:
            unique_models[mid] = {'rv': row[1], 'Z': row[3], 'N': row[4]}

    calculator = ClusterWeightCalculator()
    model_results = {}

    for mid, props in unique_models.items():
        best_z, birth_myr, indep_weight = calculator.calculate_model(
            props['rv'], props['Z'], props['N']
        )
        model_results[mid] = {
            'rv': props['rv'],
            'Z': props['Z'],
            'N': props['N'],
            'birth_z': best_z,
            'birth_time_myr': birth_myr,
            'weight': indep_weight  # Assigned independently
        }

    print(f"    - Successfully processed {len(model_results)} unique models.")
    return model_results