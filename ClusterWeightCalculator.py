import numpy as np
from scipy.optimize import minimize_scalar
from scipy.integrate import quad
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
        rate = (term ** a_z) / (1 + term ** (a_z + b_z))
        return rate

    def _metallicity_weight(self, Z, z):
        zeta = np.log10(Z / self.z_sun)
        mu_zeta = 0.153 - 0.074 * (z ** 1.34)
        sigma_zeta = 0.5
        weight = np.exp(- ((zeta - mu_zeta) ** 2) / (2 * sigma_zeta ** 2))
        return weight

    def _k_gw_correction(self, Z, rv):
        """
        Calculates the correction factor K_GW(Z) to account for clusters
        that evaporate prior to the present day (Eq 17).
        """
        Delta = 10 ** 5.33
        M_min = 10 ** 4
        M_max = 10 ** 8

        match Z:
            case 0.03:
                A = 3.6
                alpha = 2
                beta = 2.2
            case 0.3:
                A = 1.3
                alpha = 1.9
                beta = 2.7
            case 1.0:
                A = 1.3
                alpha = 2.4
                beta = 2.1

        alpha = 1.0
        beta = 1.0

        def phi_m0(m):
            # Eq 9: Birth mass function
            return (m ** -2) * np.exp(-m / self.m_star)

        def phi_m(m_val):
            # Eq 16: Present-day mass function
            return (m_val + Delta) ** -2 * np.exp(-(m_val + Delta) / (self.m_star / 2))

        def mass_dependency(m):
            # From Eq 4 & 5: (M_0 / 10^6 M_sun)^alpha
            return A * (m / 1e6) ** alpha * rv ** beta

        def num_integrand(m):
            return phi_m0(m) * mass_dependency(m)

        def den_integrand(m):
            return phi_m(m / 2.0) * mass_dependency(m)

        # Integrate over the mass limits (10^4 to 10^8)
        num, _ = quad(num_integrand, M_min, M_max)
        den, _ = quad(den_integrand, M_min, M_max)

        return num / den

    def calculate_model(self, rv, Z, N, avg_GC_mass):
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
        w_correction = self._k_gw_correction(Z, rv)
        w_z = -res.fun

        # Verify this value matches rate*Avg M_GC = f_cl = 0.258
        phi = 0.01 * ((1 + res.x) ** 2.6) / (1 + ((1 + res.x) / 3.2) ** 6.2)
        #print(self._formation_rate_density(res.x) * avg_GC_mass * w_mass * 100 / phi)

        # This is the standalone weight for this specific configuration
        independent_weight = w_mass * w_radius * w_z * w_correction

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

    unique_mids = np.unique(data[:, 0])
    GC_avg_masses = {}
    for mid in unique_mids:
        model_data = data[data[:, 0] == mid]
        GC_avg_masses[mid] = {np.sum((model_data[:, 8]) + model_data[:, 9]) / len(model_data[:, 8])}

    unique_models = {}
    for row in data:
        mid = row[0]
        if mid not in unique_models:
            unique_models[mid] = {'rv': row[1], 'Z': row[3], 'N': row[4], 'Mean M': list(GC_avg_masses[mid])[0]}

    calculator = ClusterWeightCalculator()
    model_results = {}

    for mid, props in unique_models.items():
        best_z, birth_myr, indep_weight = calculator.calculate_model(
            props['rv'], props['Z'], props['N'], props['Mean M']
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