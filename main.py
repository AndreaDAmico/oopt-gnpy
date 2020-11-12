
from pathlib import Path
from numpy import load, pi,transpose,exp,outer,log
from matplotlib import pyplot as plt
from time import time

from gnpy.core.info import create_input_spectral_information
from gnpy.core.elements import Fiber, RamanFiber
from gnpy.core.parameters import SimParams
from gnpy.tools.json_io import load_json
from gnpy.core.science_utils import RamanSolver, NliSolver
TEST_DIR = Path(__file__).parent/'tests'

def new_implementation():
    power = 1e-3
    eqpt_params = load_json(TEST_DIR / 'data' / 'eqpt_config.json')
    spectral_info_params = eqpt_params['SI'][0]
    spectral_info_params.pop('power_dbm')
    spectral_info_params.pop('power_range_db')
    spectral_info_params.pop('tx_osnr')
    spectral_info_params.pop('sys_margins')
    spectral_info = create_input_spectral_information(power=power, **spectral_info_params)

    SimParams.set_params(load_json(TEST_DIR / 'data' / 'sim_params.json'))
    fiber_dict = load_json(TEST_DIR / 'data' / 'raman_fiber_config.json')
    # fiber_dict['operational'] = {}
    fiber = Fiber(**fiber_dict)
    sim_params = SimParams.get()

    srs = RamanSolver.calculate_stimulated_raman_scattering(spectral_info, fiber, sim_params)
    alpha = fiber.alpha(spectral_info.frequency)
    # plt.plot(srs.z, transpose(log(srs.rho)),'o')
    # plt.plot(srs.z, transpose(outer(-alpha/2,srs.z)),'r.')
    # dz = srs.z[1:] - srs.z[:-1]
    # plt.show()


    beta2 = fiber.params.beta2
    beta3 = fiber.params.beta3
    f_ref_beta = fiber.params.ref_frequency
    for i in range(0,10):
        cut_carrier = spectral_info.carriers[0]
        f_eval = cut_carrier.frequency
        pump_carrier = spectral_info.carriers[i]
        dn = abs(pump_carrier.channel_number - cut_carrier.channel_number)
        dispersion_tolerance = sim_params.nli_params.dispersion_tolerance
        phase_shift_tolerance = sim_params.nli_params.phase_shift_tolerance

        slot_width = max(spectral_info.slot_width)

        delta_z = sim_params.raman_params.spatial_resolution
        k_tol = dispersion_tolerance * abs(alpha[i])
        phi_tol = phase_shift_tolerance / delta_z
        f_cut_resolution = min(k_tol, phi_tol) / abs(beta2) / (4 * pi ** 2 * (1 + dn) * slot_width)
        f_pump_resolution = min(k_tol, phi_tol) / abs(beta2) / (4 * pi ** 2 * slot_width)
        ti = time()
        nli = NliSolver._generalized_psi(f_eval, cut_carrier, pump_carrier, f_cut_resolution, f_pump_resolution, srs, alpha[i], beta2,
                     beta3, f_ref_beta)
        print('Time new: ', time() - ti)

        ti = time()
        new = NliSolver._ggn_numeric_psi(f_eval, cut_carrier, pump_carrier, f_cut_resolution, f_pump_resolution, srs,
                                         alpha[i], beta2,
                                         beta3, f_ref_beta)
        print('Time old: ', time() - ti)
        plt.plot(i, nli, 'bo')
        plt.plot(i, new, 'r.')

if __name__ == '__main__':
    new_implementation()

