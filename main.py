
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
    spectral_info_params['roll_off'] = 0
    spectral_info = create_input_spectral_information(power=power, **spectral_info_params)

    SimParams.set_params(load_json(TEST_DIR / 'data' / 'sim_params.json'))
    fiber_dict = load_json(TEST_DIR / 'data' / 'raman_fiber_config.json')
    # fiber_dict['operational'] = {}
    # fiber_dict['params'].pop('raman_efficiency')
    fiber = RamanFiber(**fiber_dict)
    sim_params = SimParams.get()
    sim_params.raman_params.spatial_resolution = 10e3
    sim_params.raman_params.flag = False

    srs = RamanSolver.calculate_stimulated_raman_scattering(spectral_info, fiber, sim_params)
    alpha = fiber.alpha(spectral_info.frequency)
    # plt.plot(srs.z, transpose(log(srs.rho)),'o')
    # plt.plot(srs.z, transpose(outer(-alpha/2,srs.z)),'r.')
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
        k_tol = dispersion_tolerance * abs(alpha[i])
        f_cut_resolution = k_tol / abs(beta2) / (4 * pi ** 2 * (1 + dn) * slot_width)
        f_pump_resolution = k_tol / abs(beta2) / (4 * pi ** 2 * slot_width)
        ti = time()
        new = NliSolver._ggn_numeric_psi(cut_carrier, pump_carrier, srs, beta2, fiber.params.length)

        print('Time new: ', time() - ti)

        ti = time()
        old = NliSolver._generalized_psi(f_eval, cut_carrier, pump_carrier, f_cut_resolution, f_pump_resolution, srs,
                                         alpha[i], beta2,
                                         beta3, f_ref_beta)
        print('Time old: ', time() - ti)

        ti = time()
        gn = NliSolver._psi(cut_carrier,pump_carrier,alpha[i],beta2, fiber.params.length)
        print('Time gn: ', time() - ti)
        plt.plot(i, old, 'bo', label='GGN old')
        plt.plot(i, gn, 'ko', label='GN')
        plt.plot(i, new, 'r.', label='GGN new')
        if i == 0:
            plt.legend()
        print('-------------------------------------')
    plt.show()

if __name__ == '__main__':
    new_implementation()

