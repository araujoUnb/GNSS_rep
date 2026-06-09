from pathlib import Path
import shelve

tau = {}
files = Path('.').glob('results/results_*.dat')
for file in files:
    with shelve.open(str(file), 'r') as f:
        angle_diff_deg = f['angle_diff_deg']
        delay_diff = f['delay_diff']
        C_N0 = f['C_N0']
        xi = f['xi']
        epsilon = f['epsilon']
        batch = f['batch']

        key = (angle_diff_deg, delay_diff, C_N0, xi, epsilon)

        data = {
            'real' : f['tau_real'],
            'HOSVD_ideal' : f['tau_HOSVD_ideal'],
            'bayOpt_est' : f['tau_bayOpt_est'],
            'bayOpt_ref_est' : f['tau_bayOpt_ref_est'],
            'lskrf_est' : f['tau_lskrf_est'],
            'ag_est' : f['tau_ag_est'],
        }

        if key in tau:
            tau[key][batch] = data
        else: 
            tau[key] = {
                batch : data
            }

with shelve.open(f'results.dat', 'n') as f:
    f['tau'] = tau

