# %%

from matplotlib.offsetbox import DEBUG

from lrfhss.run import *
import itertools
import os
import time
from collections import defaultdict
from joblib import Parallel, delayed
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from lrfhss.traffic import *
from lrfhss.fading import No_Fading, Rayleigh_Fading
from lrfhss.link import LinkConfig
from lrfhss.pathloss import LogDistance_PathLoss

# =====================================================================
# Simulation parameters
# =====================================================================
nNodes_points = 10                    # Number of node-count sweep points.
nNodes_min = 10000                     # Minimum total number of nodes.
nNodes_max = 100000                    # Maximum total number of nodes.
nNodes = np.linspace(nNodes_min, nNodes_max, nNodes_points, dtype=int)

seeds = [42, 123, 456, 789]            # Four fixed seeds — averaged for each node-count point.
jobs = 32                             # One worker per physical core (32-core machine).
# Total parallel tasks: nNodes_points × scenarios × seeds = 10 × 4 × 4 = 160
checkpoint_dir = 'checkpoints'        # Directory for per-task resume files.
hours = 24                            # Simulation duration in hours.
max_radius = 3000                     # Cell radius (meters).

# %%

# =====================================================================
# Path-loss model  (log-distance, shared by all links)
# =====================================================================
pathloss_model = LogDistance_PathLoss({
    'gamma': 2.32,    # Path-loss exponent.
    'd0':    1000.0,  # Reference distance (m).
    'lpld0': 128.95,  # Path loss at d0 (dB).
    'std':   7.8,     # Shadowing std deviation (dB).
})

# =====================================================================
# Relay positions - 3 relays at 1.5 km, 120 degrees apart
# =====================================================================
relay_distance = 1500                 # Relay distance from gateway (m).
relay_angles_deg = [0, 120, 240]      # Angular spacing (degrees).
relay_positions = [
    (relay_distance * np.cos(np.radians(a)),
     relay_distance * np.sin(np.radians(a)))
    for a in relay_angles_deg
]

# =====================================================================
# LoRa SF ring definitions
# SF7: 0-1 km, then SF8-SF12 in uniform 400 m rings up to 3 km.
# =====================================================================
sf_rings = [
    {'sf': 7,  'min_d': 0,    'max_d': 1000},
    {'sf': 8,  'min_d': 1000, 'max_d': 1400},
    {'sf': 9,  'min_d': 1400, 'max_d': 1800},
    {'sf': 10, 'min_d': 1800, 'max_d': 2200},
    {'sf': 11, 'min_d': 2200, 'max_d': 2600},
    {'sf': 12, 'min_d': 2600, 'max_d': 3000},
]

# =====================================================================
# Per-node LoRa LinkConfig assigner (callable: distance -> LinkConfig)
# Each node gets its own LinkConfig based on distance to gateway.
# =====================================================================
def lora_sf_by_distance(distance):
    """Return a LoRa LinkConfig with the SF matching the node's ring."""
    for ring in sf_rings:
        if distance < ring['max_d']:
            sf = ring['sf']
            break
    else:
        sf = 12  # beyond last ring, use highest SF
    return LinkConfig(
        link_type='lora',
        payload_size=20,
        sensitivity=-120.0,
        transmission_power=14.0,
        pathloss_model=pathloss_model,
        sf=sf,
        bw=125,
        cr=1,
        lora_channels=8,
    )

# =====================================================================
# LR-FHSS LinkConfig (all options exposed)
# =====================================================================
lrfhss_link_config = LinkConfig(
    link_type='lrfhss',               # 'lrfhss' or 'lora'.
    payload_size=20,                   # Application payload size (bytes).
    sensitivity=-137.0,                # Receiver sensitivity (dBm).
    transmission_power=14.0,           # Device transmit power (dBm).
    pathloss_model=pathloss_model,     # Log-distance path-loss model.
    headers=3,                         # Number of header replicas.
    header_duration=0.233472,          # Duration of one header (seconds).
    payload_duration=0.1024,           # Duration of one payload fragment (s).
    transceiver_wait=0.006472,         # Radio turnaround / wait time (s).
    code='1/3',                        # Coding rate.
    #ocw_hz=1_523_000,                  # Occupied Channel Width (Hz) - 1.523 MHz.
    #obw_hz=488,                        # Occupied Bandwidth per sub-channel (Hz).
    #grid_spacing_hz=25_400,            # Minimum hop spacing (Hz) - 25.4 kHz.
    # US/AU - Grid structure: 3120 total channels, 52 grids, 60 channels/grid.
    ocw_hz=336_000,                  # Europe - Occupied Channel Width (Hz) - 336KHz.
    obw_hz=488,                        # Occupied Bandwidth per sub-channel (Hz).
    grid_spacing_hz=3_900,            # Europe - Minimum hop spacing (Hz) - 3.9 kHz.
    # EU - Grid structure: 688 total channels, 8 grids, 86 channels/grid.
    payloads=None,                     # Payload fragments (None = auto).
    threshold=None,                    # Decode threshold  (None = auto).
)

# =====================================================================
# Rayleigh fading parameters
# =====================================================================
rayleigh_fading_param = {
    'scale': 1.0,                      # Rayleigh distribution scale.
}

# =====================================================================
# Common Settings template (shared by all scenarios)
# =====================================================================
settings_template = {
    'number_nodes': nNodes[0],         # Overridden per run.
    'simulation_time': 1 * hours,# Total simulation duration (s).
    'payload_size': 20,                # Application payload size (bytes).
    'headers': 3,                      # LR-FHSS number of header replicas.
    'header_duration': 0.233472,       # Duration of one header (seconds).
    'payloads': None,                  # Payload fragments (None = auto).
    'threshold': None,                 # Decode threshold  (None = auto).
    'payload_duration': 0.1024,        # Duration of one payload fragment (s).
    'code': '1/3',                     # LR-FHSS coding rate.
    'traffic_class': Exponential_Traffic,  # Traffic generator class.
    'traffic_param': {'average_interval': 900},  # Traffic params.
    'transceiver_wait': 0.006472,      # Radio turnaround / wait time (s).
    'base': 'core',                    # Base station decoder.
    'window_size': 2,                  # Sliding window size.
    'window_step': 0.5,                # Sliding window step (s).
    'sensitivity': -137,               # Receiver sensitivity (dBm).
    'fading_class': Rayleigh_Fading,   # Channel fading model class.
    'min_distance': 0,                 # Minimum node distance (m).
    'max_distance': max_radius,        # Maximum node distance (m).
    'transmission_power': 14,          # Device transmit power (dBm).
    'fading_param': rayleigh_fading_param,  # Fading model parameters.
    'link_config': lrfhss_link_config, # Overridden for LoRa scenarios.
    'number_relays': 0,                # Overridden for relay scenarios.
    'relay_link_config': None,         # Relay link (None = same as devices).
    'relay_positions': None,           # Overridden for relay scenarios.
    'relay_dutycycle_period_s': 60.0,  # Relay duty-cycle period (seconds).
    'relay_dutycycle_percent': 70.0,   # Relay listening share per period (%).
    'lora_channels': 8,                # LoRa channels at base station.
    'pathloss_class': LogDistance_PathLoss,  # Path-loss model class.
    'pathloss_param': {                # Path-loss model parameters.
        'gamma': 2.32,                 #   Path-loss exponent.
        'd0':    1000.0,               #   Reference distance (m).
        'lpld0': 128.95,               #   Path loss at d0 (dB).
        'std':   7.8,                  #   Shadowing std deviation (dB).
    },
    'base_position': (0, 0),           # Gateway position.
}

# %%

# =====================================================================
# Scenario parameter table — maps scenario key → link and relay config.
# Defined at module level so joblib worker processes can pickle it.
# =====================================================================
SCENARIO_CONFIGS = {
    'LRFHSS': {
        'link_cfg':      lrfhss_link_config,
        'number_relays': 0,
        'relay_pos':     None,
    },
    'LoRa_SF_rings': {
        'link_cfg':      lora_sf_by_distance,
        'number_relays': 0,
        'relay_pos':     None,
    },
    'LRFHSS_3relays': {
        'link_cfg':      lrfhss_link_config,
        'number_relays': 3,
        'relay_pos':     relay_positions,
    },
    'LoRa_SF_rings_3relays': {
        'link_cfg':      lora_sf_by_distance,
        'number_relays': 3,
        'relay_pos':     relay_positions,
    },
}


def _ckpt_path(n, scenario_key, seed):
    """Return the checkpoint file path for a given task."""
    return os.path.join(checkpoint_dir, f'{scenario_key}_n{n}_s{seed}.pkl')


def run_single(n, scenario_key, seed):
    """Atomic task: one (node-count, scenario, seed) triple → one run_sim call.
    Returns (success_rate, goodput, tx_count) or None on failure.
    If a checkpoint file exists the result is loaded instantly (resume support)."""
    ckpt = _ckpt_path(n, scenario_key, seed)

    # ── Resume: return cached result if this task already finished ────────
    if os.path.exists(ckpt):
        with open(ckpt, 'rb') as f:
            return pickle.load(f)

    # ── Run simulation ────────────────────────────────────────────────────
    cfg = SCENARIO_CONFIGS[scenario_key]
    s_dict = settings_template.copy()
    s_dict['number_nodes']    = n
    s_dict['link_config']     = cfg['link_cfg']
    s_dict['number_relays']   = cfg['number_relays']
    s_dict['relay_positions'] = cfg['relay_pos']
    s = Settings(**s_dict)
    r = run_sim(s, seed=seed)
    if r == 1:      # failed run — do NOT checkpoint so it will retry
        return None
    result = r[0][0], r[1][0], r[2][0]   # (success, goodput, tx)

    # ── Save checkpoint atomically (write to .tmp then rename) ────────────
    tmp = ckpt + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(result, f)
    os.replace(tmp, ckpt)   # atomic on POSIX — safe against mid-write crashes

    return result


if __name__ == '__main__':

    # =====================================================================
    # Result containers - four scenarios
    # =====================================================================
    scenarios = {
        'LRFHSS': {
            'success': [], 'success_std': [], 'goodput': [], 'throughput': [],
        },
        'LoRa_SF_rings': {
            'success': [], 'success_std': [], 'goodput': [], 'throughput': [],
        },
        'LRFHSS_3relays': {
            'success': [], 'success_std': [], 'goodput': [], 'throughput': [],
        },
        'LoRa_SF_rings_3relays': {
            'success': [], 'success_std': [], 'goodput': [], 'throughput': [],
        },
    }

    output_file = 'relay_comparison.pkl'

    def save_partial_results():
        results_dict = {
            'nNodes': nNodes,
            'scenarios': scenarios,
            'sf_rings': sf_rings,
            'relay_positions': relay_positions,
            'seeds': seeds,
        }
        with open(output_file, 'wb') as f:
            pickle.dump(results_dict, f)

    # =====================================================================
    # Flat task list — every (nNodes, scenario, seed) combination.
    # 10 × 4 × 4 = 160 independent tasks dispatched to 32 workers
    # (~5 rounds of 32 simultaneous jobs).
    # =====================================================================
    os.makedirs(checkpoint_dir, exist_ok=True)
    scenario_keys = list(SCENARIO_CONFIGS.keys())
    task_list = list(itertools.product(nNodes, scenario_keys, seeds))

    cached = sum(1 for n, sc, seed in task_list if os.path.exists(_ckpt_path(n, sc, seed)))
    print(f'{cached}/{len(task_list)} tasks already cached — will be skipped.')

    start = time.perf_counter()
    print(f'Dispatching {len(task_list)} tasks across {jobs} workers '
          f'(nNodes={len(nNodes)}, scenarios={len(scenario_keys)}, seeds={seeds})')

    raw_results = Parallel(n_jobs=jobs, backend='loky', verbose=5)(
        delayed(run_single)(n, sc, seed)
        for n, sc, seed in task_list
    )

    # ── Reassemble: bucket results by (n, scenario_key), then aggregate ───
    bucket = defaultdict(list)   # (n, sc) → [(success, goodput, tx), ...]
    for (n, sc, _seed), res in zip(task_list, raw_results):
        if res is not None:
            bucket[(n, sc)].append(res)

    for n in nNodes:
        for sc in scenario_keys:
            runs = bucket[(n, sc)]
            if runs:
                succs = [r[0] for r in runs]
                gputs = [r[1] for r in runs]
                txs   = [r[2] for r in runs]
                m, s_std, g, t = (np.mean(succs), np.std(succs),
                                  np.mean(gputs), np.mean(txs))
            else:
                m, s_std, g, t = 0.0, 0.0, 0.0, 0.0
            scenarios[sc]['success'].append(m)
            scenarios[sc]['success_std'].append(s_std)
            scenarios[sc]['goodput'].append(g)
            scenarios[sc]['throughput'].append(t)

    save_partial_results()

    elapsed = time.perf_counter() - start
    print(f'\nSimulation finished in {elapsed:.1f} s')

    # =====================================================================
    # Save results
    # =====================================================================
    save_partial_results()
    print(f'Results saved to {output_file}')
