# %%

from matplotlib.offsetbox import DEBUG

from lrfhss.run import *
import time
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

loops = 100                           # Monte-Carlo repetitions per point.
jobs = 16                             # Parallel worker threads.
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
    ocw_hz=1_523_000,                  # Occupied Channel Width (Hz) - 1.523 MHz.
    obw_hz=488,                        # Occupied Bandwidth per sub-channel (Hz).
    grid_spacing_hz=25_400,            # Minimum hop spacing (Hz) - 25.4 kHz.
    # Grid structure: 3120 total channels, 52 grids, 60 channels/grid.
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
# Helper: run a scenario for one node count
# link_cfg can be a LinkConfig instance OR a callable(distance)->LinkConfig
# =====================================================================
def run_scenario(n_total, link_cfg, number_relays=0, relay_pos=None):
    DEBUG = False
    """Run *loops* Monte-Carlo repetitions and return
    (mean_success, std_success, mean_goodput, mean_tx)."""
    s_dict = settings_template.copy()
    s_dict['number_nodes'] = n_total
    s_dict['link_config'] = link_cfg
    s_dict['number_relays'] = number_relays
    s_dict['relay_positions'] = relay_pos
    s = Settings(**s_dict)

    if DEBUG:
        results = [run_sim(s, seed=1)]  # single run for quick testing
    else:    
        results = Parallel(n_jobs=jobs)(
            delayed(run_sim)(s, seed=seed) for seed in range(loops)
        )
        
    successes = [r[0][0] for r in results if r != 1]
    goodputs  = [r[1][0] for r in results if r != 1]
    txs       = [r[2][0] for r in results if r != 1]
    if len(successes) == 0:
        return 0.0, 0.0, 0.0, 0.0
    return (np.mean(successes), np.std(successes),
            np.mean(goodputs), np.mean(txs))


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
            'loops': loops,
        }
        with open(output_file, 'wb') as f:
            pickle.dump(results_dict, f)

    # =====================================================================
    # Main simulation loop
    # =====================================================================
    start = time.perf_counter()

    for n in nNodes:
        print(f'\n===== Total nodes: {n} =====')

        # ---- Scenario 1: LR-FHSS, no relays ----
        print('  [1/4] LR-FHSS, no relays')
        m, s, g, t = run_scenario(n, lrfhss_link_config)
        scenarios['LRFHSS']['success'].append(m)
        scenarios['LRFHSS']['success_std'].append(s)
        scenarios['LRFHSS']['goodput'].append(g)
        scenarios['LRFHSS']['throughput'].append(t)
        save_partial_results()

        # ---- Scenario 2: LoRa SF rings, no relays ----
        # Uses the callable: each node gets its SF based on distance.
        print('  [2/4] LoRa SF rings, no relays')
        m, s, g, t = run_scenario(n, lora_sf_by_distance)
        scenarios['LoRa_SF_rings']['success'].append(m)
        scenarios['LoRa_SF_rings']['success_std'].append(s)
        scenarios['LoRa_SF_rings']['goodput'].append(g)
        scenarios['LoRa_SF_rings']['throughput'].append(t)
        save_partial_results()

        # ---- Scenario 3: LR-FHSS + 3 relays ----
        print('  [3/4] LR-FHSS + 3 relays @ 1.5 km')
        m, s, g, t = run_scenario(n, lrfhss_link_config,
                                   number_relays=3,
                                   relay_pos=relay_positions)
        scenarios['LRFHSS_3relays']['success'].append(m)
        scenarios['LRFHSS_3relays']['success_std'].append(s)
        scenarios['LRFHSS_3relays']['goodput'].append(g)
        scenarios['LRFHSS_3relays']['throughput'].append(t)
        save_partial_results()

        # ---- Scenario 4: LoRa SF rings + 3 relays ----
        print('  [4/4] LoRa SF rings + 3 relays @ 1.5 km')
        m, s, g, t = run_scenario(n, lora_sf_by_distance,
                                   number_relays=3,
                                   relay_pos=relay_positions)
        scenarios['LoRa_SF_rings_3relays']['success'].append(m)
        scenarios['LoRa_SF_rings_3relays']['success_std'].append(s)
        scenarios['LoRa_SF_rings_3relays']['goodput'].append(g)
        scenarios['LoRa_SF_rings_3relays']['throughput'].append(t)
        save_partial_results()

    elapsed = time.perf_counter() - start
    print(f'\nSimulation finished in {elapsed:.1f} s')

    # =====================================================================
    # Save results
    # =====================================================================
    save_partial_results()
    print(f'Results saved to {output_file}')
