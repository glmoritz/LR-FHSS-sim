# LR-FHSS-sim

LR-FHSS and conventional LoRa event-driven simulator with **relay support**, built in Python using [SimPy](https://simpy.readthedocs.io/).

The simulator models LR-FHSS frequency-hopping transmissions **and** conventional LoRa transmissions under a unified link abstraction. Nodes broadcast to all receivers (sinks and relays) simultaneously. **Relays are passive listener base stations** that independently decode transmissions and forward them to the sink.

---

## Features

| Feature | Description |
|---|---|
| **LR-FHSS simulation** | Fragment-level (header + payload) frequency-hopping with configurable code rates (1/3, 2/3, 1/2, 5/6). |
| **Conventional LoRa simulation** | Single-fragment packets with SF7-12, BW 125/250/500 kHz, LoRa airtime formula, per-SF sensitivity and minimum SNR checks. |
| **Unified link abstraction** | `LinkConfig` class configures either link type. A LoRa transmission is treated as a single fragment reusing the same collision machinery. |
| **Passive relay base stations** | `Relay(Base)` — independent base stations that receive, decode, and forward packets. Relays have their own collision tracking (spatial diversity). |
| **Half-duplex constraint** | Receivers and nodes cannot listen and transmit at the same time. When a relay forwards, all ongoing receptions are cancelled. |
| **2D node positioning** | Nodes and receivers have (x, y) coordinates. Distance-based path loss is computed per receiver. |
| **Fading models** | Pluggable: No fading, Rayleigh, Rician, Nakagami-m. |
| **Traffic models** | Pluggable: Exponential, Uniform, Constant, Two-state Markovian. |
| **ACRDA receiver** | Successive Interference Cancellation (SIC) window-based decoder. |
| **Log-distance path loss** | Used for LoRa fragment reception (configurable γ, d₀, Lpl(d₀), σ). |
| **Deduplication** | When both direct and relay paths succeed, the packet is counted only once at the sink. |

---

## Installation

Download or clone the source code:

```sh
git clone https://github.com/misaelrc/LR-FHSS-sim.git
cd LR-FHSS-sim
```

Install prerequisites:

```sh
pip install -r requirements.txt
```

Install the library:

```sh
pip install .
```

For development (editable mode, changes apply immediately):

```sh
pip install -e .
```

---

## Project Structure

```
lrfhss/
├── __init__.py
├── link.py           # LinkConfig, LoRa airtime/sensitivity/SNR tables
├── lrfhss_core.py    # Fragment, Packet, Node, Base, Relay, Traffic, Fading ABCs
├── acrda.py           # ACRDA base station with SIC window
├── fading.py          # Rayleigh, Rician, Nakagami-m, No-fading implementations
├── traffic.py         # Exponential, Uniform, Constant, Markovian traffic generators
├── settings.py        # Settings class (all simulation parameters)
├── run.py             # run_sim() entry point
examples/
├── examples.ipynb     # Jupyter notebook with usage examples
conventional-lora/     # Reference conventional LoRa simulator (read-only)
```

---

## Quick Start

### 1. Pure LR-FHSS (original behavior)

```python
from lrfhss.run import run_sim
from lrfhss.settings import Settings

s = Settings(number_nodes=1000, simulation_time=3600)
result = run_sim(s)
# result = [[success_ratio], [goodput_bytes], [transmitted], [relayed]]
print(f"Success: {result[0][0]:.4f}")
```

All original parameters work exactly as before — full backward compatibility.

### 2. Pure conventional LoRa

```python
from lrfhss.link import LinkConfig

s = Settings(
    number_nodes=500,
    simulation_time=3600,
    link_config=LinkConfig(
        link_type='lora',
        sf=9,
        bw=125,
        cr=1,
        payload_size=10,
    ),
    max_distance=2000,
)
result = run_sim(s)
```

### 3. LoRa end-devices with LR-FHSS relays

```python
s = Settings(
    number_nodes=1000,
    simulation_time=3600,
    link_config=LinkConfig(link_type='lora', sf=9, bw=125, cr=1, payload_size=10),
    number_relays=3,
    relay_link_config=LinkConfig(link_type='lrfhss', payload_size=10, code='1/3'),
    relay_positions=[(500, 0), (0, 500), (-500, 0)],
    max_distance=2000,
)
result = run_sim(s)
print(f"Success: {result[0][0]:.4f}, Relayed: {result[3][0]}")
```

### 4. LR-FHSS with LR-FHSS relays

```python
s = Settings(
    number_nodes=1000,
    simulation_time=3600,
    link_config=LinkConfig(link_type='lrfhss', payload_size=10, code='1/3'),
    number_relays=2,
    relay_positions=[(800, 0), (-800, 0)],
    max_distance=2000,
)
result = run_sim(s)
```

### 5. With fading

```python
from lrfhss.fading import Rayleigh_Fading

s = Settings(
    number_nodes=500,
    simulation_time=3600,
    link_config=LinkConfig(link_type='lora', sf=12, bw=125, cr=1, payload_size=10),
    fading_class=Rayleigh_Fading,
    fading_param={'scale': 1.0},
    number_relays=2,
    relay_link_config=LinkConfig(link_type='lrfhss', payload_size=10, code='1/3'),
    relay_positions=[(600, 0), (0, 600)],
)
result = run_sim(s)
```

### 6. Parallel execution (multiple seeds)

```python
from joblib import Parallel, delayed
import numpy as np

s = Settings(number_nodes=1000, simulation_time=3600)
results = Parallel(n_jobs=8)(
    delayed(run_sim)(s, seed=seed) for seed in range(25)
)
avg_success = np.mean([r[0][0] for r in results])
print(f"Average success: {avg_success:.4f}")
```

---

## Key Classes

### `LinkConfig` ([lrfhss/link.py](lrfhss/link.py))

Defines **how** a transmission happens. Two link types:

| Parameter | `lrfhss` | `lora` |
|---|---|---|
| `link_type` | `'lrfhss'` | `'lora'` |
| Fragments | N headers + M payloads (frequency-hopping) | 1 single fragment |
| Key params | `headers`, `code`, `obw`, `payload_size` | `sf`, `bw`, `cr`, `payload_size` |
| Decode rule | ≥1 header + ≥threshold payloads OK | Single fragment succeeded |

```python
# LR-FHSS link
lc = LinkConfig(link_type='lrfhss', payload_size=10, code='1/3')

# LoRa link
lc = LinkConfig(link_type='lora', sf=9, bw=125, cr=1, payload_size=10)
```

### `Fragment` ([lrfhss/lrfhss_core.py](lrfhss/lrfhss_core.py))

A single transmission unit. Types: `'header'`, `'payload'` (LR-FHSS), or `'lora'`.
LoRa fragments carry `sf`, `bw`, `cr` for link-aware reception checks.

### `Packet` ([lrfhss/lrfhss_core.py](lrfhss/lrfhss_core.py))

A collection of fragments representing one transmission. Created via `LinkConfig`:

- **LR-FHSS**: multiple frequency-hopping fragments.
- **LoRa**: a single fragment with full LoRa airtime.

### `Node` ([lrfhss/lrfhss_core.py](lrfhss/lrfhss_core.py))

A network end-device with 2D `(x, y)` position. Broadcasts packets to **all receivers** (sinks and relays) simultaneously. The node has **no knowledge of relays** — it simply transmits fragments, and each receiver independently processes them with its own collision tracking.

### `Base` ([lrfhss/lrfhss_core.py](lrfhss/lrfhss_core.py))

A base station receiver (type `'sink'`). Handles both LR-FHSS and LoRa fragments:

- **LR-FHSS**: original fading-intensity SNR model.
- **LoRa**: log-distance path-loss → RSSI/SNR with per-SF sensitivity and minimum SNR requirements.
- **Collision detection**: per-channel, shared between both link types (separate channel spaces).
- **Deduplication**: tracks decoded packet IDs to avoid double-counting relay + direct.
- **Position**: has `(x, y)` coordinates used for distance-based reception checks.

### `Relay` ([lrfhss/lrfhss_core.py](lrfhss/lrfhss_core.py))

A relay base station (extends `Base`, type `'relay'`). Passive listener with independent collision tracking:

- **Receives like a Base**: tracks fragments on its own channels with its own collision state (spatial diversity from the sink).
- **Auto-forwards**: when `try_decode` succeeds, automatically schedules a forwarding process to the sink.
- **Half-duplex**: when forwarding starts, all ongoing receptions are cancelled. Cannot listen while transmitting.
- **New packet**: creates a fresh packet using `forward_link_config` (can be a different link type), preserving the original sender ID.

### `Settings` ([lrfhss/settings.py](lrfhss/settings.py))

All simulation parameters in one place. Key new parameters:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `link_config` | `LinkConfig` | `None` (legacy LR-FHSS) | End-device link configuration |
| `number_relays` | `int` | `0` | Number of relay nodes |
| `relay_link_config` | `LinkConfig` | Same as `link_config` | Link used by relays to forward |
| `relay_positions` | `list[(x,y)]` | `None` (random) | Fixed relay positions |
| `lora_channels` | `int` | `8` | LoRa channels at the gateway |
| `gamma` | `float` | `2.32` | Path-loss exponent |
| `d0` | `float` | `1000.0` | Reference distance (m) |
| `lpld0` | `float` | `128.95` | Path loss at d₀ (dB) |
| `std` | `float` | `7.8` | Shadowing std deviation (dB) |
| `base_position` | `tuple` | `(0, 0)` | Gateway position |

### `run_sim()` ([lrfhss/run.py](lrfhss/run.py))

Entry point. Returns `[[success_ratio], [goodput_bytes], [transmitted], [relayed_count]]`.

---

## Relay Architecture

```
                           ┌──────────────────────┐
                           │  Node (end-device)    │
                           │  broadcasts to ALL    │
                           │  receivers equally    │
                           └──────┬───────┬────────┘
                                  │       │
              same fragments      │       │  same fragments
              (independent copy)  │       │  (independent copy)
                                  ▼       ▼
                          ┌───────────┐ ┌───────────┐
                          │   Sink    │ │   Relay   │
                          │  (Base)   │ │  (Base)   │
                          │  type=    │ │  type=    │
                          │  'sink'   │ │  'relay'  │
                          └─────▲─────┘ └─────┬─────┘
                                │             │
                                │   decoded?  │
                                │   yes →     │
                                │   forward   │
                                │   new pkt   │
                                └─────────────┘
```

**Key design principles:**

1. **Passive listening**: relays are base stations that independently capture transmissions as they happen. Nodes have no knowledge of relays.
2. **Independent collision state**: each receiver (sink and every relay) maintains its own channel lists and collision tracking — spatial diversity is modeled.
3. **Half-duplex**: when a relay forwards a packet, it cancels all ongoing receptions. While transmitting, incoming fragments are silently dropped.
4. **Deduplication**: if both the sink and a relay successfully decode the same packet, it is counted only once (tracked by original packet ID).
5. **Heterogeneous links**: the relay can forward using a different link type than the original (e.g. end-device sends LoRa, relay forwards via LR-FHSS).

---

## Fading Models

| Class | Parameters | Description |
|---|---|---|
| `No_Fading` | — | Intensity = 1 (deterministic) |
| `Rayleigh_Fading` | `scale` | Rayleigh distributed envelope |
| `Rician_Fading` | `k` | Rician K-factor |
| `Nakagami_M_Fading` | `m`, `omega` | Nakagami-m shape/spread |

## Traffic Models

| Class | Parameters | Description |
|---|---|---|
| `Exponential_Traffic` | `average_interval` (s) | Poisson arrivals |
| `Uniform_Traffic` | `max_interval` (s) | Uniform inter-arrival |
| `Constant_Traffic` | `constant_interval`, `standard_deviation` | Near-periodic |
| `Two_State_Markovian_Traffic` | `transition_matrix`, `markov_time` | Bursty traffic |

---

## LoRa PHY Reference

Helper functions available in `lrfhss.link`:

```python
from lrfhss.link import lora_airtime, lora_sensitivity, lora_min_snr, lora_energy

lora_airtime(sf=9, cr=1, payload_size=65, bw=125)   # → 0.3901 seconds
lora_sensitivity(sf=9, bw=125)                        # → -131.25 dBm
lora_min_snr(sf=9)                                    # → -12.5 dB
lora_energy(tx_power_dbm=14, airtime_s=0.39)          # → Joules
```

---

## Acknowledgements

Plots can be made with the [SciencePlots](https://github.com/garrettj403/SciencePlots) library.
