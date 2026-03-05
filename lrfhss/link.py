"""
Link configuration module for LR-FHSS and conventional LoRa transmissions.

A Link defines HOW a transmission happens between two points:
  - 'lrfhss': multiple frequency-hopping fragments (headers + payloads)
  - 'lora': a single contiguous transmission on one channel

Both link types produce Fragment/Packet objects that share the same
collision and reception machinery in Base.
"""

import math
import numpy as np
import warnings


# ---------------------------------------------------------------------------
# LoRa receiver sensitivity table (measured values)
# Rows: SF7..SF12,  Columns: [SF, 125kHz, 250kHz, 500kHz]  (dBm)
# ---------------------------------------------------------------------------
_SENSI_TABLE = np.array([
    [7,  -126.5,  -124.25, -120.75],
    [8,  -127.25, -126.75, -124.0],
    [9,  -131.25, -128.25, -127.5],
    [10, -132.75, -130.25, -128.75],
    [11, -134.5,  -132.75, -128.75],
    [12, -133.25, -132.25, -132.25],
])

# Minimum SNR (dB) required for demodulation at each SF (SF7..SF12)
_SNR_REQ = np.array([-7.5, -10.0, -12.5, -15.0, -17.5, -20.0])

# ---------------------------------------------------------------------------
# LoRa inter-SF interference isolation thresholds (dB)
#
# D. Croce, M. Gucciardo, S. Mangione, G. Santaromita and I. Tinnirello,
# "Impact of LoRa Imperfect Orthogonality: Analysis of Link-Level
# Performance," in IEEE Communications Letters, vol. 22, no. 4,
# pp. 796-799, April 2018, doi: 10.1109/LCOMM.2018.2797057.
#
# _NON_ORTH_DELTA[i][j] gives the minimum SIR (dB) that a desired signal
# at SF (7+i) needs over an interferer at SF (7+j) to be decoded.
# Diagonal entries (same-SF, i.e. co-SF capture) require +1 dB SIR.
# Off-diagonal entries are negative, meaning the desired signal can be
# weaker than the interferer by that many dB and still survive.
# ---------------------------------------------------------------------------
_NON_ORTH_DELTA = np.array([
    [  1,  -8,  -9,  -9,  -9,  -9],   # desired SF7
    [-11,   1, -11, -12, -13, -13],   # desired SF8
    [-15, -13,   1, -13, -14, -15],   # desired SF9
    [-19, -18, -17,   1, -17, -18],   # desired SF10
    [-22, -22, -21, -20,   1, -20],   # desired SF11
    [-25, -25, -25, -24, -23,   1],   # desired SF12
], dtype=float)

# EU868 LoRa carrier frequencies (Hz)
LORA_CARRIER_FREQUENCIES = np.array([
    867_100_000, 867_300_000, 867_500_000, 867_700_000,
    867_900_000, 868_100_000, 868_300_000, 868_500_000,
])


# ---------------------------------------------------------------------------
# LoRa PHY helper functions
# ---------------------------------------------------------------------------

def lora_sensitivity(sf: int, bw: int) -> float:
    """Return receiver sensitivity (dBm) for a given SF and bandwidth (kHz).

    Parameters
    ----------
    sf : int
        Spreading factor (7-12).
    bw : int
        Bandwidth in kHz (125, 250 or 500).

    Returns
    -------
    float
        Sensitivity threshold in dBm.
    """
    bw_col = {125: 1, 250: 2, 500: 3}
    if sf < 7 or sf > 12:
        raise ValueError(f"SF must be 7-12, got {sf}")
    if bw not in bw_col:
        raise ValueError(f"BW must be 125, 250, or 500 kHz, got {bw}")
    return float(_SENSI_TABLE[sf - 7, bw_col[bw]])


def lora_min_snr(sf: int) -> float:
    """Return minimum SNR (dB) required for demodulation at a given SF.

    Parameters
    ----------
    sf : int
        Spreading factor (7-12).

    Returns
    -------
    float
        Minimum required SNR in dB.
    """
    if sf < 7 or sf > 12:
        raise ValueError(f"SF must be 7-12, got {sf}")
    return float(_SNR_REQ[sf - 7])


def lora_non_orth_delta(sf_desired: int, sf_interferer: int) -> float:
    """Return the minimum SIR (dB) for *sf_desired* to survive *sf_interferer*.

    A received signal at *sf_desired* can tolerate an interferer at
    *sf_interferer* as long as the Signal-to-Interferer Ratio (SIR)
    exceeds the returned threshold.  Negative thresholds mean the
    desired signal can be weaker than the interferer by that many dB.

    Source: Croce et al., IEEE Comms Letters, vol. 22, no. 4, 2018.

    Parameters
    ----------
    sf_desired : int
        Spreading factor of the desired signal (7-12).
    sf_interferer : int
        Spreading factor of the interfering signal (7-12).

    Returns
    -------
    float
        Minimum SIR in dB.
    """
    if sf_desired < 7 or sf_desired > 12:
        raise ValueError(f"sf_desired must be 7-12, got {sf_desired}")
    if sf_interferer < 7 or sf_interferer > 12:
        raise ValueError(f"sf_interferer must be 7-12, got {sf_interferer}")
    return float(_NON_ORTH_DELTA[sf_desired - 7, sf_interferer - 7])


def lora_airtime(sf: int, cr: int, payload_size: int, bw: int) -> float:
    """Compute conventional LoRa packet airtime **in seconds**.

    Implements the formula from the LoRa Design Guide (Semtech).

    Parameters
    ----------
    sf : int
        Spreading factor (7-12).
    cr : int
        Coding rate denominator minus 4, i.e. 1 for 4/5, 2 for 4/6,
        3 for 4/7, 4 for 4/8.
    payload_size : int
        Application payload size in bytes.
    bw : int
        Bandwidth in kHz (125, 250, 500).

    Returns
    -------
    float
        Total airtime in **seconds**.
    """
    H = 0   # implicit header disabled (0) or enabled (1)
    DE = 0  # low data-rate optimization

    if bw == 125 and sf in (11, 12):
        DE = 1  # mandated for BW125 with SF11/SF12
    if sf == 6:
        H = 1   # implicit header required for SF6

    Npream = 8  # preamble symbols

    Tsym = (2.0 ** sf) / bw                         # ms per symbol
    Tpream = (Npream + 4.25) * Tsym                  # ms

    payload_symb = 8 + max(
        math.ceil((8.0 * payload_size - 4.0 * sf + 28 + 16 - 20 * H)
                  / (4.0 * (sf - 2 * DE))) * (cr + 4),
        0,
    )
    Tpayload = payload_symb * Tsym                   # ms

    return (Tpream + Tpayload) / 1000.0              # convert ms -> seconds


def lora_energy(tx_power_dbm: float, airtime_s: float) -> float:
    """Compute transmission energy in Joules.

    Parameters
    ----------
    tx_power_dbm : float
        Transmit power in dBm.
    airtime_s : float
        Airtime in seconds.

    Returns
    -------
    float
        Energy in Joules.
    """
    tx_power_w = 10.0 ** (tx_power_dbm / 10.0) / 1000.0
    return tx_power_w * airtime_s


# ---------------------------------------------------------------------------
# LinkConfig
# ---------------------------------------------------------------------------

class LinkConfig:
    """Encapsulates all parameters that define a link type.

    Parameters
    ----------
    link_type : str
        ``'lrfhss'`` or ``'lora'``.

    LR-FHSS specific (ignored when link_type='lora'):
        headers, header_duration, payload_duration, transceiver_wait,
        code, payloads, threshold.

    LR-FHSS channel / grid parameters:
        ocw_hz : float
            Occupied Channel Width in Hz.  This is the total RF bandwidth
            allocated to LR-FHSS (default 1_523_000 Hz = 1.523 MHz, the
            US/FCC 915 MHz band allocation).
        obw_hz : float
            Occupied Bandwidth of a single sub-channel in Hz (default
            488 Hz, fixed by the LR-FHSS standard).
        grid_spacing_hz : float
            Minimum frequency separation between two consecutive hops
            within the same grid (default 25_400 Hz = 25.4 kHz).  This
            regulatory constraint determines the grid structure.

        Derived (computed automatically):
            total_channels : int
                ``ocw_hz / obw_hz`` (e.g. 3 120).
            num_grids : int
                ``grid_spacing_hz / obw_hz`` (e.g. 52).
            channels_per_grid : int
                ``total_channels / num_grids`` (e.g. 60).

        obw : int  (deprecated / backward-compatible)
            If set explicitly **and** none of ocw_hz/obw_hz/grid_spacing_hz
            are provided, the legacy single-grid mode is used with *obw*
            channels.  Otherwise it is overridden by the grid-based
            computation.

    LoRa specific (ignored when link_type='lrfhss'):
        sf, bw, cr, lora_channels.

    Common:
        payload_size, sensitivity, transmission_power, pathloss_model.

    pathloss_model : PathLoss or None
        Pluggable path-loss model used for every fragment on this link.
        Must be an instance of a :class:`~lrfhss.lrfhss_core.PathLoss`
        sub-class (e.g. :class:`~lrfhss.pathloss.LogDistance_PathLoss`).
        If *None* the model must be supplied explicitly when creating a
        :class:`~lrfhss.lrfhss_core.Node` or will raise at simulation
        time.
    """

    def __init__(
        self,
        link_type: str = 'lrfhss',
        # --- common ---
        payload_size: int = 10,
        sensitivity: float = -120.0,
        transmission_power: float = 14.0,
        pathloss_model=None,
        # --- LR-FHSS specific ---
        headers: int = 3,
        header_duration: float = 0.233472,
        payload_duration: float = 0.1024,
        transceiver_wait: float = 0.006472,
        code: str = '1/3',
        obw: int = None,
        payloads: int = None,
        threshold: int = None,
        # --- LR-FHSS channel / grid parameters ---
        ocw_hz: float = 1_523_000,
        obw_hz: float = 488,
        grid_spacing_hz: float = 25_400,
        # --- LoRa specific ---
        sf: int = 9,
        bw: int = 125,
        cr: int = 1,
        lora_channels: int = 8,
    ):
        self.link_type = link_type.lower()
        if self.link_type not in ('lrfhss', 'lora'):
            raise ValueError(f"link_type must be 'lrfhss' or 'lora', got '{link_type}'")

        # Common
        self.payload_size = payload_size
        self.sensitivity = sensitivity
        self.transmission_power = transmission_power
        self.pathloss_model = pathloss_model

        if self.link_type == 'lrfhss':
            self._init_lrfhss(headers, header_duration, payload_duration,
                              transceiver_wait, code, obw, payloads, threshold,
                              payload_size, ocw_hz, obw_hz, grid_spacing_hz)
        else:
            self._init_lora(sf, bw, cr, lora_channels, payload_size)

    # ---- LR-FHSS initialisation ----
    def _init_lrfhss(self, headers, header_duration, payload_duration,
                     transceiver_wait, code, obw, payloads, threshold,
                     payload_size, ocw_hz, obw_hz, grid_spacing_hz):
        self.headers = headers
        self.header_duration = header_duration
        self.payload_duration = payload_duration
        self.transceiver_wait = transceiver_wait

        # ---- OCW / OBW / Grid channel structure ----
        self.ocw_hz = ocw_hz
        self.obw_hz = obw_hz
        self.grid_spacing_hz = grid_spacing_hz

        # Derived grid parameters
        self.total_channels = int(ocw_hz / obw_hz)          # e.g. 3120
        self.num_grids = int(grid_spacing_hz / obw_hz)      # e.g. 52
        self.channels_per_grid = int(self.total_channels     # e.g. 60
                                     / self.num_grids)

        if obw is not None:
            # Legacy single-grid mode: caller explicitly passed a
            # channel count.  Treat it as a flat pool with one grid.
            warnings.warn(
                "LinkConfig: 'obw' is deprecated for LR-FHSS.  Use "
                "'ocw_hz', 'obw_hz', and 'grid_spacing_hz' instead.  "
                "Falling back to legacy single-grid mode.",
                DeprecationWarning, stacklevel=3,
            )
            self.total_channels = obw
            self.num_grids = 1
            self.channels_per_grid = obw

        # obw exposed for Base channel allocation (= total channel pool)
        self.obw = self.total_channels

        if payloads is not None:
            self.payloads = payloads
        else:
            self.payloads = self._compute_payloads(code, payload_size)

        if threshold is not None:
            self.threshold = threshold
        else:
            self.threshold = self._compute_threshold(code, self.payloads)

        self.time_on_air = (header_duration * headers
                            + payload_duration * self.payloads
                            + transceiver_wait)

        # Not applicable but set for uniform interface
        self.sf = None
        self.bw = None
        self.cr = None
        self.lora_channels = None

    # ---- LoRa initialisation ----
    def _init_lora(self, sf, bw, cr, lora_channels, payload_size):
        self.sf = sf
        self.bw = bw
        self.cr = cr
        self.lora_channels = lora_channels

        self.time_on_air = lora_airtime(sf, cr, payload_size, bw)
        self.lora_sensitivity = lora_sensitivity(sf, bw)
        self.lora_min_snr = lora_min_snr(sf)

        # Not applicable but set for uniform interface
        self.headers = 0
        self.payloads = 0
        self.header_duration = 0.0
        self.payload_duration = 0.0
        self.transceiver_wait = 0.0
        self.obw = lora_channels  # channel count used by Packet
        self.threshold = 0

    # ---- Code-rate helpers (same logic as Settings) ----
    @staticmethod
    def _compute_payloads(code: str, payload_size: int) -> int:
        match code:
            case '1/3':
                return int(np.ceil((payload_size + 3) / 2))
            case '2/3':
                return int(np.ceil((payload_size + 3) / 4))
            case '5/6':
                return int(np.ceil((payload_size + 3) / 5))
            case '1/2':
                return int(np.ceil((payload_size + 3) / 3))
            case _:
                warnings.warn(f"code='{code}' invalid, using '1/3'.")
                return int(np.ceil((payload_size + 3) / 2))

    @staticmethod
    def _compute_threshold(code: str, payloads: int) -> int:
        match code:
            case '1/3':
                return int(np.ceil(payloads / 3))
            case '2/3':
                return int(np.ceil((2 * payloads) / 3))
            case '5/6':
                return int(np.ceil((5 * payloads) / 6))
            case '1/2':
                return int(np.ceil(payloads / 2))
            case _:
                return int(np.ceil(payloads / 3))

    def __repr__(self):
        if self.link_type == 'lrfhss':
            return (f"LinkConfig(type=lrfhss, headers={self.headers}, "
                    f"payloads={self.payloads}, threshold={self.threshold}, "
                    f"OCW={self.ocw_hz/1e6:.3f}MHz, "
                    f"OBW={self.obw_hz}Hz, "
                    f"grids={self.num_grids}, "
                    f"ch/grid={self.channels_per_grid}, "
                    f"total_ch={self.total_channels}, "
                    f"ToA={self.time_on_air:.4f}s)")
        else:
            return (f"LinkConfig(type=lora, SF={self.sf}, BW={self.bw}kHz, "
                    f"CR=4/{self.cr+4}, channels={self.lora_channels}, "
                    f"ToA={self.time_on_air:.4f}s)")
