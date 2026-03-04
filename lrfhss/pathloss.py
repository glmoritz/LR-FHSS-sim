"""
Pluggable path-loss models for LR-FHSS and LoRa simulations.

All models sub-class :class:`~lrfhss.lrfhss_core.PathLoss` and must
implement :meth:`path_loss_db`, which returns the **one-way path loss
in dB** for a transmitter-receiver separation in metres.

The returned value feeds directly into the link budget::

    rssi_dBm = tx_power_dBm - path_loss_db(distance) + fading_dB

Usage example
-------------
::

    from lrfhss.pathloss import LogDistance_PathLoss
    from lrfhss.link import LinkConfig

    pl = LogDistance_PathLoss({'gamma': 2.32, 'd0': 1000.0,
                               'lpld0': 128.95, 'std': 7.8})
    lc = LinkConfig(link_type='lrfhss', pathloss_model=pl)
"""

import math
import warnings
import numpy as np
from lrfhss.lrfhss_core import PathLoss


# ---------------------------------------------------------------------------
# Log-distance path loss  (default model, mirrors conventional LoRa setup)
# ---------------------------------------------------------------------------

class LogDistance_PathLoss(PathLoss):
    """Log-distance path loss model with log-normal shadowing.

    .. math::

        PL(d) = PL(d_0) + 10 \\gamma \\log_{10}\\!\\left(\\frac{d}{d_0}\\right)
                + X_\\sigma

    where :math:`X_\\sigma \\sim \\mathcal{N}(0, \\sigma^2)`.

    Parameters
    ----------
    pathloss_param : dict, optional
        Accepted keys:

        ``gamma`` : float
            Path-loss exponent (default ``2.32``).
        ``d0`` : float
            Reference distance in metres (default ``1000.0``).
        ``lpld0`` : float
            Path loss at *d0* in dB (default ``128.95``).
        ``std`` : float
            Log-normal shadowing standard deviation in dB (default ``7.8``).

        All keys are optional; missing keys produce a warning and fall
        back to the default values.
    """

    def __init__(self, pathloss_param=None):
        if pathloss_param is None:
            pathloss_param = {}
        super().__init__(pathloss_param)

        if 'gamma' not in self.pathloss_param:
            warnings.warn(
                'LogDistance_PathLoss: "gamma" missing, using 2.32.',
                stacklevel=2,
            )
            self.pathloss_param['gamma'] = 2.32

        if 'd0' not in self.pathloss_param:
            warnings.warn(
                'LogDistance_PathLoss: "d0" missing, using 1000.0 m.',
                stacklevel=2,
            )
            self.pathloss_param['d0'] = 1000.0

        if 'lpld0' not in self.pathloss_param:
            warnings.warn(
                'LogDistance_PathLoss: "lpld0" missing, using 128.95 dB.',
                stacklevel=2,
            )
            self.pathloss_param['lpld0'] = 128.95

        if 'std' not in self.pathloss_param:
            warnings.warn(
                'LogDistance_PathLoss: "std" missing, using 7.8 dB.',
                stacklevel=2,
            )
            self.pathloss_param['std'] = 7.8

    def path_loss_db(self, distance: float) -> float:
        """Return path loss in dB for *distance* metres.

        A minimum distance of 1 m is enforced to avoid ``log10(0)``.

        Parameters
        ----------
        distance : float
            Transmitter–receiver separation in metres.

        Returns
        -------
        float
            Path loss in dB.
        """
        d = max(distance, 1.0)
        gamma = self.pathloss_param['gamma']
        d0    = self.pathloss_param['d0']
        lpld0 = self.pathloss_param['lpld0']
        std   = self.pathloss_param['std']
        shadowing = np.random.normal(0.0, std) if std > 0 else 0.0
        return 10.0 * gamma * math.log10(d / d0) + lpld0 + shadowing

    def __repr__(self):
        p = self.pathloss_param
        return (f"LogDistance_PathLoss(gamma={p['gamma']}, d0={p['d0']} m, "
                f"lpld0={p['lpld0']} dB, std={p['std']} dB)")


# ---------------------------------------------------------------------------
# Free-space path loss  (Friis, for reference / comparison)
# ---------------------------------------------------------------------------

class FreeSpace_PathLoss(PathLoss):
    """Friis free-space path loss model.

    .. math::

        PL(d) = 20 \\log_{10}(d) + 20 \\log_{10}(f) - 147.55

    Parameters
    ----------
    pathloss_param : dict, optional
        Accepted keys:

        ``frequency`` : float
            Carrier frequency in Hz (default ``868e6`` — EU868 band).
    """

    def __init__(self, pathloss_param=None):
        if pathloss_param is None:
            pathloss_param = {}
        super().__init__(pathloss_param)

        if 'frequency' not in self.pathloss_param:
            warnings.warn(
                'FreeSpace_PathLoss: "frequency" missing, using 868e6 Hz.',
                stacklevel=2,
            )
            self.pathloss_param['frequency'] = 868e6

    def path_loss_db(self, distance: float) -> float:
        """Return Friis free-space path loss in dB.

        Parameters
        ----------
        distance : float
            Transmitter–receiver separation in metres.

        Returns
        -------
        float
            Path loss in dB.
        """
        d = max(distance, 1.0)
        f = self.pathloss_param['frequency']
        # Friis: PL = 20*log10(4*pi*d*f/c)
        #            = 20*log10(d) + 20*log10(f) + 20*log10(4*pi/c)
        # 20*log10(4*pi/c) = 20*log10(4*pi/3e8) ≈ -147.55 dB
        return 20.0 * math.log10(d) + 20.0 * math.log10(f) - 147.55

    def __repr__(self):
        return f"FreeSpace_PathLoss(frequency={self.pathloss_param['frequency'] / 1e6:.1f} MHz)"
