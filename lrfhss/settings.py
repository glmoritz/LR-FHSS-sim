import numpy as np
import warnings
from lrfhss.traffic import *
from lrfhss.fading import *
from lrfhss.pathloss import LogDistance_PathLoss
from lrfhss.link import LinkConfig
import inspect


class Settings():
    """Simulation settings supporting LR-FHSS, LoRa, and relay modes.

    Parameters
    ----------
    Backward-compatible LR-FHSS parameters (original API):
        number_nodes, simulation_time, payload_size, headers,
        header_duration, payloads, threshold, payload_duration, code,
        traffic_class, traffic_param, transceiver_wait, obw, base,
        window_size, window_step, sensitivity, fading_class,
        max_distance, transmission_power, fading_param.

    New parameters:
        link_config : LinkConfig or None
            When provided, replaces the LR-FHSS-specific params above.
            Allows using either 'lrfhss' or 'lora' link types.
        pathloss_class : PathLoss sub-class
            Class to instantiate for path-loss modelling.  Defaults to
            :class:`~lrfhss.pathloss.LogDistance_PathLoss`.
        pathloss_param : dict
            Parameters forwarded to *pathloss_class*.  For
            ``LogDistance_PathLoss`` the valid keys are ``gamma``,
            ``d0``, ``lpld0``, and ``std``.
        number_relays : int
            Number of relay nodes (default 0).
        relay_link_config : LinkConfig or None
            Link configuration relays use for forwarding.  If None,
            relays forward using the same link_config as end devices
            (must not be callable when link_config is callable).
        relay_positions : list of (x,y) tuples or None
            Fixed positions for relay nodes.  If None, relays are
            placed randomly.
        relay_dutycycle_period_s : float or None
            Relay duty-cycle period in seconds. If None, relays forward
            immediately after decode (legacy behavior).
        relay_dutycycle_percent : float or None
            Percentage of each duty-cycle period spent listening.
            Remaining time is used to relay buffered packets.
        lora_channels : int
            Number of LoRa channels at the base station (default 8).
        gamma, d0, lpld0, std : float
            **Deprecated.** Convenience shims that populate
            *pathloss_param* when it is not explicitly provided.  Use
            *pathloss_param* directly instead.
        base_position : tuple
            Gateway (x, y) position (default (0, 0)).
    """

    def __init__(self, number_nodes=80000//8, simulation_time=60*60,
                 payload_size=10, headers=3, header_duration=0.233472,
                 payloads=None, threshold=None, payload_duration=0.1024,
                 code='1/3',
                 traffic_class=Exponential_Traffic,
                 traffic_param={'average_interval': 900},
                 transceiver_wait=0.006472, obw=35, base='core',
                 window_size=2, window_step=0.5, sensitivity=-120,
                 fading_class=No_Fading,
                 min_distance=0, max_distance=2250,
                 transmission_power=14, fading_param={},
                 # ---- new parameters ----
                 link_config=None,
                 pathloss_class=LogDistance_PathLoss,
                 pathloss_param=None,
                 number_relays=0,
                 relay_link_config=None,
                 relay_positions=None,
                 relay_dutycycle_period_s=None,
                 relay_dutycycle_percent=None,
                 lora_channels=8,
                 # legacy shims — kept for backward compat
                 gamma=2.32, d0=1000.0, lpld0=128.95, std=7.8,
                 base_position=(0, 0)):

        self.number_nodes = number_nodes
        self.simulation_time = simulation_time
        self.payload_size = payload_size
        self.sensitivity = sensitivity
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.transmission_power = transmission_power
        self.base = base
        self.window_size = window_size
        self.window_step = window_step
        self.base_position = base_position

        # ---- Relay configuration ----
        self.number_relays = number_relays
        self.relay_positions = relay_positions
        self.relay_dutycycle_period_s = relay_dutycycle_period_s
        self.relay_dutycycle_percent = relay_dutycycle_percent
        self.lora_channels = lora_channels

        # ---- Path-loss model ----
        # If caller passes pathloss_param explicitly, use it.
        # Otherwise, fall back to the legacy gamma/d0/lpld0/std shims.
        if pathloss_param is not None:
            _pl_param = pathloss_param
        else:
            _pl_param = {
                'gamma': gamma,
                'd0':    d0,
                'lpld0': lpld0,
                'std':   std,
            }
        self.pathloss_model = pathloss_class(_pl_param)

        # ---- Fading / Traffic validation ----
        if not issubclass(traffic_class, Traffic):
            warnings.warn('Using an invalid traffic class.')
            exit(1)
        if not issubclass(fading_class, Fading):
            warnings.warn('Using an invalid fading class.')
            exit(1)

        self.traffic_generator = traffic_class(traffic_param)
        self.fading_generator = fading_class(fading_param)

        # ---- Link configuration ----
        if link_config is not None:
            self.link_config = link_config
            if callable(link_config):
                # Callable link_config: per-node configs resolved at runtime.
                # Use legacy params for Base / backward-compat attributes.
                _default_lc = LinkConfig(
                    link_type='lrfhss',
                    payload_size=payload_size,
                    sensitivity=sensitivity,
                    transmission_power=transmission_power,
                    headers=headers,
                    header_duration=header_duration,
                    payload_duration=payload_duration,
                    transceiver_wait=transceiver_wait,
                    code=code,
                    obw=obw,
                    payloads=payloads,
                    threshold=threshold,
                )
            else:
                _default_lc = link_config
        else:
            # Build an LR-FHSS LinkConfig from legacy parameters
            self.link_config = LinkConfig(
                link_type='lrfhss',
                payload_size=payload_size,
                sensitivity=sensitivity,
                transmission_power=transmission_power,
                headers=headers,
                header_duration=header_duration,
                payload_duration=payload_duration,
                transceiver_wait=transceiver_wait,
                code=code,
                obw=obw,
                payloads=payloads,
                threshold=threshold,
            )

        # Attach pathloss_model to link_config if not already set
        if self.link_config.pathloss_model is None:
            self.link_config.pathloss_model = self.pathloss_model

        # Expose legacy attributes from link_config for backward compat
        self.headers = self.link_config.headers
        self.header_duration = self.link_config.header_duration
        self.payload_duration = self.link_config.payload_duration
        self.transceiver_wait = self.link_config.transceiver_wait
        self.obw = self.link_config.obw
        self.payloads = self.link_config.payloads
        self.threshold = self.link_config.threshold
        self.time_on_air = self.link_config.time_on_air

        # ---- Relay link ----
        if relay_link_config is not None:
            self.relay_link_config = relay_link_config
        else:
            # Default: relays forward using the same link as end devices
            self.relay_link_config = self.link_config

        # Attach pathloss_model to relay_link_config if not already set
        if self.relay_link_config.pathloss_model is None:
            self.relay_link_config.pathloss_model = self.pathloss_model
