import random
import math
import numpy as np
from abc import ABC, abstractmethod
from collections import deque
from lrfhss.link import lora_sensitivity, lora_min_snr, lora_non_orth_delta

        
class Fragment():
    """A single transmission fragment (header, payload, or lora).

    For LR-FHSS fragments the LoRa-specific attributes (sf, bw, cr) are None.
    For conventional LoRa the fragment type is 'lora' and sf/bw/cr are set,
    enabling link-aware reception checks in Base.finish_fragment().
    """
    def __init__(self, type, duration, channel, packet, intensity,
                 sf=None, bw=None, cr=None):
        self.packet = packet
        self.duration = duration
        self.success = 0
        self.transmitted = 0
        self.type = type          # 'header', 'payload', or 'lora'
        self.channel = channel
        self.timestamp = 0
        self.id = id(self)
        self.collided = []
        self.intensity = intensity
        # LoRa-specific (None for LR-FHSS fragments)
        self.sf = sf
        self.bw = bw
        self.cr = cr

class Packet():
    """Represents a transmission from a node.

    Can be created in two ways:
    1. Legacy positional args (backward compatible, always LR-FHSS).
    2. Via a LinkConfig object which determines the link type:
       - 'lrfhss': multiple frequency-hopping fragments (headers + payloads).
       - 'lora': a single fragment covering the full LoRa airtime.

    Parameters
    ----------
    node_id : hashable
        Identifier of the transmitting node.
    link_config : LinkConfig, optional
        If provided, drives fragment creation. The remaining positional
        parameters are ignored.
    obw, headers, payloads, header_duration, payload_duration :
        Legacy LR-FHSS parameters (used when link_config is None).
    fading_generator : Fading
        Generates per-fragment fading intensity values.
    """
    def __init__(self, node_id, obw=None, headers=None, payloads=None,
                 header_duration=None, payload_duration=None,
                 fading_generator=None, *, link_config=None):
        self.id = id(self)
        self.node_id = node_id
        self.index_transmission = 0
        self.success = 0
        self.fragments = []
        self.threshold = None  # per-packet decode threshold

        if link_config is not None:
            self.link_type = link_config.link_type
            self.threshold = getattr(link_config, 'threshold', None)
            if link_config.link_type == 'lora':
                self._build_lora(link_config, fading_generator)
            else:
                self._build_lrfhss(
                    link_config.obw, link_config.headers,
                    link_config.payloads, link_config.header_duration,
                    link_config.payload_duration, fading_generator)
        else:
            # Legacy path – always LR-FHSS
            self.link_type = 'lrfhss'
            self._build_lrfhss(obw, headers, payloads,
                               header_duration, payload_duration,
                               fading_generator)

    def _build_lrfhss(self, obw, headers, payloads, header_duration,
                      payload_duration, fading_generator):
        """Create frequency-hopping header + payload fragments."""
        self.channels = random.choices(range(obw), k=headers + payloads)
        for h in range(headers):
            intensity = fading_generator.fading_function()
            self.fragments.append(
                Fragment('header', header_duration, self.channels[h],
                         self.id, intensity))
        for p in range(payloads):
            intensity = fading_generator.fading_function()
            self.fragments.append(
                Fragment('payload', payload_duration,
                         self.channels[p + h + 1], self.id, intensity))

    def _build_lora(self, link_config, fading_generator):
        """Create a single LoRa fragment spanning the full airtime."""
        channel = random.randrange(link_config.lora_channels)
        self.channels = [channel]
        intensity = fading_generator.fading_function()
        self.fragments.append(
            Fragment('lora', link_config.time_on_air, channel,
                     self.id, intensity,
                     sf=link_config.sf, bw=link_config.bw,
                     cr=link_config.cr))

    def next(self):
        self.index_transmission+=1
        try:
            return self.fragments[self.index_transmission-1]
        except:
            return False

    def clone(self):
        """Create an independent copy with the same identity.

        The clone shares the same ``id`` and ``node_id`` as the
        original but has fresh fragments with independent collision
        and success state.  Used to give each receiver its own
        tracking state when broadcasting to multiple receivers.
        """
        c = object.__new__(Packet)
        c.id = self.id
        c.node_id = self.node_id
        c.index_transmission = 0
        c.success = 0
        c.link_type = self.link_type
        c.threshold = self.threshold
        c.channels = list(self.channels)
        c.fragments = []
        for f in self.fragments:
            cf = Fragment(f.type, f.duration, f.channel, c.id,
                         f.intensity, sf=f.sf, bw=f.bw, cr=f.cr)
            c.fragments.append(cf)
        return c

# Instead of grid selection, we consider one grid of obw (usually 35 for EU) channels, as it is faster to simulate and extrapolate the number.
# Later we can implement the grid selection in case of interest of studying it.
#    def new_channels(self, obw, fragments):
#        self.channels = random.sample(range(obw), fragments)


class Traffic(ABC):
    @abstractmethod
    def __init__(self, traffic_param):
        self.traffic_param = traffic_param

    @abstractmethod
    def traffic_function(self):
        pass

class Fading(ABC):
    @abstractmethod
    def __init__(self, fading_param):
        self.fading_param = fading_param

    @abstractmethod
    def fading_function(self):
        pass


class PathLoss(ABC):
    """Abstract base class for pluggable path-loss models.

    Sub-classes must implement :meth:`path_loss_db`, which returns the
    one-way path loss **in dB** for a given distance in metres.  The
    returned value is used as::

        rssi_dBm = tx_power_dBm - path_loss_db(distance)

    Parameters
    ----------
    pathloss_param : dict
        Model-specific parameters (passed through to the sub-class).
    """

    @abstractmethod
    def __init__(self, pathloss_param):
        self.pathloss_param = pathloss_param

    @abstractmethod
    def path_loss_db(self, distance: float) -> float:
        """Return path loss in dB for *distance* metres.

        Parameters
        ----------
        distance : float
            Distance between transmitter and receiver in metres.

        Returns
        -------
        float
            Path loss in dB (positive value → attenuation).
        """
        pass


class Node():
    """A network end-device that transmits packets.

    Supports both LR-FHSS and conventional LoRa link types.  Uses 2D
    ``(x, y)`` positioning; distance to each receiver is computed
    dynamically during transmission.

    The node has **no knowledge of relays**.  It simply broadcasts
    fragments to all receivers (sinks and relays alike).  Each receiver
    independently tracks collisions, evaluates reception, and decides
    whether to decode the packet.

    Construction
    ------------
    Legacy (backward compatible)::

        Node(obw, headers, payloads, header_duration, payload_duration,
             transceiver_wait, traffic_gen, fading_gen, max_dist, tx_power)

    New (via LinkConfig)::

        Node(traffic_generator=t, fading_generator=f,
             max_distance=2250, transmission_power=14,
             link_config=my_link)
    """

    def __init__(self, obw=None, headers=None, payloads=None,
                 header_duration=None, payload_duration=None,
                 transceiver_wait=0, traffic_generator=None,
                 fading_generator=None, max_distance=2250,
                 transmission_power=14, *,
                 min_distance=0, link_config=None, position=None,
                 base_position=(0, 0), pathloss_model=None):
        self.id = id(self)
        self.transmitted = 0
        self.traffic_generator = traffic_generator
        self.fading_generator = fading_generator
        self.transmission_power = transmission_power
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.link_config = link_config
        # Path-loss model: prefer explicit arg, then link_config, then None
        if pathloss_model is not None:
            self.pathloss_model = pathloss_model
        elif link_config is not None and getattr(link_config, 'pathloss_model', None) is not None:
            self.pathloss_model = link_config.pathloss_model
        else:
            self.pathloss_model = None

        # -- 2D position ------------------------------------------------
        if position is not None:
            self.x, self.y = position[0], position[1]
        else:
            # Uniform random point inside circle of ring of radius [min_distance, max_distance]
            angle = random.uniform(0, 2 * math.pi)
            r = min_distance + (max_distance - min_distance) * random.uniform(0, 1)
            self.x = r * math.cos(angle)
            self.y = r * math.sin(angle)

        self.base_position = base_position
        bx, by = base_position
        self.distance = math.sqrt((self.x - bx) ** 2
                                  + (self.y - by) ** 2)

        # -- Packet-creation parameters ---------------------------------
        if link_config is not None:
            self.obw = link_config.obw
            self.headers = link_config.headers
            self.payloads = link_config.payloads
            self.header_duration = link_config.header_duration
            self.payload_duration = link_config.payload_duration
            self.transceiver_wait = link_config.transceiver_wait
        else:
            self.obw = obw
            self.headers = headers
            self.payloads = payloads
            self.header_duration = header_duration
            self.payload_duration = payload_duration
            self.transceiver_wait = transceiver_wait

        # -- Initial packet ---------------------------------------------
        if self.link_config is not None or self.obw is not None:
            self.packet = self._create_packet()
        else:
            self.packet = None  # bare node; call set_link_config() next

    def set_link_config(self, link_config):
        """Assign a LinkConfig after construction (e.g. per-node configs).

        Updates all packet-creation fields and creates the initial packet.
        Also picks up the pathloss_model from the new LinkConfig.
        """
        self.link_config = link_config
        self.obw = link_config.obw
        self.headers = link_config.headers
        self.payloads = link_config.payloads
        self.header_duration = link_config.header_duration
        self.payload_duration = link_config.payload_duration
        self.transceiver_wait = link_config.transceiver_wait
        if getattr(link_config, 'pathloss_model', None) is not None:
            self.pathloss_model = link_config.pathloss_model
        self.packet = self._create_packet()

    # -- helpers --------------------------------------------------------

    def _create_packet(self):
        """Create a new packet using either LinkConfig or legacy params."""
        if self.link_config is not None:
            return Packet(self.id, fading_generator=self.fading_generator,
                          link_config=self.link_config)
        return Packet(self.id, self.obw, self.headers, self.payloads,
                      self.header_duration, self.payload_duration,
                      self.fading_generator)

    def distance_to(self, x, y):
        """Euclidean distance from this node to an (x, y) point."""
        return math.sqrt((self.x - x) ** 2 + (self.y - y) ** 2)

    def next_transmission(self):
        return self.traffic_generator.traffic_function()

    def end_of_transmission(self):
        self.packet = self._create_packet()

    # -- main transmit process ------------------------------------------

    def transmit(self, env, receivers):
        """SimPy process: periodically broadcast packets to all receivers.

        Each receiver (sink or relay) receives its own independent
        clone of every packet, with separate collision tracking.
        The node is completely unaware of receiver types.

        Parameters
        ----------
        env : simpy.Environment
        receivers : Base or list[Base]
            All receivers (sinks and relays).  A single Base is
            accepted for backward compatibility.
        """
        if not isinstance(receivers, list):
            receivers = [receivers]

        while True:
            yield env.timeout(self.next_transmission())
            self.transmitted += 1

            # Per-receiver independent packet clones
            rx_packets = {}
            for rx in receivers:
                clone = self.packet.clone()
                rx.add_packet(clone)
                rx_packets[rx] = clone

            first_payload = False
            for frag_idx, master_frag in enumerate(
                    self.packet.fragments):
                if not first_payload and master_frag.type == 'payload':
                    first_payload = True
                    yield env.timeout(self.transceiver_wait)

                # -- fragment start: register at each active receiver --
                for rx in receivers:
                    if getattr(rx, '_is_transmitting', False):
                        continue
                    if not getattr(rx, '_is_listening', True):
                        continue
                    rx_frag = rx_packets[rx].fragments[frag_idx]
                    rx_frag.timestamp = env.now
                    # Store per-fragment reception context so the
                    # receiver can compute interferer RSSI later
                    # (needed for LoRa inter-SF interference model).
                    rx_frag.rx_distance = self.distance_to(rx.x, rx.y)
                    rx_frag.tx_power = self.transmission_power
                    rx_frag._pathloss_model = self.pathloss_model
                    rx.check_collision(rx_frag)
                    rx.receive_packet(rx_frag)

                yield env.timeout(master_frag.duration)

                # -- fragment end: finalize at each active receiver ----
                for rx in receivers:
                    if getattr(rx, '_is_transmitting', False):
                        continue
                    if not getattr(rx, '_is_listening', True):
                        continue
                    rx_frag = rx_packets[rx].fragments[frag_idx]
                    dist = self.distance_to(rx.x, rx.y)
                    rx.finish_fragment(rx_frag, dist,
                                       self.transmission_power,
                                       self.pathloss_model)
                    rx_pkt = rx_packets[rx]
                    if rx_pkt.success == 0:
                        rx.try_decode(rx_pkt, env.now)

            self.end_of_transmission()

class Base():
    """Gateway / base station receiver.

    Handles reception of both LR-FHSS and conventional LoRa fragments
    using the same channel-based collision tracking.

    Parameters
    ----------
    obw : int
        Number of LR-FHSS channels (OBW).  LoRa channels are allocated
        above this range automatically.
    threshold : int
        Minimum number of successful payload fragments for LR-FHSS decode.
    sensitivity : float
        LR-FHSS receiver sensitivity (dBm).
    lora_channels : int
        Number of conventional LoRa channels (default 0 = no LoRa).

    .. deprecated::
        The ``gamma``, ``d0``, ``lpld0``, and ``std`` keyword arguments are
        no longer used.  Path-loss is now configured via the
        :class:`~lrfhss.pathloss.PathLoss` model passed per-fragment
        through :meth:`finish_fragment`.  The arguments are silently
        accepted for backward compatibility but have no effect.
    """
    def __init__(self, obw, threshold, sensitivity, *,
                 lora_channels=0, position=(0, 0),
                 # --- kept for backward compat only, no longer used ---
                 gamma=None, d0=None, lpld0=None, std=None):
        if any(v is not None for v in (gamma, d0, lpld0, std)):
            import warnings
            warnings.warn(
                "Base: 'gamma', 'd0', 'lpld0', 'std' are deprecated and "
                "have no effect.  Configure path loss via the pathloss_model "
                "on LinkConfig (or Node) instead.",
                DeprecationWarning, stacklevel=2,
            )
        self.id = id(self)
        self.obw = obw
        self.lora_channels = lora_channels
        self.transmitting = {}
        # LR-FHSS channels: 0 .. obw-1
        for channel in range(obw):
            self.transmitting[channel] = []
        # LoRa channels: obw .. obw+lora_channels-1
        for channel in range(obw, obw + lora_channels):
            self.transmitting[channel] = []
        self.packets_received = {}
        self.decoded_packets = set()  # Track decoded packet keys (dedup)
        self.threshold = threshold
        self.sensitivity = sensitivity
        # 2D position (used for distance computation by Node.transmit)
        self.x, self.y = position
        self.base_type = 'sink'      

    def _packet_dedup_key(self, packet):
        """Return a stable dedup key for sink-side packet accounting."""
        pkt_id = getattr(packet, 'original_packet_id', packet.id)
        return (packet.node_id, pkt_id)

    def _compute_lora_rssi(self, fragment, distance, tx_power):
        """Compute and cache RSSI (dBm) for a LoRa fragment.

        Used by ``_finish_lora_fragment`` to evaluate the SIR against
        colliding fragments.  Result is stored on the fragment as
        ``fragment.rssi`` so it is only computed once.
        """
        if hasattr(fragment, 'rssi') and fragment.rssi is not None:
            return fragment.rssi
        pathloss_model = getattr(fragment, '_pathloss_model', None)
        if pathloss_model is None:
            # Fall back: use distance^2 free-space approximation
            pl = 20.0 * math.log10(max(distance, 1.0)) + 40.0
        else:
            pl = pathloss_model.path_loss_db(distance)
        rssi = tx_power - pl
        if getattr(fragment, 'intensity', 1) > 0:
            rssi += 20.0 * math.log10(fragment.intensity)
        fragment.rssi = rssi
        return rssi

    def add_packet(self, packet):
        pass

    def add_node(self, id):
        self.packets_received[id] = 0

    def receive_packet(self, fragment):
        # Lazily create channel bucket if it doesn't exist yet
        if fragment.channel not in self.transmitting:
            self.transmitting[fragment.channel] = []
        self.transmitting[fragment.channel].append(fragment)

    def finish_fragment(self, fragment, distance, transmission_power,
                        pathloss_model=None):
        """Mark a fragment as finished and determine success.

        For LR-FHSS and LoRa fragments alike, the path loss is evaluated
        via *pathloss_model* (a :class:`PathLoss` instance).  Both link
        types now work entirely in the **dB domain**:

        .. code-block:: text

            rssi [dBm] = tx_power [dBm]
                         - path_loss_db(distance) [dB]
                         + fading_gain [dB]          # 20·log10(intensity)

        For LoRa, an additional SNR check against the per-SF minimum is
        applied.

        If the fragment was cancelled (e.g. half-duplex clear), this
        method returns early and marks it as transmitted but failed.

        Parameters
        ----------
        fragment : Fragment
        distance : float
            Transmitter–receiver distance in metres.
        transmission_power : float
            Transmit power in dBm.
        pathloss_model : PathLoss or None
            Path-loss model to use.  If *None* a :class:`ValueError` is
            raised — every link must carry a pathloss_model.
        """
        # Safety: fragment may have been cleared by half-duplex cancel
        ch_list = self.transmitting.get(fragment.channel, [])
        if fragment not in ch_list:
            fragment.transmitted = 1
            return

        if pathloss_model is None:
            raise ValueError(
                "finish_fragment: pathloss_model is None.  "
                "Set a PathLoss instance on your LinkConfig (or Node)."
            )

        ch_list.remove(fragment)

        if fragment.type == 'lora':
            self._finish_lora_fragment(fragment, distance,
                                       transmission_power, pathloss_model)
        else:
            self._finish_lrfhss_fragment(fragment, distance,
                                          transmission_power, pathloss_model)

    def _finish_lrfhss_fragment(self, fragment, distance, transmission_power,
                                 pathloss_model):
        """LR-FHSS fragment success check — fully in the dB domain.

        RSSI is computed as::

            rssi = tx_power - path_loss_db(d) + 20·log10(intensity)

        and compared against the receiver sensitivity threshold.
        """
        path_loss = pathloss_model.path_loss_db(distance)
        rssi = transmission_power - path_loss     # dBm

        # Apply fading as a dB gain (intensity is a linear amplitude factor)
        if fragment.intensity > 0:
            rssi += 20.0 * math.log10(fragment.intensity)

        if len(fragment.collided) == 0 and rssi > self.sensitivity:
            fragment.success = 1
        fragment.transmitted = 1

    def _finish_lora_fragment(self, fragment, distance, transmission_power,
                               pathloss_model):
        """LoRa fragment success check — dB domain, per-SF sensitivity/SNR.

        Uses the same path-loss model as LR-FHSS.  Additionally checks
        the minimum SNR requirement for the fragment's spreading factor.
        """
        path_loss = pathloss_model.path_loss_db(distance)
        rssi = transmission_power - path_loss     # dBm

        # Apply fading as a dB gain (intensity is a linear amplitude factor)
        if fragment.intensity > 0:
            rssi += 20.0 * math.log10(fragment.intensity)

        # Noise floor for the LoRa bandwidth
        noise_floor = -174.0 + 10.0 * np.log10(fragment.bw * 1e3)  # dBm
        snr_db = rssi - noise_floor

        # Per-SF sensitivity and SNR requirements
        min_sensi = lora_sensitivity(fragment.sf, fragment.bw)
        min_snr = lora_min_snr(fragment.sf)

        if rssi <= min_sensi or snr_db <= min_snr:
            fragment.transmitted = 1
            return

        # --- Inter-SF interference check (Croce et al. 2018) ---
        # For each collider on the same channel, compute the SIR and
        # compare against the non-orthogonal isolation threshold.
        survived = True
        for collider in fragment.collided:
            if collider.type != 'lora':
                # LR-FHSS collider on same channel -> treat as fatal
                survived = False
                break
            # Compute (or reuse) collider's RSSI at this receiver
            c_rssi = getattr(collider, 'rssi', None)
            if c_rssi is None:
                c_dist = getattr(collider, 'rx_distance', None)
                c_txp = getattr(collider, 'tx_power', None)
                if c_dist is not None and c_txp is not None:
                    c_rssi = self._compute_lora_rssi(
                        collider, c_dist, c_txp)
                else:
                    # Cannot determine interferer strength -> assume fatal
                    survived = False
                    break
            sir_db = rssi - c_rssi  # dB
            threshold = lora_non_orth_delta(fragment.sf, collider.sf)
            if sir_db < threshold:
                survived = False
                break

        if survived:
            fragment.success = 1
        fragment.transmitted = 1

    def check_collision(self, fragment):
        """Mark mutual collisions between fragment and anything on the
        same channel currently being transmitted."""
        if fragment.channel not in self.transmitting:
            return
        for f in self.transmitting[fragment.channel]:
            f.collided.append(fragment)
            fragment.collided.append(f)

    def try_decode(self, packet, now):
        """Attempt to decode a complete packet.

        - LR-FHSS: requires >= 1 successful header AND >= threshold
          successful payloads.
        - LoRa: the single 'lora' fragment must have succeeded.
        """
        if packet.link_type == 'lora':
            return self._try_decode_lora(packet)
        else:
            return self._try_decode_lrfhss(packet)

    def _try_decode_lrfhss(self, packet):
        """Original LR-FHSS decode logic."""
        h_success = sum(1 for f in packet.fragments
                        if f.type == 'header' and f.success == 1)
        p_success = sum(1 for f in packet.fragments
                        if f.type == 'payload' and f.success == 1)

        # Use per-packet threshold if available, else fall back to Base's
        threshold = packet.threshold if packet.threshold is not None else self.threshold
        success = 1 if ((h_success > 0)
                        and (p_success >= threshold)) else 0
        if success == 1:
            packet.success = 1
            dedup_key = self._packet_dedup_key(packet)
            if dedup_key not in self.decoded_packets:
                self.decoded_packets.add(dedup_key)
                self.packets_received[packet.node_id] += 1
            return True
        else:
            return False

    def _try_decode_lora(self, packet):
        """LoRa decode: the single fragment must have succeeded."""
        if len(packet.fragments) > 0 and packet.fragments[0].success == 1:
            packet.success = 1
            dedup_key = self._packet_dedup_key(packet)
            if dedup_key not in self.decoded_packets:
                self.decoded_packets.add(dedup_key)
                self.packets_received[packet.node_id] += 1
            return True
        return False


class Relay(Base):
    """Relay base station — passive listener that forwards decoded packets.

    A Relay extends :class:`Base` so it receives and tracks fragments on
    its own channels with independent collision state, exactly like a
    sink.  When ``try_decode`` succeeds, the relay automatically
    schedules a forwarding process that retransmits the packet to the
    sink using ``forward_link_config``.

    Half-duplex constraint
    ----------------------
    A relay cannot listen and transmit at the same time.  When
    forwarding starts, **all ongoing receptions are cancelled**
    (channels are cleared).  While the relay is transmitting, incoming
    fragments from other nodes are silently dropped (checked in
    ``Node.transmit()`` via ``_is_transmitting``).

    Parameters
    ----------
    obw : int
        Number of LR-FHSS channels.
    threshold : int
        Minimum payload fragments for LR-FHSS decode.
    sensitivity : float
        Receiver sensitivity (dBm).
    position : tuple (x, y)
        2D position of the relay.
    forward_link_config : LinkConfig
        Link configuration used when forwarding to the sink.  Must have
        a ``pathloss_model`` set for the relay→sink hop.
    fading_generator : Fading
        Fading model used for forwarded-packet fragment generation.
    transmission_power : float
        Transmit power in dBm for forwarded packets.

    .. deprecated::
        The ``lora_channels``, ``gamma``, ``d0``, ``lpld0``, ``std``
        keyword arguments are forwarded to :class:`Base` for backward
        compatibility but have no effect on path-loss calculation.
    """

    def __init__(self, obw, threshold, sensitivity, *,
                 position, forward_link_config, fading_generator,
                 transmission_power=14,
                 dutycycle_period_s=None,
                 dutycycle_percent=None,
                 lora_channels=0, gamma=2.32, d0=1000.0,
                 lpld0=128.95, std=7.8):
        super().__init__(obw, threshold, sensitivity,
                         lora_channels=lora_channels, gamma=gamma,
                         d0=d0, lpld0=lpld0, std=std,
                         position=position)
        self.base_type = 'relay'
        self.forward_link_config = forward_link_config
        self.fading_generator = fading_generator
        self.transmission_power = transmission_power
        self.dutycycle_period_s = dutycycle_period_s
        self.dutycycle_percent = dutycycle_percent
        self.relayed = 0
        self._is_transmitting = False
        self._is_listening = True
        self._forward_queue = deque()
        self._queued_packet_ids = set()
        # Set by run.py before simulation starts
        self.env = None
        self.sink = None

    def _dutycycle_enabled(self):
        return (self.dutycycle_period_s is not None
                and self.dutycycle_percent is not None
                and self.dutycycle_period_s > 0
                and 0 <= self.dutycycle_percent < 100)

    def start(self):
        """Start relay background process when duty-cycle mode is enabled."""
        if self.env is None:
            return
        if self._dutycycle_enabled():
            self.env.process(self._dutycycle_loop())

    # -- override try_decode to trigger forwarding ----------------------

    def try_decode(self, packet, now):
        """Decode and, on success, schedule forwarding to the sink."""
        decoded = super().try_decode(packet, now)
        if decoded and self.env is not None and self.sink is not None:
            pkt_id = getattr(packet, 'original_packet_id', packet.id)
            dedup_key = (packet.node_id, pkt_id)
            if dedup_key not in self._queued_packet_ids:
                self._queued_packet_ids.add(dedup_key)
                self._forward_queue.append((packet.node_id, pkt_id))
            if not self._dutycycle_enabled():
                self.env.process(self._flush_queue())
        return decoded

    # -- half-duplex helpers --------------------------------------------

    def _cancel_receptions(self):
        """Cancel all ongoing fragment receptions (half-duplex).

        Clears every channel so that any fragments currently being
        received are lost.  ``Base.finish_fragment`` will detect the
        missing fragment and mark it as failed.
        """
        for ch in self.transmitting:
            self.transmitting[ch].clear()

    # -- forwarding process ---------------------------------------------

    def _forward(self, node_id, original_pkt_id):
        """SimPy process: retransmit one decoded packet to the sink.

        Creates a new packet using ``forward_link_config`` and sends
        it through the sink's reception pipeline.  The forwarded packet
        carries the original sender's ``node_id`` so the sink credits
        the correct device.  ``original_pkt_id`` is stored on the
        forwarded packet to prevent double-counting if the sink also
        decoded the direct path.

        TX mode (``_is_transmitting = True``) is set by the caller
        (``_flush_queue`` or ``_dutycycle_loop``) before this process
        is started.
        """
        # Small processing / turnaround delay
        yield self.env.timeout(0.001)

        fwd = Packet(node_id,
                     fading_generator=self.fading_generator,
                     link_config=self.forward_link_config)
        fwd.original_packet_id = original_pkt_id

        self.sink.add_packet(fwd)
        first_payload = False
        tw = (self.forward_link_config.transceiver_wait
              if self.forward_link_config.link_type == 'lrfhss' else 0)

        dist_to_sink = math.sqrt(
            (self.x - self.sink.x) ** 2 + (self.y - self.sink.y) ** 2)

        for frag in fwd.fragments:
            if not first_payload and frag.type == 'payload':
                first_payload = True
                yield self.env.timeout(tw)
            frag.timestamp = self.env.now
            self.sink.check_collision(frag)
            self.sink.receive_packet(frag)
            yield self.env.timeout(frag.duration)
            fwd_pathloss = getattr(self.forward_link_config,
                                   'pathloss_model', None)
            self.sink.finish_fragment(frag, dist_to_sink,
                                     self.transmission_power,
                                     fwd_pathloss)
            if fwd.success == 0:
                self.sink.try_decode(fwd, self.env.now)

        self.relayed += 1
        self._is_transmitting = False
        self._is_listening = True

    def _flush_queue(self):
        """Forward buffered packets (used by non-duty-cycle mode)."""
        if self._is_transmitting or len(self._forward_queue) == 0:
            return
        self._cancel_receptions()
        self._is_listening = False
        self._is_transmitting = True
        yield self.env.timeout(0.001)

        while len(self._forward_queue) > 0:
            node_id, pkt_id = self._forward_queue.popleft()
            yield self.env.process(self._forward(node_id, pkt_id))
            self._queued_packet_ids.discard((node_id, pkt_id))
            # _forward resets _is_transmitting; re-arm for next packet
            self._is_transmitting = True

        self._is_transmitting = False
        self._is_listening = True

    def _dutycycle_loop(self):
        """Alternate listen and relay windows based on duty-cycle config."""
        period = float(self.dutycycle_period_s)
        listen_time = period * (float(self.dutycycle_percent) / 100.0)
        relay_time = period - listen_time

        while True:
            if listen_time > 0:
                self._is_listening = True
                self._is_transmitting = False
                yield self.env.timeout(listen_time)

            if relay_time > 0:
                self._cancel_receptions()
                self._is_listening = False
                self._is_transmitting = True
                yield self.env.timeout(0.001)

                window_end = self.env.now + relay_time
                while len(self._forward_queue) > 0 and self.env.now < window_end:
                    node_id, pkt_id = self._forward_queue.popleft()
                    yield self.env.process(self._forward(node_id, pkt_id))
                    self._queued_packet_ids.discard((node_id, pkt_id))

                remaining = window_end - self.env.now
                if remaining > 0:
                    yield self.env.timeout(remaining)
