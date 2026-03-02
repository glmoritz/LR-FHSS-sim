import random
import math
import numpy as np
from abc import ABC, abstractmethod
from lrfhss.link import lora_sensitivity, lora_min_snr

        
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

        if link_config is not None:
            self.link_type = link_config.link_type
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
        
class Node():
    """A network node that transmits packets and optionally relays.

    Supports both LR-FHSS and conventional LoRa link types.  Uses 2D
    ``(x, y)`` positioning; distance to the base station is derived
    automatically.

    Relay mode
    ----------
    When ``relay_enabled=True`` the node listens for completed
    transmissions from other nodes, checks whether it can decode them
    (distance-based RSSI/SNR + collision), and if so forwards a **new**
    packet to the gateway using ``relay_link_config``.

    * **LoRa relay** – the single LoRa fragment is evaluated; if OK the
      relay retransmits immediately.
    * **LR-FHSS relay** – the relay waits for the *whole* packet
      (all fragments) and then does a standard decode check
      (≥1 header + ≥threshold payloads OK at relay distance).

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
                 link_config=None, position=None,
                 relay_enabled=False, relay_link_config=None,
                 base_position=(0, 0)):
        self.id = id(self)
        self.transmitted = 0
        self.traffic_generator = traffic_generator
        self.fading_generator = fading_generator
        self.transmission_power = transmission_power
        self.max_distance = max_distance
        self.link_config = link_config

        # -- 2D position ------------------------------------------------
        if position is not None:
            self.x, self.y = position[0], position[1]
        else:
            # Uniform random point inside circle of radius max_distance
            angle = random.uniform(0, 2 * math.pi)
            r = max_distance * math.sqrt(random.uniform(0, 1))
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
        self.packet = self._create_packet()

        # -- Relay configuration ----------------------------------------
        self.relay_enabled = relay_enabled
        self.relay_link_config = relay_link_config
        self.relayed = 0
        self._is_transmitting = False

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

    def transmit(self, env, bs, relays=None):
        """SimPy process: periodically transmit packets.

        Parameters
        ----------
        env : simpy.Environment
        bs : Base
            The gateway / base station receiver.
        relays : list[Node] or None
            Nodes with ``relay_enabled=True``.  After each packet
            completes, eligible relays are offered the packet for
            forwarding if the base failed to decode it directly.
        """
        while True:
            yield env.timeout(self.next_transmission())
            self._is_transmitting = True
            self.transmitted += 1
            bs.add_packet(self.packet)
            next_fragment = self.packet.next()
            first_payload = 0
            while next_fragment:
                if (first_payload == 0
                        and next_fragment.type == 'payload'):
                    first_payload = 1
                    yield env.timeout(self.transceiver_wait)
                next_fragment.timestamp = env.now
                bs.check_collision(next_fragment)
                bs.receive_packet(next_fragment)
                yield env.timeout(next_fragment.duration)
                bs.finish_fragment(next_fragment, self.distance,
                                   self.transmission_power)
                if self.packet.success == 0:
                    bs.try_decode(self.packet, env.now)
                next_fragment = self.packet.next()
            self._is_transmitting = False

            # -- offer to relays if base could not decode ---------------
            if relays and self.packet.success == 0:
                for relay in relays:
                    if (relay.id != self.id
                            and relay.relay_enabled
                            and not relay._is_transmitting):
                        if relay._can_relay_decode(
                                self.packet, self, bs):
                            env.process(
                                relay._relay_forward(
                                    env, bs, self.packet, self))
                            break  # first eligible relay wins

            self.end_of_transmission()

    # -- relay: decode check --------------------------------------------

    def _can_relay_decode(self, packet, sender, bs):
        """Check whether this relay can decode *packet* from *sender*.

        Uses the same physics as ``Base.finish_fragment()`` but
        evaluated at the relay's position (distance to *sender*).
        Collision information is reused from the base-station
        processing — a simplification; in practice collisions may
        differ spatially.
        """
        dist = self.distance_to(sender.x, sender.y)
        if dist <= 0:
            dist = 1.0

        if packet.link_type == 'lora':
            frag = packet.fragments[0]
            return self._frag_ok_lora(frag, dist,
                                      sender.transmission_power, bs)

        # LR-FHSS — need ≥1 header + ≥threshold payloads OK
        h_ok = sum(1 for f in packet.fragments
                   if f.type == 'header'
                   and self._frag_ok_lrfhss(f, dist,
                                             sender.transmission_power,
                                             bs))
        p_ok = sum(1 for f in packet.fragments
                   if f.type == 'payload'
                   and self._frag_ok_lrfhss(f, dist,
                                             sender.transmission_power,
                                             bs))
        return h_ok > 0 and p_ok >= bs.threshold

    def _frag_ok_lrfhss(self, frag, dist, tx_power, bs):
        """Would an LR-FHSS fragment succeed at this relay's distance?"""
        if len(frag.collided) > 0:
            return False
        tx_W = 10 ** (tx_power / 10) / 1000
        sens_W = 10 ** (bs.sensitivity / 10) / 1000
        snr = (tx_W * (frag.intensity ** 2)) / (dist ** 4)
        return snr > sens_W

    def _frag_ok_lora(self, frag, dist, tx_power, bs):
        """Would a LoRa fragment succeed at this relay's distance?"""
        if len(frag.collided) > 0:
            return False
        if dist <= 0:
            dist = 1.0
        lpl = (10 * bs.gamma * math.log10(dist / bs.d0)
               + np.random.normal(bs.lpld0, bs.std))
        rssi = tx_power - lpl
        if frag.intensity > 0:
            rssi += 20.0 * math.log10(frag.intensity)
        noise_floor = -174.0 + 10.0 * np.log10(frag.bw * 1e3)
        snr_db = rssi - noise_floor
        return (rssi > lora_sensitivity(frag.sf, frag.bw)
                and snr_db > lora_min_snr(frag.sf))

    # -- relay: forward -------------------------------------------------

    def _relay_forward(self, env, bs, original_packet, sender):
        """SimPy process: retransmit a decoded packet to the gateway.

        Creates a brand-new packet using ``relay_link_config`` and
        transmits it to the base station.  The forwarded packet carries
        the original sender's ``node_id`` so the base credits the right
        device.  An ``original_packet_id`` tag prevents the base from
        double-counting if the direct path also succeeds later (e.g.
        ACRDA SIC).
        """
        self._is_transmitting = True

        # Small processing / turnaround delay
        yield env.timeout(0.001)

        fwd_packet = Packet(original_packet.node_id,
                            fading_generator=self.fading_generator,
                            link_config=self.relay_link_config)
        fwd_packet.original_packet_id = original_packet.id

        bs.add_packet(fwd_packet)
        next_fragment = fwd_packet.next()
        first_payload = 0
        tw = (self.relay_link_config.transceiver_wait
              if self.relay_link_config.link_type == 'lrfhss' else 0)
        while next_fragment:
            if (first_payload == 0
                    and next_fragment.type == 'payload'):
                first_payload = 1
                yield env.timeout(tw)
            next_fragment.timestamp = env.now
            bs.check_collision(next_fragment)
            bs.receive_packet(next_fragment)
            yield env.timeout(next_fragment.duration)
            bs.finish_fragment(next_fragment, self.distance,
                               self.transmission_power)
            if fwd_packet.success == 0:
                bs.try_decode(fwd_packet, env.now)
            next_fragment = fwd_packet.next()

        self.relayed += 1
        self._is_transmitting = False

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
    gamma : float
        Path-loss exponent for log-distance model (LoRa).
    d0 : float
        Reference distance in metres (LoRa path-loss).
    lpld0 : float
        Path loss at reference distance d0 in dB (LoRa).
    std : float
        Shadowing standard deviation in dB (LoRa).
    """
    def __init__(self, obw, threshold, sensitivity, *,
                 lora_channels=0, gamma=2.32, d0=1000.0,
                 lpld0=128.95, std=7.8):
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
        self.decoded_packets = set()  # Track decoded packet IDs (dedup)
        self.threshold = threshold
        self.sensitivity = sensitivity
        # Log-distance path-loss parameters (used for LoRa fragments)
        self.gamma = gamma
        self.d0 = d0
        self.lpld0 = lpld0
        self.std = std

    def add_packet(self, packet):
        pass

    def add_node(self, id):
        self.packets_received[id] = 0

    def receive_packet(self, fragment):
        # Lazily create channel bucket if it doesn't exist yet
        if fragment.channel not in self.transmitting:
            self.transmitting[fragment.channel] = []
        self.transmitting[fragment.channel].append(fragment)

    def finish_fragment(self, fragment, distance, transmission_power):
        """Mark a fragment as finished and determine success.

        For LR-FHSS fragments: uses the existing fading-intensity SNR model.
        For LoRa fragments: uses log-distance path-loss -> RSSI/SNR checks
        with per-SF sensitivity and minimum SNR requirements.
        """
        self.transmitting[fragment.channel].remove(fragment)

        if fragment.type == 'lora':
            # --- Conventional LoRa reception check ---
            self._finish_lora_fragment(fragment, distance, transmission_power)
        else:
            # --- LR-FHSS reception check (original logic) ---
            self._finish_lrfhss_fragment(fragment, distance, transmission_power)

    def _finish_lrfhss_fragment(self, fragment, distance, transmission_power):
        """Original LR-FHSS fragment success check."""
        transmission_power_W = 10 ** (transmission_power / 10) / 1000
        sensitivity_W = 10 ** (self.sensitivity / 10) / 1000

        snr = (transmission_power_W * (fragment.intensity ** 2)) / (distance ** 4)

        if len(fragment.collided) == 0 and snr > sensitivity_W:
            fragment.success = 1
        fragment.transmitted = 1

    def _finish_lora_fragment(self, fragment, distance, transmission_power):
        """Conventional LoRa fragment success check using log-distance
        path-loss model, per-SF sensitivity and SNR requirement."""
        # Guard against zero distance (co-located)
        if distance <= 0:
            distance = 1.0

        # Log-distance path loss (dB)
        lpl = (10 * self.gamma * math.log10(distance / self.d0)
               + np.random.normal(self.lpld0, self.std))
        rssi = transmission_power - lpl  # dBm

        # Apply fading intensity as additional gain/loss (dB scale)
        # intensity is a linear multiplier from the Fading class, convert:
        #   gain_dB = 20*log10(intensity)  (voltage-domain fading)
        if fragment.intensity > 0:
            rssi += 20.0 * math.log10(fragment.intensity)

        # Noise floor for LoRa bandwidth
        noise_floor = -174.0 + 10.0 * np.log10(fragment.bw * 1e3)  # dBm
        snr_db = rssi - noise_floor

        # Check sensitivity and minimum SNR for the SF
        min_sensi = lora_sensitivity(fragment.sf, fragment.bw)
        min_snr = lora_min_snr(fragment.sf)

        if (len(fragment.collided) == 0
                and rssi > min_sensi
                and snr_db > min_snr):
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

        success = 1 if ((h_success > 0)
                        and (p_success >= self.threshold)) else 0
        if success == 1:
            packet.success = 1
            pkt_id = getattr(packet, 'original_packet_id', packet.id)
            if pkt_id not in self.decoded_packets:
                self.decoded_packets.add(pkt_id)
                self.packets_received[packet.node_id] += 1
            return True
        else:
            return False

    def _try_decode_lora(self, packet):
        """LoRa decode: the single fragment must have succeeded."""
        if len(packet.fragments) > 0 and packet.fragments[0].success == 1:
            packet.success = 1
            pkt_id = getattr(packet, 'original_packet_id', packet.id)
            if pkt_id not in self.decoded_packets:
                self.decoded_packets.add(pkt_id)
                self.packets_received[packet.node_id] += 1
            return True
        return False
