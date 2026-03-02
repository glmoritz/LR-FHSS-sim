from lrfhss.lrfhss_core import *
from lrfhss.acrda import BaseACRDA
from lrfhss.settings import Settings
from lrfhss.link import LinkConfig

import simpy


def run_sim(settings: Settings, seed=0):
    """Run a single simulation and return results.

    Supports three modes depending on ``settings``:
    1. **Pure LR-FHSS** (original) — no relays, LR-FHSS link.
    2. **Pure LoRa** — no relays, LoRa link.
    3. **Relay** — end devices broadcast to all receivers (sink + relays).
       Relays are passive listeners that independently decode and
       forward to the sink.

    Returns
    -------
    list
        ``[[success_ratio], [goodput_bytes], [transmitted],
          [relayed_count]]``
        or ``1`` if no transmissions occurred.
    """
    random.seed(seed)
    np.random.seed(seed)
    env = simpy.Environment()

    # ---- Parameters ----
    lora_ch = getattr(settings, 'lora_channels', 0)
    gamma = getattr(settings, 'gamma', 2.32)
    d0 = getattr(settings, 'd0', 1000.0)
    lpld0 = getattr(settings, 'lpld0', 128.95)
    std = getattr(settings, 'std', 7.8)
    base_pos = getattr(settings, 'base_position', (0, 0))

    link_config = getattr(settings, 'link_config', None)
    relay_link_config = getattr(settings, 'relay_link_config', None)
    number_relays = getattr(settings, 'number_relays', 0)
    relay_positions = getattr(settings, 'relay_positions', None)

    # ---- Sink (main base station) ----
    if settings.base == 'acrda':
        bs = BaseACRDA(settings.obw, settings.window_size,
                       settings.window_step, settings.time_on_air,
                       settings.threshold, settings.sensitivity,
                       lora_channels=lora_ch, gamma=gamma, d0=d0,
                       lpld0=lpld0, std=std, position=base_pos)
        env.process(bs.sic_window(env))
    else:
        bs = Base(settings.obw, settings.threshold, settings.sensitivity,
                  lora_channels=lora_ch, gamma=gamma, d0=d0,
                  lpld0=lpld0, std=std, position=base_pos)

    # ---- Relay base stations (passive listeners) ----
    relays = []
    fwd_link = relay_link_config or link_config
    for i in range(number_relays):
        pos = relay_positions[i] if relay_positions else (0, 0)
        relay = Relay(
            settings.obw, settings.threshold, settings.sensitivity,
            position=pos,
            forward_link_config=fwd_link,
            fading_generator=settings.fading_generator,
            transmission_power=settings.transmission_power,
            lora_channels=lora_ch, gamma=gamma, d0=d0,
            lpld0=lpld0, std=std,
        )
        # Bind SimPy env and sink reference for forwarding
        relay.env = env
        relay.sink = bs
        relays.append(relay)

    # ---- Receivers list: sink + all relays ----
    receivers = [bs] + relays

    # ---- End-device nodes ----
    nodes = []
    for i in range(settings.number_nodes):
        if link_config is not None:
            node = Node(
                traffic_generator=settings.traffic_generator,
                fading_generator=settings.fading_generator,
                max_distance=settings.max_distance,
                transmission_power=settings.transmission_power,
                link_config=link_config,
                base_position=base_pos,
            )
        else:
            # Legacy path (backward compatible)
            node = Node(settings.obw, settings.headers, settings.payloads,
                        settings.header_duration, settings.payload_duration,
                        settings.transceiver_wait,
                        settings.traffic_generator,
                        settings.fading_generator,
                        settings.max_distance, settings.transmission_power)

        # Register node at every receiver
        for rx in receivers:
            rx.add_node(node.id)
        nodes.append(node)
        env.process(node.transmit(env, receivers))

    # ---- Run ----
    env.run(until=settings.simulation_time)

    # ---- Results ----
    # Count successes at the SINK only (final destination)
    success = sum(v for k, v in bs.packets_received.items())
    transmitted = sum(n.transmitted for n in nodes)
    total_relayed = sum(r.relayed for r in relays)

    if transmitted == 0:
        return 1
    else:
        return [[success / transmitted],
                [success * settings.payload_size],
                [transmitted],
                [total_relayed]]


if __name__ == "__main__":
    s = Settings()
    print(s.sensitivity)
    print(run_sim(s))