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
    3. **Relay** — end devices transmit; relay nodes listen and forward
       failed packets to the gateway.

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

    # ---- Base station ----
    lora_ch = getattr(settings, 'lora_channels', 0)
    gamma = getattr(settings, 'gamma', 2.32)
    d0 = getattr(settings, 'd0', 1000.0)
    lpld0 = getattr(settings, 'lpld0', 128.95)
    std = getattr(settings, 'std', 7.8)
    base_pos = getattr(settings, 'base_position', (0, 0))

    if settings.base == 'acrda':
        bs = BaseACRDA(settings.obw, settings.window_size,
                       settings.window_step, settings.time_on_air,
                       settings.threshold, settings.sensitivity)
        env.process(bs.sic_window(env))
    else:
        bs = Base(settings.obw, settings.threshold, settings.sensitivity,
                  lora_channels=lora_ch, gamma=gamma, d0=d0,
                  lpld0=lpld0, std=std)

    # ---- Link configs ----
    link_config = getattr(settings, 'link_config', None)
    relay_link_config = getattr(settings, 'relay_link_config', None)
    number_relays = getattr(settings, 'number_relays', 0)
    relay_positions = getattr(settings, 'relay_positions', None)

    # ---- Relay nodes ----
    relays = []
    for i in range(number_relays):
        pos = relay_positions[i] if relay_positions else None
        relay = Node(
            traffic_generator=settings.traffic_generator,
            fading_generator=settings.fading_generator,
            max_distance=settings.max_distance,
            transmission_power=settings.transmission_power,
            link_config=relay_link_config or link_config,
            position=pos,
            base_position=base_pos,
            relay_enabled=True,
            relay_link_config=relay_link_config or link_config,
        )
        bs.add_node(relay.id)
        relays.append(relay)
        env.process(relay.transmit(env, bs, relays=None))

    relay_list = relays if relays else None

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
        bs.add_node(node.id)
        nodes.append(node)
        env.process(node.transmit(env, bs, relays=relay_list))

    # ---- Run ----
    env.run(until=settings.simulation_time)

    # ---- Results ----
    # Only count end-device packets (not relay self-transmissions)
    ed_node_ids = {n.id for n in nodes}
    success = sum(v for k, v in bs.packets_received.items()
                  if k in ed_node_ids)
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