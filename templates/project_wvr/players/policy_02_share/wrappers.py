"""A shared copy, made by `tools.share` -- the State only."""
from __future__ import annotations
import math
import numpy as np
from aircombat_gym.wvr import obs as O
MUZZLE_MS = 1050.0
WEZ_R_MAX = 1500.0
TRACK_LOCK_S = 1.0
T_MAX_S = 240.0
OMEGA_MAX_RAD_S = 0.20943951023931956

def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi

class State:

    def __init__(self):
        self.dim = len(self.channels(np.zeros(O.STATE_DIM, dtype=np.float32)))

    def channels(self, obs) -> dict[str, float]:
        d = O.unpack(obs)
        dx = d['opp_x'] - d['own_x']
        dy = d['opp_y'] - d['own_y']
        dz = d['opp_h'] - d['own_h']
        r = math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-09
        dvx = d['opp_vx'] - d['own_vx']
        dvy = d['opp_vy'] - d['own_vy']
        dvz = d['opp_vz'] - d['own_vz']
        tof = r / MUZZLE_MS
        (ax, ay) = (dx + dvx * tof, dy + dvy * tof)
        lead = _wrap(math.atan2(ax, ay) - d['own_psi'])
        (nx, ny, nz) = (math.sin(d['opp_psi']), math.cos(d['opp_psi']), math.sin(d['opp_theta']))
        aa = math.acos(max(-1.0, min(1.0, (dx * nx + dy * ny + dz * nz) / r)))
        r_dot = (dx * dvx + dy * dvy + dz * dvz) / r
        own_v = math.sqrt(d['own_vx'] ** 2 + d['own_vy'] ** 2 + d['own_vz'] ** 2)
        opp_v = math.sqrt(d['opp_vx'] ** 2 + d['opp_vy'] ** 2 + d['opp_vz'] ** 2)
        return {'range': math.exp(-r / WEZ_R_MAX), 'lead_sin': math.sin(lead), 'lead_cos': math.cos(lead), 'aa_sin': math.sin(aa), 'aa_cos': math.cos(aa), 'closure': r_dot / 340.0, 'own_speed': own_v / 340.0, 'opp_speed': opp_v / 340.0, 'speed_margin': (own_v - opp_v) / 340.0, 'opp_bank': math.sin(d['opp_phi']), 'opp_yaw': d['opp_r'] / OMEGA_MAX_RAD_S, 'own_health': d['own_health'], 'opp_health': d['opp_health'], 'lock_progress': min(1.0, d['own_track_time'] / TRACK_LOCK_S), 'threat_lock': min(1.0, d['opp_track_time'] / TRACK_LOCK_S), 'in_wez': d['own_in_wez'], 't_remaining': d['t_remaining'] / T_MAX_S}

    def __call__(self, obs) -> np.ndarray:
        ch = self.channels(obs)
        return np.fromiter(ch.values(), dtype=np.float32, count=len(ch))
