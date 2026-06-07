"""System configuration for the GNSS time-delay simulation.

Centralizes every scenario/system parameter that used to be hard-coded inside
the simulation scripts, so a Monte-Carlo job (e.g. a SLURM array task) is fully
described by a single ``SystemConfig`` instance.
"""

from dataclasses import dataclass


@dataclass
class SystemConfig:
    # --- transmitter / signal ---
    sat_id: int = 20                 # GPS satellite (PRN) id
    bandwidth: float = 1023e6        # one-sided bandwidth B [Hz]
    fc: float = 1575.42e6            # carrier frequency (L1) [Hz]
    time_period: float = 1e-3        # coherent integration period T [s]

    # --- receiver / array ---
    n_antennas: int = 8              # ULA elements
    n_epochs: int = 30               # number of snapshots (periods)
    delay_granularity: int = 11      # controls #correlators and tau grid

    # --- operating point ---
    cn0_db: float = 48.0             # carrier-to-noise density [dB-Hz]

    # --- geometry (defaults match the legacy scripts) ---
    delta_phi_deg: float = 60.0      # LOS/NLOS azimuth separation [deg]

    # --- array calibration error (A = A_D + epsilon * A_P, A_P ~ CN(0,1)) ---
    # epsilon = 0.0 reproduces the legacy (perfectly calibrated) scenarios.
    epsilon: float = 0.0
    smr_db: float = 5.0              # signal-to-multipath ratio [dB] (reserved)

    @property
    def chip_period(self) -> float:
        return 1.0 / self.bandwidth

    @property
    def n_samples(self) -> int:
        return int(2 * self.bandwidth * self.time_period)
