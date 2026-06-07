"""System configuration for the GNSS time-delay simulation.

Centralizes every scenario/system parameter that used to be hard-coded inside
the simulation scripts, so a Monte-Carlo job (e.g. a SLURM array task) is fully
described by a single ``SystemConfig`` instance.
"""

from dataclasses import dataclass


@dataclass
class SystemConfig:
    # --- transmitter / signal ---
    # NOTE: defaults below are the WORKING config that reproduces the reference
    # deltaTau data. The paper Table states B=1.023 MHz (N=2046) and Q=11, but
    # that config (with full Q_omega) collapses estimator performance (~100 m,
    # outlier ~1). The mismatch is pending the original code — see the memory
    # note gnss-lskrf-bo-fidelity-open.
    sat_id: int = 20                 # GPS satellite (PRN) id
    bandwidth: float = 1023e6        # working B (paper Table says 1.023e6)
    fc: float = 1575.42e6            # carrier frequency (L1) [Hz]
    time_period: float = 1e-3        # coherent integration period T [s]

    # --- receiver / array ---
    n_antennas: int = 8              # ULA elements
    n_epochs: int = 30               # number of snapshots (periods)
    n_correlators: int = 22          # bank size (working; paper Table says Q=11)
    n_qw: int = 7                    # signal-subspace dim kept by whitening (denoising)
    delay_granularity: int = 11      # estimator delay-grid resolution

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
