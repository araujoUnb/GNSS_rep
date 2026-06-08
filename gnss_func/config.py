"""System configuration for the GNSS time-delay simulation.

Centralizes every scenario/system parameter that used to be hard-coded inside
the simulation scripts, so a Monte-Carlo job (e.g. a SLURM array task) is fully
described by a single ``SystemConfig`` instance.

Defaults follow the paper Table (B = 1.023 MHz, N = 2046, Q = 11 correlators,
Delta_phi = 5 deg, SMR = 5 dB, C/N0 = 48 dB-Hz). The methods are implemented as
described in the paper and validated against the stored results in analysis/.
"""

from dataclasses import dataclass


@dataclass
class SystemConfig:
    # --- transmitter / signal (paper Table) ---
    sat_id: int = 20                 # GPS satellite (PRN) id
    bandwidth: float = 1.023e6       # one-sided bandwidth B [Hz]
    fc: float = 1575.42e6            # carrier frequency (L1) [Hz]
    time_period: float = 1e-3        # coherent integration period T [s] -> N=2046

    # --- receiver / array ---
    n_antennas: int = 8              # ULA elements
    n_epochs: int = 30               # number of snapshots (periods)
    n_correlators: int = 11          # Q correlators in the bank (paper)
    # whitening subspace dim: None -> full Q (paper's Q_omega, Q_omega^H Q_omega=I_Q)
    n_qw: int = None
    delay_granularity: int = 11      # estimator delay-grid resolution

    # --- operating point ---
    cn0_db: float = 48.0             # carrier-to-noise density [dB-Hz]

    # --- geometry ---
    delta_phi_deg: float = 5.0       # LOS/NLOS azimuth separation [deg] (paper)

    # --- array calibration error (A = A_D + epsilon * A_P, A_P ~ CN(0,1)) ---
    epsilon: float = 0.0             # 0 -> perfectly calibrated
    smr_db: float = 5.0              # signal-to-multipath ratio [dB] (paper)
    ap_unit_variance: bool = True    # A_P: True E|x|^2=1 (/sqrt2); False E|x|^2=2

    @property
    def chip_period(self) -> float:
        return 1.0 / self.bandwidth

    @property
    def n_samples(self) -> int:
        return int(2 * self.bandwidth * self.time_period)
