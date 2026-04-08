import numpy as np


class ChaosWindGenerator:
    """Applies randomized wind + body torques as external disturbances.

    - Wind is applied via `model.opt.wind[:3]`.
    - Torques are applied via `data.xfrc_applied[body_id, 3:6]`.

    Set intensities to 0 to disable either effect.
    """

    def __init__(
        self,
        force_intensity: float = 5.0,
        torque_intensity: float = 0.01,
        body_name: str = "Drone",
    ):
        self.force_intensity = float(force_intensity)
        self.torque_intensity = float(torque_intensity)
        self.body_name = str(body_name)

    def apply_disturbance(self, model, data) -> None:
        if self.force_intensity != 0.0:
            wind_vec = np.random.uniform(-self.force_intensity, self.force_intensity, 3)
            model.opt.wind[:3] = wind_vec

        if self.torque_intensity != 0.0:
            body_id = model.body(self.body_name).id
            torques = np.random.uniform(-self.torque_intensity, self.torque_intensity, 3)
            data.xfrc_applied[body_id, 3:6] = torques
