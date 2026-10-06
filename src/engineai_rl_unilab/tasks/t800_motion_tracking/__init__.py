"""T800 motion tracking on UniLab's shared Manager-Based runtime."""

from unilab.base import registry
from unilab.envs import (
    ManagerBasedRlEnv,
    ManagerBasedRlEnvCfg,
    make_manager_based_rl_env,
)

from engineai_rl_unilab.assets import ensure_t800_assets


def make_engineai_t800_motion_env(
    cfg: ManagerBasedRlEnvCfg,
    num_envs: int = 1,
    backend_type: str = "mujoco",
) -> ManagerBasedRlEnv:
    """Resolve robot assets once before materializing the shared runtime."""
    ensure_t800_assets()
    return make_manager_based_rl_env(cfg, num_envs=num_envs, backend_type=backend_type)


registry.register_env_config("EngineAIT800MotionTracking", ManagerBasedRlEnvCfg)
registry.register_env(
    "EngineAIT800MotionTracking", make_engineai_t800_motion_env, sim_backend="mujoco"
)
# Keep the integration seam; a Motrix owner and its validation are separate work.
registry.register_env(
    "EngineAIT800MotionTracking", make_engineai_t800_motion_env, sim_backend="motrix"
)
registry.register_env(
    "EngineAIT800MotionTracking", make_engineai_t800_motion_env, sim_backend="mjwarp"
)
