"""T800 motion tracking on UniLab's shared Manager-Based runtime."""

from engineai_rl_unilab.assets import ensure_t800_assets
from engineai_rl_unilab.tasks.t800.motion_preflight import preflight_motion
from unilab.base import registry
from unilab.envs import (
    ManagerBasedRlEnv,
    ManagerBasedRlEnvCfg,
    make_manager_based_rl_env,
)


def make_engineai_t800_motion_env(
    cfg: ManagerBasedRlEnvCfg,
    num_envs: int = 1,
    backend_type: str = "mujoco",
) -> ManagerBasedRlEnv:
    """Reject invalid references before allocating environments or applying DR."""
    ensure_t800_assets()
    cfg = preflight_motion(cfg)
    return make_manager_based_rl_env(cfg, num_envs=num_envs, backend_type=backend_type)


registry.register_env_config("EngineAIT800MotionTracking", ManagerBasedRlEnvCfg)
registry.register_env(
    "EngineAIT800MotionTracking", make_engineai_t800_motion_env, sim_backend="mujoco"
)
# Keep the integration seam; a Motrix owner and its validation are separate work.
registry.register_env(
    "EngineAIT800MotionTracking", make_engineai_t800_motion_env, sim_backend="motrix"
)
