"""Closed-loop generator-in-the-loop WBT experiment for the G1 robot.

Same as ``g1_29dof_wbt`` except:
- the motion command is the frozen-generator closed-loop term, and
- the two rewards that need per-body orientations / angular velocities are
  zero-weighted (the generator's representation carries no per-body
  orientations, mirroring the paper's feature set).
"""

from dataclasses import replace

from holosoma.config_values.wbt.g1 import reward as g1_reward
from holosoma.config_values.wbt.g1.command_gen import g1_29dof_wbt_gen_command
from holosoma.config_values.wbt.g1.experiment import g1_29dof_wbt

_terms = dict(g1_reward.g1_29dof_wbt_reward.terms)
for _name in ("motion_relative_body_orientation_error_exp", "motion_global_body_ang_vel"):
    _terms[_name] = replace(_terms[_name], weight=0.0)
_gen_reward = replace(g1_reward.g1_29dof_wbt_reward, terms=_terms)

g1_29dof_wbt_gen = replace(
    g1_29dof_wbt,
    training=replace(g1_29dof_wbt.training, name="g1_29dof_wbt_gen_manager"),
    command=g1_29dof_wbt_gen_command,
    reward=_gen_reward,
)

__all__ = ["g1_29dof_wbt_gen"]
