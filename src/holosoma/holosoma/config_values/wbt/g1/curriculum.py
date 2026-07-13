"""Whole Body Tracking curriculum presets for the G1 robot."""

from holosoma.config_types.curriculum import CurriculumManagerCfg, CurriculumTermCfg

g1_29dof_wbt_curriculum = CurriculumManagerCfg(
    params={
        "num_compute_average_epl": 1000,
    },
    setup_terms={
        "average_episode_tracker": CurriculumTermCfg(
            func="holosoma.managers.curriculum.terms.locomotion:AverageEpisodeLengthTracker",
            params={},
        ),
    },
    reset_terms={},
    step_terms={},
)

g1_29dof_wbt_gen_terrain_curriculum = CurriculumManagerCfg(
    params={
        "num_compute_average_epl": 1000,
    },
    setup_terms={
        "average_episode_tracker": CurriculumTermCfg(
            func="holosoma.managers.curriculum.terms.locomotion:AverageEpisodeLengthTracker",
            params={},
        ),
        "terrain_curriculum": CurriculumTermCfg(
            func="holosoma.managers.curriculum.terms.terrain:TerrainCurriculum",
            params={
                "enabled": True,
                "initial_level": 0,
                "min_level": 0,
                "max_level": None,
                "success_min_episode_fraction": 0.9,
                # Implementation choice: the paper does not publish an
                # obstacle-crossing distance. Concentric obstacles start just
                # outside the 1 m clear spawn square, so 1.5 m verifies that
                # the robot progressed beyond the first obstacle band.
                "crossing_distance_m": 1.5,
                "promote_success_streak": 5,
                "demote_failure_streak": 2,
                "skip_first_episode": True,
            },
        ),
    },
    reset_terms={},
    step_terms={},
)

__all__ = ["g1_29dof_wbt_curriculum", "g1_29dof_wbt_gen_terrain_curriculum"]
