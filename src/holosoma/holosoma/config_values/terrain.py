from holosoma.config_types.terrain import (
    MeshType,
    TerrainCurriculumLayoutCfg,
    TerrainManagerCfg,
    TerrainTermCfg,
)

terrain_locomotion_plane = TerrainManagerCfg(
    terrain_term=TerrainTermCfg(
        func="holosoma.managers.terrain.terms.locomotion:TerrainLocomotion",
        mesh_type=MeshType.PLANE,
        horizontal_scale=1.0,
        vertical_scale=0.005,
        border_size=40,
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=0.0,
        terrain_length=8.0,
        terrain_width=8.0,
        num_rows=10,
        num_cols=20,
        max_slope=0.3,
        platform_size=2.0,
        step_width_range=[0.30, 0.40],
        amplitude_range=[0.01, 0.05],
        slope_treshold=0.75,
    )
)

terrain_locomotion_mix = TerrainManagerCfg(
    terrain_term=TerrainTermCfg(
        func="holosoma.managers.terrain.terms.locomotion:TerrainLocomotion",
        mesh_type=MeshType.TRIMESH,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        border_size=40,
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=0.0,
        terrain_length=8.0,
        terrain_width=8.0,
        num_rows=10,
        num_cols=20,
        terrain_config={
            "flat": 0.2,
            "rough": 0.6,
            "low_obstacles": 0.2,
            "smooth_slope": 0.0,
            "rough_slope": 0.0,
        },
        max_slope=0.3,
        slope_treshold=0.75,
    )
)

# These terrain dimensions and obstacle geometries are implementation choices;
# the source paper does not publish the corresponding values.
terrain_locomotion_curriculum = TerrainManagerCfg(
    terrain_term=TerrainTermCfg(
        func="holosoma.managers.terrain.terms.locomotion:TerrainLocomotion",
        mesh_type=MeshType.TRIMESH,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        border_size=40,
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=0.0,
        terrain_length=8.0,
        terrain_width=8.0,
        num_rows=10,
        num_cols=20,
        terrain_config={
            "flat": 0.25,
            "box": 0.25,
            "stair": 0.25,
            "hurdle": 0.25,
        },
        curriculum_layout=TerrainCurriculumLayoutCfg(
            enabled=True,
            terrain_types=["flat", "box", "stair", "hurdle"],
            difficulty_min=0.0,
            difficulty_max=1.0,
            spawn_clearance_radius=1.0,
            box_height_range=[0.05, 0.30],
            box_size=0.6,
            box_spacing=1.4,
            stair_height_range=[0.05, 0.35],
            stair_step_width=0.4,
            hurdle_height_range=[0.05, 0.35],
            hurdle_width=0.2,
            hurdle_spacing=1.0,
        ),
        max_slope=0.3,
        slope_treshold=0.75,
    )
)

terrain_load_obj = TerrainManagerCfg(
    terrain_term=TerrainTermCfg(
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=0.0,
        mesh_type=MeshType.LOAD_OBJ,
        func="holosoma.managers.terrain.terms.locomotion:TerrainLocomotion",
        obj_file_path="holosoma/data/motions/g1_29dof/whole_body_tracking/terrain_parkour.obj",
    )
)

DEFAULTS = {
    "terrain_locomotion_plane": terrain_locomotion_plane,
    "terrain_locomotion_mix": terrain_locomotion_mix,
    "terrain_locomotion_curriculum": terrain_locomotion_curriculum,
    "terrain_load_obj": terrain_load_obj,
}
