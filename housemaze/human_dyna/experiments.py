
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import os.path

from housemaze.human_dyna import env as maze
from housemaze.human_dyna import mazes


def maze1_all(config):
    """Maze 1: testing offtaskness for all 3 spaces."""
    env_kwargs = config.get('rlenv', {}).get('ENV_KWARGS', {})
    num_groups = env_kwargs.pop('NUM_GROUPS', 3)
    char2key, group_set, task_objects = mazes.get_group_set(num_groups)

    pretrain_params = mazes.get_pretraining_reset_params(
        group_set=group_set,
    )
    main_params = mazes.get_maze_reset_params(
        groups=group_set,
        char2key=char2key,
        maze_str=mazes.maze1,
        curriculum=True,
    )
    train_params = maze.EnvParams(
        reset_params=jtu.tree_map(
            lambda *v: jnp.stack(v), *(pretrain_params + main_params)),
    )

    test_params = maze.EnvParams(
        reset_params=jtu.tree_map(
            lambda *v: jnp.stack(v), *main_params),
    ).replace(training=False)

    return train_params, test_params, task_objects


def maze3_open(config):
    """Maze 3: testing if open space is skipped. should be."""
    env_kwargs = config.get('rlenv', {}).get('ENV_KWARGS', {})
    num_groups = env_kwargs.pop('NUM_GROUPS', 1)
    char2key, group_set, task_objects = mazes.get_group_set(num_groups)

    pretrain_params = mazes.get_pretraining_reset_params(
        group_set=group_set,
    )
    main_params = mazes.get_maze_reset_params(
        groups=group_set,
        char2key=char2key,
        maze_str=mazes.maze3,
        label=jnp.array(0),
        curriculum=True,
    )
    main_open_params = mazes.get_maze_reset_params(
        groups=group_set,
        char2key=char2key,
        maze_str=mazes.maze3_open,
        label=jnp.array(2),
    )

    train_params = pretrain_params + main_params
    train_params = maze.EnvParams(
        reset_params=jtu.tree_map(
            lambda *v: jnp.stack(v), *train_params),
    )

    test_params = main_params + main_open_params
    test_params = maze.EnvParams(
        reset_params=jtu.tree_map(
            lambda *v: jnp.stack(v), *test_params),
    ).replace(training=False)

    return train_params, test_params, task_objects


def maze3_randomize(config):
    """Maze 3: testing if open space is skipped. should be."""
    env_kwargs = config.get('rlenv', {}).get('ENV_KWARGS', {})
    num_groups = env_kwargs.pop('NUM_GROUPS', 1)
    char2key, group_set, task_objects = mazes.get_group_set(num_groups)

    pretrain_params = mazes.get_pretraining_reset_params(
        group_set=group_set,
    )
    main_params = mazes.get_maze_reset_params(
        groups=group_set,
        char2key=char2key,
        maze_str=mazes.maze3,
        randomize_agent=True,
    )
    main_open_params = mazes.get_maze_reset_params(
        groups=group_set,
        char2key=char2key,
        maze_str=mazes.maze3_open,
        randomize_agent=True,
    )

    train_params = pretrain_params + main_params
    train_params = maze.EnvParams(
        randomize_agent=True,
        reset_params=jtu.tree_map(
            lambda *v: jnp.stack(v), *train_params),
    )

    test_params = main_params + main_open_params
    test_params = maze.EnvParams(
        training=False,
        randomize_agent=False,
        reset_params=jtu.tree_map(
            lambda *v: jnp.stack(v), *test_params),
    )

    return train_params, test_params, task_objects


def maze5_two_paths(config):
    """Maze 3: testing if open space is skipped. should be."""
    env_kwargs = config.get('rlenv', {}).get('ENV_KWARGS', {})
    num_groups = env_kwargs.pop('NUM_GROUPS', 1)
    char2key, group_set, task_objects = mazes.get_group_set(num_groups)

    pretrain_params = mazes.get_pretraining_reset_params(
        group_set=group_set,
    )
    main_params = mazes.get_maze_reset_params(
        groups=group_set,
        char2key=char2key,
        maze_str=mazes.maze5,
    )

    train_params = pretrain_params + main_params
    train_params = maze.EnvParams(
        reset_params=jtu.tree_map(
            lambda *v: jnp.stack(v), *train_params),
    )

    test_params = main_params
    test_params = maze.EnvParams(
        training=False,
        reset_params=jtu.tree_map(
            lambda *v: jnp.stack(v), *test_params),
    )

    return train_params, test_params, task_objects
