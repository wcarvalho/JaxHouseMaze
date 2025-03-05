import jax.numpy as jnp
import jax.tree_util as jtu

from housemaze.human_dyna import mazes, multitask_env


def maze1_all(config):
  """Maze 1: testing offtaskness for all 3 spaces."""
  env_kwargs = config.get("rlenv", {}).get("ENV_KWARGS", {})
  num_groups = env_kwargs.pop("NUM_GROUPS", 3)
  char2key, group_set, task_objects = mazes.get_group_set(num_groups)

  pretrain_params = mazes.get_pretraining_reset_params(
    groups=group_set,
  )
  main_params = mazes.get_maze_reset_params(
    groups=group_set,
    char2key=char2key,
    maze_str=mazes.maze1,
    label=jnp.array(0),
    curriculum=True,
  )
  train_params = multitask_env.EnvParams(
    reset_params=jtu.tree_map(
      lambda *v: jnp.stack(v), *(pretrain_params + main_params)
    ),
  )

  test_params = multitask_env.EnvParams(
    reset_params=jtu.tree_map(lambda *v: jnp.stack(v), *main_params),
  ).replace(training=False)

  return train_params, test_params, task_objects


def maze3_open(config):
  """Maze 3: testing if open space is skipped. should be."""
  env_kwargs = config.get("rlenv", {}).get("ENV_KWARGS", {})
  num_groups = env_kwargs.pop("NUM_GROUPS", 1)
  char2key, group_set, task_objects = mazes.get_group_set(num_groups)

  pretrain_params = mazes.get_pretraining_reset_params(
    groups=group_set,
  )
  main_params = mazes.get_maze_reset_params(
    groups=group_set,
    char2key=char2key,
    maze_str=mazes.maze3,
    # label=jnp.array(Labels.large),
    curriculum=True,
  )
  main_open_params = mazes.get_maze_reset_params(
    groups=group_set,
    char2key=char2key,
    maze_str=mazes.maze3_open,
    # label=jnp.array(Labels.shortcut),
  )

  train_params = pretrain_params + main_params
  train_params = multitask_env.EnvParams(
    reset_params=jtu.tree_map(lambda *v: jnp.stack(v), *train_params),
  )

  test_params = main_params + main_open_params
  test_params = multitask_env.EnvParams(
    reset_params=jtu.tree_map(lambda *v: jnp.stack(v), *test_params),
  ).replace(training=False)

  return train_params, test_params, task_objects


def maze3_randomize(config):
  """Maze 3: testing if open space is skipped. should be."""
  env_kwargs = config.get("rlenv", {}).get("ENV_KWARGS", {})
  num_groups = env_kwargs.pop("NUM_GROUPS", 1)
  char2key, group_set, task_objects = mazes.get_group_set(num_groups)

  pretrain_params = mazes.get_pretraining_reset_params(
    groups=group_set,
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
  train_params = multitask_env.EnvParams(
    randomize_agent=True,
    reset_params=jtu.tree_map(lambda *v: jnp.stack(v), *train_params),
  )

  test_params = main_params + main_open_params
  test_params = multitask_env.EnvParams(
    training=False,
    randomize_agent=False,
    reset_params=jtu.tree_map(lambda *v: jnp.stack(v), *test_params),
  )

  return train_params, test_params, task_objects


def maze5_two_paths(config):
  """Maze 3: testing if open space is skipped. should be."""
  env_kwargs = config.get("rlenv", {}).get("ENV_KWARGS", {})
  num_groups = env_kwargs.pop("NUM_GROUPS", 1)
  char2key, group_set, task_objects = mazes.get_group_set(num_groups)

  pretrain_params = mazes.get_pretraining_reset_params(
    groups=group_set,
  )
  main_params = mazes.get_maze_reset_params(
    groups=group_set,
    char2key=char2key,
    maze_str=mazes.maze5,
    curriculum=True,
  )

  train_params = pretrain_params + main_params
  train_params = multitask_env.EnvParams(
    reset_params=jtu.tree_map(lambda *v: jnp.stack(v), *train_params),
  )

  test_params = main_params
  test_params = multitask_env.EnvParams(
    training=False,
    reset_params=jtu.tree_map(lambda *v: jnp.stack(v), *test_params),
  )

  return train_params, test_params, task_objects


def basic_make_exp_block(
  config,
  train_mazes,
  eval_mazes,
  train_kwargs=None,
  eval_kwargs=None,
  pretrain_level=None,
  max_starting_locs=10,
):
  def make_int(i):
    return jnp.array(i, dtype=jnp.int32)

  train_kwargs = train_kwargs or dict()
  eval_kwargs = eval_kwargs or dict()

  # setup
  env_kwargs = config.get("rlenv", {}).get("ENV_KWARGS", {})
  num_groups = env_kwargs.pop("NUM_GROUPS", 2)
  char2key, group_set, task_objects = mazes.get_group_set(num_groups)

  all_train_params = []
  all_eval_params = []

  all_mazes = list(set(train_mazes + eval_mazes))
  maze2idx = {
    maze_name: idx for idx, maze_name in enumerate(all_mazes + [pretrain_level])
  }
  if pretrain_level:
    all_train_params += mazes.get_maze_reset_params(
      groups=group_set,
      char2key=char2key,
      maze_str=getattr(mazes, pretrain_level),
      label=make_int(maze2idx[pretrain_level]),
      curriculum=True,
      swap_train_test=True,
      max_starting_locs=max_starting_locs,
    )

  for maze_name in train_mazes:
    params = mazes.get_maze_reset_params(
      groups=group_set,
      char2key=char2key,
      maze_str=getattr(mazes, maze_name),
      label=make_int(maze2idx[maze_name]),
      curriculum=True,
      max_starting_locs=max_starting_locs,
    )
    all_train_params += params
    all_eval_params += params

  for maze_name in eval_mazes:
    params = mazes.get_maze_reset_params(
      groups=group_set,
      char2key=char2key,
      maze_str=getattr(mazes, maze_name),
      label=make_int(maze2idx[maze_name]),
    )
    all_eval_params += params

  train_params = multitask_env.EnvParams(
    **train_kwargs,
    reset_params=jtu.tree_map(lambda *v: jnp.stack(v), *all_train_params),
  )

  test_params = multitask_env.EnvParams(
    **eval_kwargs,
    training=False,
    reset_params=jtu.tree_map(lambda *v: jnp.stack(v), *all_eval_params),
  )

  label2name = {idx: name for idx, name in enumerate(all_mazes)}
  return train_params, test_params, task_objects, label2name


def exp1_block1(config):
  train_mazes = ["maze3"]
  eval_mazes = ["maze3_open2"]
  return basic_make_exp_block(config, train_mazes, eval_mazes)


def exp1_block2(config):
  train_mazes = ["maze3_r"]
  eval_mazes = ["maze3_onpath_shortcut_r", "maze3_offpath_shortcut_r"]
  return basic_make_exp_block(config, train_mazes, eval_mazes)


def exp1_block3(config):
  train_mazes = ["maze5"]
  eval_mazes = ["maze5"]
  return basic_make_exp_block(config, train_mazes, eval_mazes)


def exp1_block4(config):
  train_mazes = ["maze6"]
  eval_mazes = ["maze6_flipped_offtask"]
  return basic_make_exp_block(config, train_mazes, eval_mazes)


def exp1(config, analysis_eval: bool = False):
  train_mazes = ["maze3", "maze3_r", "maze5", "maze6"]
  if analysis_eval:
    eval_mazes = [
      "maze3_open2",
      "maze3_onpath_shortcut_r",
      "maze3_offpath_shortcut_r",
      "maze5",
      "maze6_flipped_offtask",
    ]
  else:
    eval_mazes = train_mazes
  return basic_make_exp_block(config, train_mazes, eval_mazes, max_starting_locs=10)


def exp2(config, analysis_eval: bool = False):
  train_mazes = ["big_m1_maze3", "big_m2_maze2", "big_m3_maze1"]
  if analysis_eval:
    eval_mazes = [
      "big_m1_maze3",
      "big_m1_maze3_shortcut",
      "big_m2_maze2",
      "big_m2_maze2_onpath",
      "big_m2_maze2_offpath",
      "big_m3_maze1",
    ]
  else:
    eval_mazes = train_mazes
  return basic_make_exp_block(
    config,
    train_mazes,
    eval_mazes,
    pretrain_level="big_practice_maze",
    max_starting_locs=20,
  )


def exp_test(config, analysis_eval: bool = False):
  del analysis_eval
  train_mazes = ["big_test_level"]
  eval_mazes = train_mazes
  return basic_make_exp_block(
    config,
    train_mazes,
    eval_mazes,
    pretrain_level="big_practice_maze",
    max_starting_locs=20,
  )
