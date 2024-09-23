from enum import IntEnum
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from housemaze.human_dyna import multitask_env as maze
from housemaze.human_dyna.utils import make_reset_params, load_groups
from housemaze.utils import from_str, from_str_spawning, find_optimal_path, reverse
from housemaze import levels as default_levels


class Labels(IntEnum):
    large = 0
    small = 1
    shortcut = 2


#################################################
# Copied from JaxUED
#################################################
StandardMaze = """
.....#>...#..
.###.####.##.
.#...........
.########.###
........#....
######.#####.
....#..#.....
.##...##.####
..#.#..#...#.
#.#.##.###.#.
#.#..#...#...
#.##.###.###.
...#...#.#...
"""


StandardMaze2 = """
...#.#....#..
.#.#.####...#
.#........#..
.########.###
...#..#.#.#..
##.#.##.#.#..
>#.#....#.##.
.#.##.###..#.
.#..#..###.#.
.##.##.#.#.#.
.#...#.#.#.#.
.#.#.#.#.#.#.
...#...#.....
"""


StandardMaze3 = """
...>#.#......
.####.#.####.
.#....#.#....
...####.#.#.#
##.#....#.#..
...#.##.#.##.
.#.#.#..#..#.
.#.#.#.###.##
.#...#.#.#...
.###.#.#.###.
.#...#.#...#.
.#.###.#.#.#.
.#...#...#...
"""

SixteenRooms = """
...#..#..#...
.>.......#...
...#..#......
#.###.##.###.
...#.........
......#..#...
##.#.##.###.#
...#.....#...
...#..#......
.####.##.#.##
...#..#..#...
......#......
...#.....#...
"""

SixteenRooms2 = """
...#.....#...
.>....#..#...
...#..#..#...
####.##.###.#
...#..#......
......#..#...
#.#####.#####
...#..#..#...
...#.........
##.##.##.####
...#..#..#...
......#......
...#..#..#...
"""

#################################################
# Custom mazes
#################################################

maze0 = """
.....#...A#..
.###.####.##.
.#........F..
.####.###.###
.............
######>#####.
...B#........
.##...##.####
..#.#..#.E.#.
#.#.##.###.#.
#.#.C#.D.#...
#.##.###.###.
...#...#.#...
""".strip()

maze1 = """
.#.C...##....
.#..D...####.
.######......
......######.
.#.#..#......
.#.#.##..#...
##.#.#>.###.#
A..#.##..#...
.B.#.........
#####.#..###.
......####.#.
.######E.#.#.
........F#...
""".strip()

maze2 = """
...#.....#...
......#..#...
...#..#..#E.F
####.##.###.#
...#..#......
.#...>#..#...
.######.#####
...#.....#...
.#.##.##.####
.#.#..#..#..A
##....#......
C..#..#..#...
.D.#..#..#.B.
""".strip()

maze3 = """
E..#.....#A..
......#..#...
F..#..#..#..B
####.##.##.##
...#.#.......
.#...##..#...
.############
...#.........
.#.##.##.##.#
.#.#..#..#...
##...#####...
.>.#.##..#C.D
...#....##...
""".strip()

maze3_onpath = """
E..#.....#A..
......#..#...
F..#..#..#..B
####.##.##.##
...#.#.......
.#...##..#...
>############
...#.........
.#.##.##.##.#
.#.#..#..#...
##...#####...
...#.##..#C.D
...#....##...
""".strip()


maze3_onpath_shortcut = """
E..#.....#A..
......#..#...
F..#..#..#..B
####.##.##.##
...#.#.......
.#...##..#...
>###.#######.
...#.........
.#.##.#.###.#
.#.#..#..#...
##...###.#...
...#.##..#C.D
...#....##...
""".strip()


maze3_offpath_shortcut = """
E..#.....#A..
......#..#...
F..#..#..#..B
####.##.##.##
...#.#.......
.#...##..#...
.###.#######.
...#.........
.#.##.#.###.#
.#.#..#..#...
##...###.#...
...#.##..#C.D
...#..>.##...
""".strip()

maze3_open = """
E..#.....#A..
......#..#...
F..#..#..#..B
####.##.##.##
...#.#.......
.#...##..#...
.######.#####
...#.........
.#.##.##.##.#
.#.#..#..#...
##...#####...
.>.#.##..#C.D
...#....##...
""".strip()

maze3_open2 = """
E..#.....#A..
......#..#...
F..#..#..#..B
####.##.##.##
...#.#.......
.#...##..#...
.###########.
...#.....#...
.#.##.##.##.#
.#.#..#......
##...#####...
.>.#.##..#C.D
...#....##...
""".strip()

maze3_r = reverse(maze3)
maze3_onpath_shortcut_r = reverse(maze3_onpath_shortcut)
maze3_offpath_shortcut_r = reverse(maze3_offpath_shortcut)

maze4 = """
C..#....#..A.
.D.#.#####..B
.###...#.....
...###.#.....
.#.#...#..###
.#.#.###.....
##.#...#.##.#
...#.###.#...
...#..#......
.####.#.###.#
.....>#.##...
.##.###..#...
.........#E.F
""".strip()

maze5 = """
C.#......B...
.D#.######...
.##....#.....
..####.#...A.
...#.#.#.####
.#.#.#.#.....
.#...#.####.#
...#...#.##..
...#..##..##.
.#.##.#....#.
...#..#.##...
.######..#...
.>.......#E.F
""".strip()

maze6 = """
E..#.....#A.B
......##.#...
F..#..#..#.#.
####.##.##.##
...#.#.....#.
.#...##......
.###########.
...#......#..
.#.##.##.##.#
.#.#..#...###
##...####....
.>.#.##.##...
...#.....#C.D
""".strip()
maze6 = reverse(maze6)

maze6_flipped_offtask = """
E..#.....#A..
......##.#...
F..#..#..#.#.
####.##.##.##
...#.#.....#.
.#...##......
.###########.
...#......#B.
.#.##.##.##.#
.#.#..#...###
##...####....
.>.#.##.##...
...#.....#C.D
""".strip()
maze6_flipped_offtask = reverse(maze6_flipped_offtask)

#################################################
# Bigger mazes (13 x 26)
#################################################
# practice

big_practice_maze = """
..........................
##########################
..#.......#...A#....#..#..
#.##..###.#....##.#.##.##.
..#...#...#.B..#.......#..
#.###.########.####.######
..........................
###########>#####.####.##.
....#...C#................
.#####D..#.##.############
...#.###....#...#....#..#.
##.#.#.#.##.###.#....#..#.
.#...#.#..#...#....#......
.###.#.##.###.####.###.##.
.#..........#.#....#......
##########################
..........................
..........................
""".strip()

# manipulation 1
# TRAIN
big_m1_maze3 = """
..........................
##########################
...#@#@...#........#@#A...
.#.#.####...#.####.#.##.#B
.#........#......#.#....#.
.########.####.#.#.####...
...#.@#.#@#C...#.#....#@##
##.#.##.#.#..D##.#@##.#...
.#.#....#.#####@.#..#.#.#.
.#.##.###..#....###.#.#.#.
.#..#..###.#....#.#.#...#.
.#.###.#.#.##.###.#.#.###.
.#...#...#.##.#...#.#...#.
.#.#.#.#.#.##.#.#.#.###.#.
>..#...#.#......#...#...#.
##########################
..........................
..........................
""".strip()

# TEST
big_m1_maze3_shortcut = """
..........................
##########################
...#.#....#........#.#A...
.#.#.####...#.####.#.##.#B
.#........#......#.#....#.
.########.####.#.#.####...
...#..#.#.#C...#.#....#.##
##.#.##.#.#..D##.#.##.#...
.#.#....#.#####..#..#.#.#.
.#.##.###..#....###.#.#.#.
.#..#..###.#....#.#.#...#.
.#.###.#...##.###.#.#####.
.#...#...#.##.#...#.#...#.
.#.#.#.#.#.##.#.#.#.#.#.#.
>..#...#.#......#.........
##########################
..........................
..........................
""".strip()

# manipulation 2
# TRAIN
big_m2_maze2 = """
..........................
##########################
.....#...#.....@#@#@..#C.D
.#.#.#.#.#.#..###.###.##.#
.#.#.#.#...#....#...#..#.#
.#.#...##.##..#.###.##.#.#
.#.###..#..#.A#...#..#.#.@
.#..###.##.#.####@##.>.##.
.##.#....#.#.B....#..#....
..#.#.##.#.########.######
..#.#.#..#.......#@.......
###.########.###.########.
..#........#............#.
#...####.#.#..##.####.###.
..#....#.#.#..@#...@#....@
##########################
..........................
..........................
""".strip()

# TEST

big_m2_maze2_onpath = """
..........................
##########################
.....#...#.....@#@#@..#C.D
.#.#.#.#.#.#..###.###.##.#
.#.#.#.#...#....#...#..#.#
.#.#...##.##..#.###>##.#.#
.#.###..#..#.A#...#..#.#.@
.#..###.##.#.####@##...##.
.##.#....#.#.B....#..#....
..#.#.##.#.########.######
..#.#.#..#.......#@.......
###.########.###.########.
..#........#............#.
#...####.#.#..##.####.###.
..#....#.#.#..@#...@#....@
##########################
..........................
..........................
""".strip()
big_m2_maze2_offpath = """
..........................
##########################
.....#...#......#.#...#C.D
.#.#.#.#.#.#..###.###.##.#
.#.#.#.#...#....#...#..#.#
.#.#...##.##..#.###.##.#.#
.#.###..#..#.A#...#..#.#..
.#..###.##.#.####.##...##.
.##.#....#.#.B....#..#....
..#.#.##.#.########.######
..#.#.#..#.......#........
###.########.###.########.
..#.....>..#............#.
#...####.#.#..##.####.###.
..#....#.#.#...#....#.....
##########################
..........................
..........................
""".strip()


# manipulation 3
big_m3_maze1 = """
...##..........#......B...
...#..#####....#.######...
...#.###..#.####....#.....
...#......#.#..####.#...A.
#.#####.###.#...#.#.#.####
..........#...#.#.#.#....@
##.##########.#...#.####.#
...#..##....#...#...#@##..
.......#..#.#...#.###..##.
##.##.##..#####.###...#.#.
..........#@..#.#@#.#.....
#..##########.#.#.#.####.#
..##..##....#.###...#.##..
.#....@#..#.....#..##.@##.
.#.##.##..#####.##.#....#.
.#.#..#...#.....#@.#.##.@.
.#.#..#...#..#######..#...
.>.#.....@#...........#C.D
""".strip()


#################################################
# utils
#################################################

def groups_to_char2key(group_set):
    chars = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    char2key = dict()
    for idx, char in enumerate(chars):
        i, j = idx // 2, idx % 2
        if i >= len(group_set):
            break
        char2key[char] = group_set[i, j]
    return char2key

def get_group_set(num_groups, group_set = None):
    if group_set is None:
        list_of_groups = load_groups()
        group_set = list_of_groups[0]

    char2key = groups_to_char2key(group_set)

    assert num_groups <= 3
    task_group_set = group_set[:num_groups]
    task_objects = task_group_set.reshape(-1)

    return char2key, task_group_set, task_objects

def make_int_array(x): return jnp.asarray(x, dtype=jnp.int32)

def get_pretraining_reset_params(
    groups,
    make_env_params: bool = False,
    max_starting_locs: int = 16,
    pretrain_level: str = None,
    ):
    pretrain_level = pretrain_level or default_levels.two_objects
    list_of_reset_params = []
    # -------------
    # pretraining levels
    # -------------
    for group in groups:
      map_init = maze.MapInit(
        *from_str(
              pretrain_level,
              char_to_key=dict(A=group[0], B=group[1]),
              check_grid_letters=False),
        spawn_locs=from_str_spawning(pretrain_level)
        )
      list_of_reset_params.append(
          make_reset_params(
              map_init=map_init,
              train_objects=group[:1],
              test_objects=group[1:],
              max_objects=len(groups),
              label=jnp.array(1),
              starting_locs=make_int_array(
                  np.ones((len(groups), max_starting_locs, 2))*-1)
          )
      )
    if make_env_params:
        return maze.EnvParams(
            reset_params=jtu.tree_map(
                lambda *v: jnp.stack(v), *list_of_reset_params),
        )
    return list_of_reset_params

def get_maze_reset_params(
        groups,
        maze_str,
        char2key,
        num_starting_locs: int = None,
        max_starting_locs: int = 20,
        make_env_params: bool = False,
        curriculum: bool = False,
        label: jnp.ndarray = jnp.array(0),
        swap_train_test: bool = False,
        **kwargs,
    ):
    num_starting_locs = num_starting_locs or max_starting_locs
    train_objects = groups[:, 0]
    test_objects = groups[:, 1]
    if swap_train_test:
        train_objects = groups[:, 1]
        test_objects = groups[:, 0]
    map_init = maze.MapInit(
        *from_str(maze_str, char_to_key=char2key),
        spawn_locs=from_str_spawning(maze_str),
        )

    all_starting_locs = np.ones(
        (len(groups), max_starting_locs, 2))*-1

    if curriculum:
        for idx, goal in enumerate(train_objects):
            path = find_optimal_path(
                map_init.grid, map_init.agent_pos, np.array([goal]))
            max_path_locs = min(num_starting_locs, len(path))
            width = len(path)//max_path_locs
            starting_locs = np.array([path[i] for i in range(0, len(path), width)])
            all_starting_locs[idx, :len(starting_locs)] = starting_locs

    reset_params = make_reset_params(
        map_init=map_init,
        train_objects=train_objects,
        test_objects=test_objects,
        max_objects=len(groups),
        starting_locs=make_int_array(all_starting_locs),
        curriculum=jnp.array(curriculum),
        label=label,
        **kwargs,
    )
    if make_env_params:
        return maze.EnvParams(
            reset_params=jtu.tree_map(
                lambda *v: jnp.stack(v), *[reset_params]),
        )
    return [reset_params]
