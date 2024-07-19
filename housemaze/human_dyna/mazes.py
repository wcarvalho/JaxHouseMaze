import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from housemaze.human_dyna import env as maze
from housemaze.human_dyna.utils import make_reset_params
from housemaze.utils import from_str, find_optimal_path
from housemaze import levels as default_levels

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
#####.#..####
......####.#.
.######E.#.#.
........F#...
""".strip()

maze2 = """
...#.....#...
......#..#E.F
...#..#..#...
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
##...#####.C.
.>.#.##..#...
...#....##.D.
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
##...#####.C.
.>.#.##..#...
...#....##.D.
""".strip()


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
.##.###..#E.F
.........#...
""".strip()




def make_int_array(x): return jnp.asarray(x, dtype=jnp.int32)

def get_pretraining_reset_params(
    group_set,
    make_env_params: bool = False,
    max_starting_locs: int = 10,
    ):
    pretrain_level = default_levels.two_objects
    list_of_reset_params = []
    # -------------
    # pretraining levels
    # -------------
    for group in group_set:
      list_of_reset_params.append(
          make_reset_params(
              map_init=maze.MapInit(*from_str(
                  pretrain_level, char_to_key=dict(A=group[0], B=group[1]))),
              train_objects=group[:1],
              test_objects=group[1:],
              max_objects=len(group_set),
              label=jnp.array(1),
              starting_locs=make_int_array(
                  np.ones((len(group_set), max_starting_locs, 2))*-1)
          )
      )
    if make_env_params:
        return maze.EnvParams(
            reset_params=jtu.tree_map(
                lambda *v: jnp.stack(v), *list_of_reset_params),
        )
    return list_of_reset_params

def get_maze_reset_params(
        group_set,
        maze_str,
        char2key,
        num_starting_locs: int = 4,
        max_starting_locs: int = 10,
        make_env_params: bool = False,
        curriculum: bool = False,
    ):
    train_objects = group_set[:, 0]
    test_objects = group_set[:, 1]

    all_starting_locs = np.ones(
        (len(group_set), max_starting_locs, 2))*-1
    if curriculum:
        for idx, goal in enumerate(train_objects):
            path = find_optimal_path(
                map_init.grid, map_init.agent_pos, np.array([goal]))
            width = len(path)//num_starting_locs
            starting_locs = np.array([path[i] for i in range(0, len(path), width)])
            all_starting_locs[idx, :len(starting_locs)] = starting_locs

    map_init = maze.MapInit(*from_str(
        maze_str,
        char_to_key=char2key))
    reset_params = make_reset_params(
        map_init=map_init,
        train_objects=train_objects,
        test_objects=test_objects,
        max_objects=len(group_set),
        starting_locs=make_int_array(all_starting_locs),
        curriculum=jnp.array(curriculum),
    )
    if make_env_params:
        return maze.EnvParams(
            reset_params=jtu.tree_map(
                lambda *v: jnp.stack(v), *[reset_params]),
        )
    return [reset_params]