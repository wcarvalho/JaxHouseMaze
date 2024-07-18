import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from housemaze.human_dyna import env as maze
from housemaze.human_dyna.utils import make_reset_params
from housemaze.utils import from_str

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

maze4 = """
A..#....#..C.
.B.#.#####..D
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


def make_env_params(group_set, maze_str):

  max_objects = 3
  list_of_reset_params = []
  all_starting_locs = np.ones((len(group_set), 10, 2))*-1

  train_objects = group_set[:max_objects, 0]
  test_objects = group_set[:max_objects, 1]

  
  map_init = maze.MapInit(*from_str(
      maze_str,
      char_to_key=dict(
          A=group_set[0, 0],
          B=group_set[0, 1],
          C=group_set[1, 0],
          D=group_set[1, 1],
          E=group_set[2, 0],
          F=group_set[2, 1],
      )))
  list_of_reset_params.append(
    make_reset_params(
        map_init=map_init,
        train_objects=train_objects,
        test_objects=test_objects,
        max_objects=max_objects,
        starting_locs=make_int_array(all_starting_locs),
        curriculum=jnp.array(False),
    ))
  return maze.EnvParams(
      reset_params=jtu.tree_map(
          lambda *v: jnp.stack(v), *list_of_reset_params),
  )
