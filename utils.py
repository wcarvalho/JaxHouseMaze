from typing import Optional
import jax
import jax.numpy as jnp
from collections import deque

import numpy as np


def sample_groups(
        categories,
        ordered_keys,
        rng,
        num_groups: int = 4,
        elem_per_group: int = 2):
    potential_groups = list(categories.values())
    potential_groups = [[ordered_keys.index(i) for i in g] for g in potential_groups]
    assert num_groups >= len(potential_groups)

    # Splitting the key for sampling group indices
    key1, key2 = jax.random.split(rng)
    groups_idxs = jax.random.choice(key1, jnp.arange(
        len(potential_groups)), shape=(num_groups,), replace=False)

    groups = []
    for idx in groups_idxs:
        # Split the key for each iteration to ensure different randomness
        key2, subkey = jax.random.split(key2)
        groups.append(
            jax.random.choice(subkey, jnp.array(potential_groups[idx]), shape=(elem_per_group,), replace=False))

    return jnp.array(groups)


def sample_n_groups(
        categories,
        ordered_keys,
        n: int,
        num_groups: int = 4,
        elem_per_group: int = 2):
    list_of_groups = []
    for seed in range(n):
        rng = jax.random.PRNGKey(seed)
        groups = sample_groups(
            categories=categories,
            ordered_keys=ordered_keys,
            rng=rng,
            num_groups=num_groups,
            elem_per_group=elem_per_group)
        list_of_groups.append(groups)
    return jnp.array(list_of_groups)


def find_optimal_path(grid, agent_pos, goal):
    rows, cols, _ = grid.shape
    queue = deque([(agent_pos, [agent_pos])])
    visited = set()

    while queue:
        current_pos, path = queue.popleft()

        if grid[current_pos[0], current_pos[1], 0] == goal:

            return np.array([p for p in path])

        visited.add(current_pos)

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_x, new_y = current_pos[0] + dx, current_pos[1] + dy

            if (
                0 <= new_x < rows
                and 0 <= new_y < cols
                and (new_x, new_y) not in visited
                and grid[new_x, new_y, 0] != 1
            ):
                new_path = path + [(new_x, new_y)]
                queue.append(((new_x, new_y), new_path))

    return None


def from_str(
        level_str: str,
        char_to_key: dict,
        object_to_index: Optional[dict] = None):

  level_str = level_str.strip()
  rows = level_str.split('\n')
  nrows = len(rows)
  assert all(len(row) == len(rows[0])
              for row in rows), "All rows must have same length"
  ncols = len(rows[0])

  def make_element(k: str):
      return np.array([object_to_index[k]], dtype=np.uint8)
  grid = np.zeros((nrows, ncols, 1), dtype=np.uint8)

  agent_pos = None
  agent_dir = None

  for y, row in enumerate(rows):
      for x, char in enumerate(row):
          if char == '.':  # EMPTY
              continue
          elif char == '#': # WALL
              grid[y, x] = np.array(1)
          elif char == '>':
              assert agent_pos is None, "Agent position can only be set once."
              agent_pos, agent_dir = (y, x), 0
          elif char == 'v':
              assert agent_pos is None, "Agent position can only be set once."
              agent_pos, agent_dir = (y, x), 1
          elif char == '<':
              assert agent_pos is None, "Agent position can only be set once."
              agent_pos, agent_dir = (y, x), 2
          elif char == '^':
              assert agent_pos is None, "Agent position can only be set once."
              agent_pos, agent_dir = (y, x), 3
          elif char in char_to_key:
              key = char_to_key[char]
              if isinstance(key, str):
                grid[y, x] = make_element(key)
              elif isinstance(key, np.ndarray):
                grid[y, x] = key
              elif isinstance(key, np.int32):
                grid[y, x] = key
              else:
                raise NotImplementedError(f"type: {type(key)}")
          else:
              raise RuntimeError(f"Unknown char: {char}")

  assert agent_pos is not None
  assert agent_dir is not None

  return grid, agent_pos, agent_dir


#class AutoResetWrapper(Wrapper):

#    def __auto_reset(self, key, params, timestep):
#        key, key_ = jax.random.split(key)
#        return self._env.reset(key_, params)

#    def step(self,
#             key: jax.random.KeyArray,
#             prior_timestep,
#             action,
#             params):
#        return jax.lax.cond(
#            prior_timestep.last(),
#            lambda: self.__auto_reset(key, params, prior_timestep),
#            lambda: self._env.step(key, prior_timestep, action, params),
#        )
