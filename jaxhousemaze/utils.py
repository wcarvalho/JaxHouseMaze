from typing import Optional
from collections import deque

import jax
import os.path
import jax.numpy as jnp
import numpy as np
import pickle

def replace_color(image, old_color, new_color):
    # Convert the image and colors to JAX arrays if they aren't already
    image = np.asarray(image)
    old_color = np.asarray(old_color)
    new_color = np.asarray(new_color)

    # Create a mask where all pixels match the old_color
    mask = np.all(image == old_color, axis=-1)

    # Replace the color
    image[mask] = new_color

    return image


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


def load_image_dict(file: str = None, add_borders: bool = False):
    if file is None:
        current_file_path = os.path.abspath(__file__)
        current_directory = os.path.dirname(current_file_path)
        file = f"{current_directory}/image_data.pkl"
        print(f"No file specified for image dict.\nUsing: {file}")

    if not add_borders:
        def add_border(x): return x

    with open(file, 'rb') as f:
        image_dict = pickle.load(f)

    tile_size = image_dict['images'].shape[-2]

    images = image_dict['images']

    new_images = []
    for image in images:
        image = replace_color(image, (255, 255, 255), (0, 0, 0))
        image = add_border(image)
    new_images = np.array(new_images)

    extra_keys = [
        ('wall', np.tile([100, 100, 100], (tile_size, tile_size, 1))),
        ('empty', add_border(np.tile([0, 0, 0], (tile_size, tile_size, 1)))),
    ]

    for key, img in extra_keys:
        assert not key in image_dict['keys']
        image_dict['keys'] = [key] + image_dict['keys']
        image_dict['images'] = jnp.concatenate(
            (img[None], image_dict['images']))

    return image_dict

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
                object = make_element(key)
              elif isinstance(key, np.ndarray):
                object = key
              elif isinstance(key, np.int32):
                object = key
              else:
                raise NotImplementedError(f"type: {type(key)}")
              grid[y, x] = object
          else:
              raise RuntimeError(f"Unknown char: {char}")

  assert agent_pos is not None
  assert agent_dir is not None

  return grid, agent_pos, agent_dir


class AutoResetWrapper:

    def __init__(self, env):
        self._env = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)

    def __auto_reset(self, key, params, timestep):
        key, key_ = jax.random.split(key)
        return self._env.reset(key_, params)

    def step(self,
             key: jax.Array,
             prior_timestep,
             action,
             params):
        return jax.lax.cond(
            prior_timestep.last(),
            lambda: self.__auto_reset(key, params, prior_timestep),
            lambda: self._env.step(key, prior_timestep, action, params),
        )

