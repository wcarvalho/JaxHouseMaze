from typing import Optional

from flax import struct
import jax
import jax.numpy as jnp

import numpy as np

# Map of agent direction indices to vectors
DIR_TO_VEC = jnp.array([
    (1, 0),  # right
    (0, 1),  # down
    (-1, 0),  # left
    (0, -1),  # up
], dtype=jnp.int8)


class Observation(struct.PyTreeNode):
   image: jax.Array
   task_w: jax.Array
   state_features: jax.Array
   has_occurred: jax.Array
   pocket: jax.Array
   direction: jax.Array
   position: jax.Array
   prev_action: jax.Array


@struct.dataclass
class EnvParams:
    time_limit: int = 250

class StepType(jnp.uint8):
    FIRST: jax.Array = jnp.asarray(0, dtype=jnp.uint8)
    MID: jax.Array = jnp.asarray(1, dtype=jnp.uint8)
    LAST: jax.Array = jnp.asarray(2, dtype=jnp.uint8)


class TaskState(struct.PyTreeNode):
   feature_counts: jax.Array
   features: jax.Array

@struct.dataclass
class EnvState:
    # episode information
    key: jax.Array
    step_num: jax.Array

    # map info
    agent_pos: jax.Array
    agent_dir: int
    grid: jax.Array

    # task info
    task_w: jax.Array
    offtask_w: Optional[jax.Array] = None
    task_state: Optional[TaskState] = None


class TimeStep(struct.PyTreeNode):
    state: EnvState

    step_type: StepType
    reward: jax.Array
    discount: jax.Array
    observation: jax.Array

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

def from_str(
        level_str: str,
        char_to_key: dict,
        object_to_index: dict):

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
              grid[y, x] = make_element('wall')
          elif char == '>':
              assert agent_pos is None, "Agent position can only be set once."
              agent_pos, agent_dir = (x, y), 0
          elif char == 'v':
              assert agent_pos is None, "Agent position can only be set once."
              agent_pos, agent_dir = (x, y), 1
          elif char == '<':
              assert agent_pos is None, "Agent position can only be set once."
              agent_pos, agent_dir = (x, y), 2
          elif char == '^':
              assert agent_pos is None, "Agent position can only be set once."
              agent_pos, agent_dir = (x, y), 3
          elif char in char_to_key:
              key = char_to_key[char]
              grid[y, x] = make_element(key)
          else:
              raise RuntimeError(f"Unknown char: {char}")

  assert agent_pos is not None
  assert agent_dir is not None

  return grid, agent_pos, agent_dir

class HouseMaze:

    def reset(self, params: EnvParams, key: jax.Array) -> TimeStep:
        import ipdb; ipdb.set_trace()
        #state = self._generate_problem(params, key)
        #timestep = TimeStep(
        #    state=state,
        #    step_type=StepType.FIRST,
        #    reward=jnp.asarray(0.0),
        #    discount=jnp.asarray(1.0),
        #    observation=transparent_field_of_view(
        #        state.grid, state.agent, params.view_size, params.view_size),
        #)
        #return timestep

    def step(self, params: EnvParams, timestep: TimeStep, action: jax.Array) -> TimeStep:
        import ipdb; ipdb.set_trace()
        #new_grid, new_agent, changed_position = take_action(
        #    timestep.state.grid, timestep.state.agent, action)
        #new_grid, new_agent = check_rule(
        #    timestep.state.rule_encoding, new_grid, new_agent, action, changed_position)

        #new_state = timestep.state.replace(
        #    grid=new_grid,
        #    agent=new_agent,
        #    step_num=timestep.state.step_num + 1,
        #)
        #new_observation = transparent_field_of_view(
        #    new_state.grid, new_state.agent, params.view_size, params.view_size)

        ## checking for termination or truncation, choosing step type
        #terminated = check_goal(
        #    new_state.goal_encoding, new_state.grid, new_state.agent, action, changed_position)
        #truncated = jnp.equal(new_state.step_num, self.time_limit(params))

        #reward = jax.lax.select(
        #    terminated, 1.0 - 0.9 * (new_state.step_num / self.time_limit(params)), 0.0)

        #step_type = jax.lax.select(
        #    terminated | truncated, StepType.LAST, StepType.MID)
        #discount = jax.lax.select(
        #    terminated, jnp.asarray(0.0), jnp.asarray(1.0))

        #timestep = TimeStep(
        #    state=new_state,
        #    step_type=step_type,
        #    reward=reward,
        #    discount=discount,
        #    observation=new_observation,
        #)
        #return timestep



def make_grid():
   pass

def make_observation():
   return Observation(
      
   )
