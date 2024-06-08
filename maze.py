from typing import Optional, List, Callable, Tuple

from enum import IntEnum
from flax import struct
import jax
import jax.numpy as jnp


Grid = jax.Array
AgentPos = jax.Array
AgentDir = jax.Array
ActionOutput = Tuple[Grid, AgentPos, AgentDir]

# Map of agent direction indices to vectors
DIR_TO_VEC = jnp.array([
    ( 0, 1),  # right
    ( 1, 0),  # down
    ( 0, -1),  # left
    (-1, 0),  # up
], dtype=jnp.int8)


class Actions(IntEnum):
    left = 0    # Turn left
    right = 1   # Turn right
    forward = 2  # Move forward


class Observation(struct.PyTreeNode):
   image: jax.Array
   task_w: jax.Array
   state_features: jax.Array
   pocket: jax.Array
   position: jax.Array
   direction: jax.Array
   prev_action: jax.Array


@struct.dataclass
class MapInit:
    grid: jax.Array
    agent_pos: jax.Array
    agent_dir: jax.Array


@struct.dataclass
class ResetParams:
    map_init: MapInit
    train_objects: jax.Array
    test_objects: jax.Array
    starting_locs: Optional[jax.Array] = None
    curriculum: bool = False


@struct.dataclass
class EnvParams:
    reset_params: List[ResetParams]
    time_limit: int = 250
    training: bool = True

class TaskState(struct.PyTreeNode):
   features: jax.Array
   grid: jax.Array


class StepType(jnp.uint8):
    FIRST: jax.Array = jnp.asarray(0, dtype=jnp.uint8)
    MID: jax.Array = jnp.asarray(1, dtype=jnp.uint8)
    LAST: jax.Array = jnp.asarray(2, dtype=jnp.uint8)

def object_positions(grid, objects):
    def loc(p):
        y, x = jnp.nonzero(p, size=1)
        return y, x

    where_present = (grid == objects).astype(jnp.int32)
    y, x = jax.vmap(loc, 2)(where_present)
    object_positions = jnp.concatenate((y, x), axis=1)

    present = where_present.any(axis=(0, 1)).astype(jnp.int32)[:, None]
    not_found = jnp.full(object_positions.shape, -1)
    return object_positions*present + (1-present)*not_found

class TaskRunner(struct.PyTreeNode):
  """_summary_

  members:
      task_objects (jax.Array): [task_objects]

  Returns:
      _type_: _description_
  """
  task_objects: jax.Array
  convert_type: Callable[[jax.Array],
                         jax.Array] = lambda x: x.astype(jnp.int32)

  def task_vector(self, object):
     return self.convert_type((object[None] == self.task_objects))

  def reset(self, grid: jax.Array, agent_pos: jax.Array):
    """Get initial features.

    Args:
        visible_grid (GridState): _description_
        agent (AgentState): _description_

    Returns:
        _type_: _description_
    """
    features = self.convert_type(jnp.zeros_like(self.task_objects))
    return TaskState(
        features=features,
        grid=grid,
    )

  def step(self, prior_state: TaskState, grid: jax.Array, agent_pos: jax.Array):
    # was an object removed?
    def count(g):
       present = (g == self.task_objects[None, None])
       return present.sum(axis=(0, 1))

    count_prev = count(prior_state.grid)
    count_now = count(grid)

    decrease = count_prev - count_now
    return TaskState(
        grid=grid,
        features=self.convert_type(decrease)
    )

@struct.dataclass
class EnvState:
    # episode information
    key: jax.Array
    step_num: jax.Array

    # map info
    grid: jax.Array
    agent_pos: jax.Array
    agent_dir: int

    # task info
    task_runner: TaskRunner
    reset_params_idx: jax.Array
    task_w: jax.Array
    offtask_w: Optional[jax.Array] = None
    task_state: Optional[TaskState] = None

class TimeStep(struct.PyTreeNode):
    state: EnvState

    step_type: StepType
    reward: jax.Array
    discount: jax.Array
    observation: Observation

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST


def make_observation():
   return Observation(

   )



def take_action(
    state: EnvState,
    action: jax.Array) -> jax.Array:

    grid = state.grid
    agent_pos = state.agent_pos
    agent_dir = state.agent_dir

    # Update agent position (forward action)
    fwd_pos = jnp.minimum(
        jnp.maximum(agent_pos + (action == Actions.forward)
                    * DIR_TO_VEC[agent_dir], 0),
        jnp.array([grid.shape[0]-1, grid.shape[1]-1], dtype=jnp.uint32)
    )

    # Can't go past wall
    wall_map = grid == 1
    fwd_pos_has_wall = wall_map[fwd_pos[0], fwd_pos[1]]

    agent_pos = (fwd_pos_has_wall*state.agent_pos +
                 (~fwd_pos_has_wall)*fwd_pos).astype(jnp.uint32)

    # automatically "collect" (remove object) once go over it.
    # do so by setting to empty cell
    grid = grid.at[agent_pos[0], agent_pos[1]].set(0)

    # Update agent direction (left_turn or right_turn action)
    agent_dir_offset = 0 + (action == Actions.right) - \
        (action == Actions.left)
    agent_dir = (agent_dir + agent_dir_offset) % 4


    return grid, agent_pos, agent_dir

class HouseMaze:

    def __init__(self, task_runner_cls: TaskRunner = TaskRunner):
        self.task_runner_cls = task_runner_cls

    def reset(self, rng: jax.Array, params: EnvParams) -> TimeStep:
        """
        
        1. Sample level.
        """
        ##################
        # sample level
        ##################
        nlevels = len(params.reset_params)
        rng, rng_ = jax.random.split(rng)
        reset_params_idx = jax.random.randint(rng_, shape=(), minval=0, maxval=nlevels)
        reset_params = params.reset_params[reset_params_idx]

        grid = reset_params.map_init.grid
        agent_dir = reset_params.map_init.agent_dir

        ##################
        # sample pair
        ##################
        npairs = len(reset_params.train_objects)
        pair_idx = jax.random.randint(rng_, shape=(), minval=0, maxval=npairs)

        ##################
        # sample position (function of which pair has been choice)
        ##################
        def sample_pos_from_curriculum(rng_):
            locs = jax.lax.dynamic_index_in_dim(
                reset_params.starting_locs, pair_idx, keepdims=False,
            )
            return jax.random.choice(rng_, locs)

        rng, rng_ = jax.random.split(rng)
        agent_pos = jax.lax.cond(
            reset_params.curriculum,
            sample_pos_from_curriculum,
            lambda _: reset_params.map_init.agent_pos,
            rng_
        )

        ##################
        # sample task objects
        ##################
        train_object = jax.lax.dynamic_index_in_dim(
            reset_params.train_objects, pair_idx, keepdims=False,
        )
        test_object = jax.lax.dynamic_index_in_dim(
            reset_params.test_objects, pair_idx, keepdims=False,
        )

        def train_sample(rng):
            return train_object, test_object

        def test_sample(rng):
            return test_object, train_object

        def train_or_test_sample(rng):
            return jax.lax.cond(
                jax.random.bernoulli(rng),
                train_sample,
                test_sample,
                rng
            )
        rng, rng_ = jax.random.split(rng)
        task_object, offtask_object = jax.lax.cond(
            params.training,
            train_sample,
            train_or_test_sample,
            rng_
        )

        ##################
        # create task vectors
        ##################
        task_objects = jnp.concatenate(
            (reset_params.train_objects, reset_params.test_objects))
        task_runner = self.task_runner_cls(task_objects)

        task_w = task_runner.task_vector(task_object)
        offtask_w = task_runner.task_vector(offtask_object)
        task_state = task_runner.reset(grid, agent_pos)

        ##################
        # create ouputs
        ##################
        state = EnvState(
            key=rng,
            step_num=jnp.asarray(0),
            grid=grid,
            agent_pos=agent_pos,
            agent_dir=agent_dir,
            reset_params_idx=reset_params_idx,
            task_w=task_w,
            offtask_w=offtask_w,
            task_runner=task_runner,
            task_state=task_state,
        )

        timestep = TimeStep(
            state=state,
            step_type=StepType.FIRST,
            reward=jnp.asarray(0.0),
            discount=jnp.asarray(1.0),
            observation=None
        )
        return timestep

    def step(self, rng: jax.Array, timestep: TimeStep, action: jax.Array, params: EnvParams) -> TimeStep:
        grid, agent_pos, agent_dir = take_action(
            timestep.state,
            action)

        task_state = timestep.state.task_runner.step(
            timestep.state.task_state, grid, agent_pos)

        state = timestep.state.replace(
            grid=grid,
            agent_pos=agent_pos,
            agent_dir=agent_dir,
            task_state=task_state,
        )

        terminated = (task_state.features > 0).any()  # any object picked up
        reward = (timestep.state.task_w*task_state.features).sum(-1)
        truncated = jnp.equal(state.step_num, params.time_limit)

        step_type = jax.lax.select(
            terminated | truncated, StepType.LAST, StepType.MID)
        discount = jax.lax.select(
            terminated, jnp.asarray(0.0), jnp.asarray(1.0))

        timestep = TimeStep(
            state=state,
            step_type=step_type,
            reward=reward,
            discount=discount,
            observation=None,
        )
        return timestep



