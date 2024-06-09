from typing import Optional, List, Callable, Tuple


from enum import IntEnum
from gymnax.environments import spaces
from flax import struct
from functools import partial
import jax
import jax.numpy as jnp
import distrax


Grid = jax.Array
AgentPos = jax.Array
AgentDir = jax.Array
ActionOutput = Tuple[Grid, AgentPos, AgentDir]

# Map of agent direction indices to vectors
DIR_TO_VEC = jnp.array([
    (0, 1),  # right
    (1, 0),  # down
    (0, -1),  # left
    (-1, 0),  # up
], dtype=jnp.int8)


class MinigridActions(IntEnum):
    left = 0
    right = 1
    forward = 2


class KeyboardActions(IntEnum):
    right = 0
    down = 1
    left = 2
    up = 3


class Observation(struct.PyTreeNode):
   image: jax.Array
   task_w: jax.Array
   state_features: jax.Array
   position: jax.Array
   direction: jax.Array
   prev_action: jax.Array


@struct.dataclass
class MapInit:
    grid: jax.Array
    agent_pos: jax.Array
    agent_dir: jax.Array


@struct.dataclass
class EnvParams:
    map_init: MapInit
    objects: jax.Array
    time_limit: int = 250


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
    map_idx: jax.Array
    task_w: jax.Array
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


def make_binary_vector(obj, num_categories):
    binary_vector = jnp.zeros(num_categories)

    # Extract the category and color vectors from the obj
    category_idx = obj[0]

    # Set the corresponding indices in the binary vector to 1
    binary_vector = binary_vector.at[category_idx].set(1)

    return binary_vector


def position_to_two_hot(agent_position, grid_shape):
    # Extract the position and grid dimensions
    y, x = agent_position
    max_y, max_x = grid_shape

    # Initialize one-hot vectors
    one_hot_x = jnp.zeros(max_x)
    one_hot_y = jnp.zeros(max_y)

    # Set the corresponding positions to 1
    one_hot_x = one_hot_x.at[x].set(1)
    one_hot_y = one_hot_y.at[y].set(1)

    return jnp.concatenate((one_hot_x, one_hot_y))



def take_action(
        state: EnvState,
        action: jax.Array) -> jax.Array:

    grid = state.grid
    agent_pos = state.agent_pos
    agent_dir = state.agent_dir

    # Update agent position (forward action)
    fwd_pos = jnp.minimum(
        jnp.maximum(agent_pos + (action == MinigridActions.forward)
                    * DIR_TO_VEC[agent_dir], 0),
        jnp.array([grid.shape[0]-1, grid.shape[1]-1], dtype=jnp.int32)
    )

    # Can't go past wall
    wall_map = grid == 1
    fwd_pos_has_wall = wall_map[fwd_pos[0], fwd_pos[1]]

    agent_pos = (fwd_pos_has_wall*state.agent_pos +
                 (~fwd_pos_has_wall)*fwd_pos).astype(jnp.int32)

    # automatically "collect" (remove object) once go over it.
    # do so by setting to empty cell
    grid = grid.at[agent_pos[0], agent_pos[1]].set(0)

    # Update agent direction (left_turn or right_turn action)
    agent_dir_offset = 0 + (action == MinigridActions.right) - \
        (action == MinigridActions.left)
    agent_dir = (agent_dir + agent_dir_offset) % 4

    return grid, agent_pos, agent_dir


class HouseMaze:
    """Simple environment where you get reward for collecting an object.
    
    Episode ends when all objects are collected or at a time-limit."""

    def __init__(
            self,
            num_categories: int,
            task_runner: TaskRunner,
            action_spec: str = 'keyboard',
    ):
        self.num_categories = num_categories
        self.task_runner = task_runner
        self.action_spec = action_spec

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(self.num_actions(params))

    def num_actions(self, params: Optional[EnvParams] = None):
        if self.action_spec == 'keyboard':
            return 4
        elif self.action_spec == 'minigrid':
            return 3

    def action_onehot(self, action):
        num_actions = self.num_actions() + 1
        one_hot = jnp.zeros((num_actions))
        one_hot = one_hot.at[action].set(1)
        return one_hot

    def make_observation(
            self,
            state: EnvState,
            prev_action: jax.Array):
        """This converts all inputs into binary vectors. this faciitates processing with a neural network."""

        grid = state.grid
        agent_pos = state.agent_pos
        agent_dir = state.agent_dir

        # direction
        direction = jnp.zeros((len(DIR_TO_VEC)))
        direction = direction.at[agent_dir].set(1)

        make_binary_vector_ = lambda g: make_binary_vector(g, num_categories=self.num_categories)

        make_binary_vector_grid = jax.vmap(jax.vmap(make_binary_vector_))

        observation = Observation(
            image=make_binary_vector_grid(grid),
            state_features=state.task_state.features.astype(jnp.float32),
            task_w=state.task_w.astype(jnp.float32),
            direction=direction,
            position=position_to_two_hot(agent_pos, grid.shape[:2]),
            prev_action=self.action_onehot(prev_action),
        )
        # just to be safe?
        observation = jax.tree_map(lambda x: jax.lax.stop_gradient(x), observation)

        return observation

    def reset(self, rng: jax.Array, params: EnvParams) -> TimeStep:
        """
        Sample map and then sample random object in map as task object.
        """
        ##################
        # sample level
        ##################
        ndim = params.map_init.grid.ndim
        if ndim == 3:
            # single choice
            map_idx = jnp.array(0)
            map_init = params.map_init
        elif ndim == 4:
            # multiple to choose from
            nlevels = len(params.map_init.grid)
            rng, rng_ = jax.random.split(rng)

            # select one
            map_idx = jax.random.randint(
                rng_, shape=(), minval=0, maxval=nlevels)

            # index across each pytree member
            def index(p): return jax.lax.dynamic_index_in_dim(
                p, map_idx, keepdims=False)
            map_init = jax.tree_map(index, params.map_init)
        else:
            raise NotImplementedError

        grid = map_init.grid
        agent_dir = map_init.agent_dir
        agent_pos = map_init.agent_pos

        ##################
        # sample task object
        ##################
        present_objects = (grid == params.objects[None, None])
        present_objects = present_objects.any(axis=(0,1))
        #jax.debug.print("{x}", x=present_objects)
        object_sampler = distrax.Categorical(
            logits=present_objects.astype(jnp.float32))
        rng, rng_ = jax.random.split(rng)
        object_idx = object_sampler.sample(seed=rng_)
        task_object = jax.lax.dynamic_index_in_dim(
            params.objects, object_idx, keepdims=False,
        )

        ##################
        # create task vectors
        ##################
        task_w = self.task_runner.task_vector(task_object)
        task_state = self.task_runner.reset(grid, agent_pos)

        ##################
        # create ouputs
        ##################
        state = EnvState(
            key=rng,
            step_num=jnp.asarray(0),
            grid=grid,
            agent_pos=agent_pos,
            agent_dir=agent_dir,
            map_idx=map_idx,
            task_w=task_w,
            task_state=task_state,
        )

        reset_action = self.num_actions() + 1
        timestep = TimeStep(
            state=state,
            step_type=StepType.FIRST,
            reward=jnp.asarray(0.0),
            discount=jnp.asarray(1.0),
            observation=self.make_observation(
                state,
                prev_action=reset_action)
        )
        return timestep

    def step(self, rng: jax.Array, timestep: TimeStep, action: jax.Array, params: EnvParams) -> TimeStep:

        if self.action_spec == 'keyboard':
            grid, agent_pos, agent_dir = take_action(
                timestep.state.replace(agent_dir=action),
                action=MinigridActions.forward)
        elif self.action_spec == 'minigrid':
            grid, agent_pos, agent_dir = take_action(
                timestep.state,
                action)
        else:
            raise NotImplementedError(self.action_spec)

        task_state = self.task_runner.step(
            timestep.state.task_state, grid, agent_pos)

        state = timestep.state.replace(
            grid=grid,
            agent_pos=agent_pos,
            agent_dir=agent_dir,
            task_state=task_state,
        )

        task_w = timestep.state.task_w.astype(jnp.float32)
        features = task_state.features.astype(jnp.float32)
        reward = (task_w*features).sum(-1)
        terminated = reward > 0  # get task object
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
            observation=self.make_observation(
                state,
                prev_action=action),
        )
        return timestep
