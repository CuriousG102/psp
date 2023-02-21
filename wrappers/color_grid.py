import enum
import logging

from gym import spaces
import numpy as np

from tia.Dreamer.dmc2gym import wrappers
import tia.Dreamer.local_dm_control_suite as suite


def _split_action_space(action_dims, num_colors):
    # TODO: Add support for more variation in the future.
    assert num_colors == 2 ** len(action_dims)
    return [2] * len(action_dims)


def _split_reward_space(reward_range, num_colors, max_evil):
    num_non_single = sum(isinstance(r, tuple) for r in reward_range)
    num_single = len(reward_range) - num_non_single
    if max_evil:
        assert num_colors >= len(reward_range)
    else:
        assert num_colors >= 2 * num_non_single + num_single
        assert num_non_single > 0 or len(reward_range) > 1

    if num_non_single == 0:
        assert num_colors == len(reward_range)
    colors_per_range = [
        num_colors // num_non_single - num_single
        if isinstance(r, tuple) else 1
        for r in reward_range
    ]
    num_left = num_colors - sum(colors_per_range)
    while num_left != 0:
        for i in range(len(reward_range)):
            if not isinstance(reward_range[i], tuple):
                continue
            colors_per_range[i] += 1
            num_left -= 1
            if num_left == 0:
                break

    return colors_per_range


def _get_min_colors_needed(evil_level, reward_range, action_dims_to_split):
    if evil_level is EvilEnum.MAXIMUM_EVIL:
        return len(reward_range) * 2 ** len(action_dims_to_split)
    elif evil_level is EvilEnum.EVIL_REWARD:
        num_non_single = sum(isinstance(r, tuple) for r in reward_range)
        num_single = len(reward_range) - num_non_single
        return 2 * num_non_single + num_single
    elif evil_level is EvilEnum.EVIL_ACTION:
        return 2 ** len(action_dims_to_split)
    elif evil_level is EvilEnum.EVIL_SEQUENCE:
        return 2
    elif evil_level is EvilEnum.MINIMUM_EVIL:
        return 2
    elif evil_level is EvilEnum.RANDOM:
        return 2
    elif evil_level is EvilEnum.NONE:
        return 0
    else:
        raise ValueError(f'{evil_level} not implemented.')


def _get_num_colors_log_message(num_colors_per_cell, extra):
    return (
        f'num_colors_per_cell {num_colors_per_cell} '
        f'cannot be satisfied. {extra} colors remain for each '
        f'cell.')


def _get_colors_for_action_and_reward_max_evil(
        num_colors_per_cell, action_dims_to_split, reward_range):
    colors_for_action = 2 ** len(action_dims_to_split)
    colors_for_reward = num_colors_per_cell // colors_for_action
    total = colors_for_action * colors_for_reward
    if total < num_colors_per_cell:
        # TODO: Test this logic
        extra = (
                num_colors_per_cell
                - colors_for_action * colors_for_reward)
        logging.warning(
            _get_num_colors_log_message(num_colors_per_cell, extra))
    return (
        _split_action_space(action_dims_to_split, colors_for_action),
        _split_reward_space(reward_range, colors_for_reward, True))


def _get_colors_for_evil_action(num_colors_per_cell, action_dims_to_split):
    colors_for_action = 2 ** len(action_dims_to_split)
    if colors_for_action < num_colors_per_cell:
        extra = num_colors_per_cell - colors_for_action
        logging.warning(
            _get_num_colors_log_message(num_colors_per_cell, extra))
    return _split_action_space(action_dims_to_split, num_colors_per_cell)


def _get_reward_idx(reward, reward_range, colors_per_reward_range):
    range_found = False
    start = 0
    for i, (r, num_colors_in_range) in enumerate(zip(
            reward_range, colors_per_reward_range)):
        if isinstance(r, tuple):
            range_min, range_max = r
            if range_min <= reward <= range_max:
                range_found = True
                break
        else:
            if r == reward:
                range_found = True
                break
        start += num_colors_in_range

    if not range_found:
        raise ValueError(f'Unexpected reward {reward}')
    if isinstance(r, tuple):
        num_colors_in_range = colors_per_reward_range[i]
        delta = (range_max - range_min) / num_colors_in_range
        return start + int((reward - range_min) / delta)
    else:
        return start


def _get_action_idx(action, action_dims_to_split, colors_per_action_dim):
    action = [
        action_ for i, action_ in enumerate(action)
        if i in action_dims_to_split
    ]

    for action_ in action:
        assert -1 <= action_ <= 1

    pow = 1
    for num_colors in colors_per_action_dim:
        pow *= num_colors
    action_idx = 0

    for num_colors, action_ in zip(colors_per_action_dim, action):
        delta = 2 / num_colors
        action_idx += (
                int((action_ + (1 - 1e-6)) / delta) * (pow // num_colors))
        pow //= num_colors

    return action_idx


def _get_background_image_from_color_grid(
    color_grid,
    height,
    width,
    num_cells_per_dim
):
    bg_image = color_grid
    bg_image = np.repeat(
        bg_image, height // num_cells_per_dim, axis=0)
    bg_image = np.concatenate([
        bg_image,
        np.repeat(
            bg_image[-1:, ...],
            height % num_cells_per_dim,
            axis=0)
    ], axis=0)
    bg_image = np.repeat(
        bg_image, width // num_cells_per_dim, axis=1)
    bg_image = np.concatenate([
        bg_image,
        np.repeat(
            bg_image[:, -1:, :],
            width % num_cells_per_dim,
            axis=1)
    ], axis=1)
    return bg_image


class EvilEnum(enum.Enum):
    MAXIMUM_EVIL = enum.auto()
    EVIL_REWARD = enum.auto()
    EVIL_ACTION = enum.auto()
    EVIL_SEQUENCE = enum.auto()
    MINIMUM_EVIL = enum.auto()
    RANDOM = enum.auto()
    NONE = enum.auto()


class DmcColorGridWrapper(wrappers.DMCWrapper):
    def __init__(
        self,
        domain_name,
        task_name,
        num_cells_per_dim,
        num_colors_per_cell,
        evil_level,
        action_dims_to_split=[],
        task_kwargs=None,
        visualize_reward={},
        from_pixels=False,
        height=84,
        width=84,
        camera_id=0,
        frame_skip=1,
        environment_kwargs=None
    ):
        assert 'random' in task_kwargs, 'please specify a seed, for deterministic behaviour'
        self._from_pixels = from_pixels
        self._height = height
        self._width = width
        self._camera_id = camera_id
        self._frame_skip = frame_skip

        # create task
        self._env = suite.load(
            domain_name=domain_name,
            task_name=task_name,
            task_kwargs=task_kwargs,
            visualize_reward=visualize_reward,
            environment_kwargs=environment_kwargs
        )

        # true and normalized action spaces
        self._true_action_space = wrappers._spec_to_box([self._env.action_spec()])
        self._norm_action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=self._true_action_space.shape,
            dtype=np.float32
        )

        self.evil_level = evil_level
        self.num_colors_per_cell = num_colors_per_cell
        self.num_cells_per_dim = num_cells_per_dim
        self.action_dims_to_split = action_dims_to_split
        if domain_name == 'cheetah' and task_name == 'run':
            self.reward_range = [(0, 1,)]
        elif domain_name == 'hopper' and task_name == 'stand':
            self.reward_range = [0, (.8, 1)]
        else:
            # TODO: Also support walker run.
            raise ValueError('Other tasks not supported')

        min_colors_needed = _get_min_colors_needed(
            evil_level, self.reward_range, self.action_dims_to_split)
        if self.num_colors_per_cell < min_colors_needed:
            raise ValueError(
                f'{num_colors_per_cell} insufficient for minimum colors '
                f'needed {min_colors_needed}')

        if evil_level is EvilEnum.MAXIMUM_EVIL:
            self.colors_per_action_dim, self.colors_per_reward_range = (
                _get_colors_for_action_and_reward_max_evil(
                    self.num_colors_per_cell,
                    self.action_dims_to_split,
                    self.reward_range
                )
            )
            self.num_action_indices = 1
            for n in self.colors_per_action_dim:
                self.num_action_indices *= n
        elif evil_level is EvilEnum.EVIL_REWARD:
            self.colors_per_reward_range = _split_reward_space(
                self.reward_range, self.num_colors_per_cell, False)
        elif evil_level is EvilEnum.EVIL_ACTION:
            self.colors_per_action_dim = _get_colors_for_evil_action(
                self.num_colors_per_cell, self.action_dims_to_split)

        np.random.seed(task_kwargs.get('random', 1))
        self._color_grid = np.random.randint(255, size=[
            self.num_cells_per_dim, self.num_cells_per_dim,
            self.num_colors_per_cell, 3])

         # create observation space
        if from_pixels:
            self._observation_space = spaces.Box(
                low=0, high=255, shape=[3, height, width], dtype=np.uint8
            )
        else:
            self._observation_space = wrappers._spec_to_box(
                self._env.observation_spec().values()
            )

        self._internal_state_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=self._env.physics.get_state().shape,
            dtype=np.float32
        )

        # set seed
        self.seed(seed=task_kwargs.get('random', 1))
        self.num_steps_taken = 0

    def _get_obs(self, time_step, action, reward):
        if self._from_pixels:
            obs = self.render(
                height=self._height,
                width=self._width,
                camera_id=self._camera_id
            )
            if self.evil_level != EvilEnum.NONE:
                mask = np.logical_and(
                    (obs[:, :, 2] > obs[:, :, 1]),
                    (obs[:, :, 2] > obs[:, :, 0]))  # hardcoded for dmc
                bg = self._get_background_image(time_step, action, reward)
                obs[mask] = bg[mask]
                obs = obs.copy()
        else:
            obs = wrappers._flatten_obs(time_step.observation)

        return obs

    def reset(self):
        self.num_steps_taken = 0
        return super().reset()

    def _get_background_image(self, time_step, action, reward):
        if ((action is None and reward is None)
                or self.evil_level is EvilEnum.RANDOM):
            random_idx = np.random.randint(
                self.num_colors_per_cell,
                size=[self.num_cells_per_dim, self.num_cells_per_dim, 1, 1])
            color_grid = np.take_along_axis(
                self._color_grid, random_idx, 2).squeeze()
        elif self.evil_level is EvilEnum.MAXIMUM_EVIL:
            reward_idx = _get_reward_idx(
                reward, self.reward_range, self.colors_per_reward_range)
            action_idx = _get_action_idx(
                action, self.action_dims_to_split, self.colors_per_action_dim)
            color_grid = self._color_grid[
                 :, :,
                 reward_idx * self.num_action_indices + action_idx,
                 :]
        elif self.evil_level is EvilEnum.EVIL_REWARD:
            reward_idx = _get_reward_idx(
                reward, self.reward_range, self.colors_per_reward_range)
            color_grid = self._color_grid[:, :, reward_idx, :]
        elif self.evil_level is EvilEnum.EVIL_ACTION:
            action_idx = _get_action_idx(
                action, self.action_dims_to_split, self.colors_per_action_dim)
            color_grid = self._color_grid[:, :, action_idx, :]
        elif self.evil_level is EvilEnum.EVIL_SEQUENCE:
            step_idx = self.num_steps_taken % self.num_colors_per_cell
            color_grid = self._color_grid[:, :, step_idx, :]
        elif self.evil_level is EvilEnum.MINIMUM_EVIL:
            random_idx = np.random.randint(self.num_colors_per_cell)
            color_grid = self._color_grid[:, :, random_idx, :]
        else:
            raise ValueError(f'{self.evil_level} not supported.')

        self.num_steps_taken += 1

        return _get_background_image_from_color_grid(
            color_grid, self._height, self._width, self.num_cells_per_dim)


