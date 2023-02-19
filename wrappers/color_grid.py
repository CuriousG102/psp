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
    # colors_per_dim = [
    #     num_colors // len(action_dims) for _ in range(len(action_dims))]
    # for i in range(num_colors % len(action_dims)):
    #     colors_per_dim[i] += 1
    # return colors_per_dim


def _split_reward_space(reward_range, num_colors, max_evil):
    if max_evil:
        assert num_colors >= len(reward_range)
    else:
        assert num_colors >= len(reward_range) * 2
    colors_per_range = [
        num_colors // len(reward_range) for _ in range(len(reward_range))]
    for i in range(num_colors % len(reward_range)):
        colors_per_range[i] += 1
    return colors_per_range


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

        if evil_level == EvilEnum.MAXIMUM_EVIL:
            min_colors_needed = (
                    len(self.reward_range)
                    * 2 ** len(self.action_dims_to_split))
        elif evil_level == EvilEnum.EVIL_REWARD:
            min_colors_needed = 2 * len(self.reward_range)
        elif evil_level == EvilEnum.EVIL_ACTION:
            min_colors_needed = 2 ** len(self.action_dims_to_split)
        else:
            min_colors_needed = 0

        if self.num_colors_per_cell < min_colors_needed:
            raise ValueError(
                f'{num_colors_per_cell} insufficient for minimum colors '
                f'needed {min_colors_needed}')

        if evil_level == EvilEnum.MAXIMUM_EVIL:
            colors_for_action = 2 ** len(self.action_dims_to_split)
            colors_for_reward = (
                self.num_colors_per_cell // colors_for_action)
            total = colors_for_action * colors_for_reward
            if total < self.num_colors_per_cell:
                # TODO: Test this logic
                extra = (
                        self.num_colors_per_cell
                        - colors_for_action * colors_for_reward)
                logging.warn(
                    f'num_colors_per_cell {self.num_colors_per_cell} '
                    f'cannot be satisfied. {extra} colors remain for each '
                    f'cell.')
            self.colors_per_action_dim = _split_action_space(
                self.action_dims_to_split, colors_for_action)
            self.colors_per_reward_range = _split_reward_space(
                self.reward_range, colors_for_reward, True)
        elif evil_level == EvilEnum.EVIL_REWARD:
            self.colors_per_reward_range = _split_reward_space(
                self.reward_range, self.num_colors_per_cell, False)
        elif evil_level == EvilEnum.EVIL_ACTION:
            colors_for_action = 2 ** len(self.action_dims_to_split)
            if colors_for_action < self.num_colors_per_cell:
                extra = self.num_colors_per_cell - colors_for_action
                logging.warn(
                    f'num_colors_per_cell {self.num_colors_per_cell} '
                    f'cannot be satisfied. {extra} colors remain for each '
                    f'cell.')
            self.colors_per_action_dim = _split_action_space(
                self.action_dims_to_split, self.num_colors_per_cell)

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

    def _get_reward_idx(self, reward):
        range_found = False
        for i, (range_min, range_max) in enumerate(self.reward_range):
            if range_min <= reward <= range_max:
                range_found = True
                break
        if not range_found:
            raise ValueError(f'Unexpected reward {reward}')
        num_colors_in_range = self.colors_per_reward_range[i]
        delta = (range_max - range_min) / num_colors_in_range
        return int((reward - range_min) / delta)

    def _get_action_idx(self, action):
        action = [
            action_ for i, action_ in enumerate(action)
            if i in self.action_dims_to_split
        ]

        for action_ in action:
            assert -1 <= action_ <= 1

        pow = 1
        for num_colors in self.colors_per_action_dim:
            pow *= num_colors
        action_idx = 0

        for num_colors, action_ in zip(self.colors_per_action_dim, action):
            delta = 2 / num_colors
            action_idx += (
                    int((action_ + (1 - 1e-6)) / delta) * (pow // num_colors))
            pow //= num_colors

        return action_idx


    def _get_background_image(self, time_step, action, reward):
        if ((action is None and reward is None)
                or self.evil_level is EvilEnum.RANDOM):
            random_idx = np.random.randint(
                self.num_colors_per_cell,
                size=[self.num_cells_per_dim, self.num_cells_per_dim, 1, 1])
            color_grid = np.take_along_axis(
                self._color_grid, random_idx, 2).squeeze()
        elif self.evil_level is EvilEnum.MAXIMUM_EVIL:
            reward_idx = self._get_reward_idx(reward)
            action_idx = self._get_action_idx(action)
            color_grid = self._color_grid[
                 :, :,
                 (reward_idx * len(self.colors_per_action_dim)) + action_idx,
                 :]
        elif self.evil_level is EvilEnum.EVIL_REWARD:
            reward_idx = self._get_reward_idx(reward)
            color_grid = self._color_grid[:, :, reward_idx, :]
        elif self.evil_level is EvilEnum.EVIL_ACTION:
            action_idx = self._get_action_idx(action)
            color_grid = self._color_grid[:, :, action_idx, :]
        elif self.evil_level is EvilEnum.EVIL_SEQUENCE:
            step_idx = time_step % self.num_colors_per_cell
            color_grid = self._color_grid[:, :, step_idx, :]
        elif self.evil_level is EvilEnum.MINIMUM_EVIL:
            random_idx = np.random.randint(self.num_colors_per_cell)
            color_grid = self._color_grid[:, :, random_idx, :]
        else:
            raise ValueError(f'{self.evil_level} not supported.')

        bg_image = color_grid
        bg_image = np.repeat(
            bg_image, self._height // self.num_cells_per_dim, axis=0)
        bg_image = np.concatenate([
            bg_image,
            np.repeat(
                bg_image[-1:, ...],
                self._height % self.num_cells_per_dim,
                axis=0)
        ], axis=0)
        bg_image = np.repeat(
            bg_image, self._width // self.num_cells_per_dim, axis=1)
        bg_image = np.concatenate([
            bg_image,
            np.repeat(
                bg_image[:, -1:, :],
                self._width % self.num_cells_per_dim,
                axis=1)
        ], axis=1)
        return bg_image


