from jax import numpy as jnp
from keras_cv import models as keras_cv_models
import sam_keras

class Agent:

  configs = {}  # dict of dicts

  def __init__(self, obs_space, act_space, step, config):
    pass

  def dataset(self, generator_fn):
    raise NotImplementedError(
        'dataset(generator_fn) -> generator_fn')

  def policy(self, obs, state=None, mode='train'):
    raise NotImplementedError(
        "policy(obs, state=None, mode='train') -> act, state")

  def train(self, data, state=None):
    raise NotImplementedError(
        'train(data, state=None) -> outs, state, metrics')

  def report(self, data):
    raise NotImplementedError(
        'report(data) -> metrics')

  def save(self):
    raise NotImplementedError('save() -> data')

  def load(self, data):
    raise NotImplementedError('load(data) -> None')

  def sync(self):
    # This method allows the agent to sync parameters from its training devices
    # to its policy devices in the case of a multi-device agent.
    pass


class Env:

  def __init__(self, sam_seg_config=None):
    self._sam_seg_config = sam_seg_config
    if self._sam_seg_config and self._sam_seg_config.use_sam:
      self._sam_model = keras_cv_models.SegmentAnythingModel.from_preset(
        self._sam_seg_config.model_name)
      self._predictor = sam_keras.SAMPredictor(self._sam_model)
      self._mask_generator = sam_keras.SAMAutomaticMaskGenerator(
          predictor=self._predictor,
          **self._sam_seg_config['generator_config']
      )

  def _maybe_apply_sam_to_image(self, image, prefix):
    # TODO: Make sure the image format being passed in here is correct. It's
    #       expected to be [0, 1] floats in HWC, but mask generator expects
    #       [0, 255) uint8s in HWC, so we have to convert it.
    if self._sam_seg_config is None:
      return {}
    image = image * 255.
    image = jnp.cast(image, jnp.uint8)
    masks = self._mask_generator.generate(image, verbose=0)
    return {
      f'{prefix}_masks': [mask['segmentation'] for mask in masks]
    }

  def __len__(self):
    return 0  # Return positive integer for batched envs.

  def __bool__(self):
    return True  # Env is always truthy, despite length zero.

  def __repr__(self):
    return (
        f'{self.__class__.__name__}('
        f'len={len(self)}, '
        f'obs_space={self.obs_space}, '
        f'act_space={self.act_space})')

  @property
  def obs_space(self):
    # The observation space must contain the keys is_first, is_last, and
    # is_terminal. Commonly, it also contains the keys reward and image. By
    # convention, keys starting with log_ are not consumed by the agent.
    raise NotImplementedError('Returns: dict of spaces')

  @property
  def act_space(self):
    # The observation space must contain the keys action and reset. This
    # restriction may be lifted in the future.
    raise NotImplementedError('Returns: dict of spaces')

  def step(self, action):
    raise NotImplementedError('Returns: dict')

  def render(self):
    raise NotImplementedError('Returns: array')

  def close(self):
    pass


class Wrapper:

  def __init__(self, env):
    self.env = env

  def __len__(self):
    return len(self.env)

  def __bool__(self):
    return bool(self.env)

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    try:
      return getattr(self.env, name)
    except AttributeError:
      raise ValueError(name)


class Replay:

  def __len__(self):
    raise NotImplementedError('Returns: total number of steps')

  @property
  def stats(self):
    raise NotImplementedError('Returns: metrics')

  def add(self, transition, worker=0):
    raise NotImplementedError('Returns: None')

  def add_traj(self, trajectory):
    raise NotImplementedError('Returns: None')

  def dataset(self):
    raise NotImplementedError('Yields: trajectory')

  def prioritize(self, keys, priorities):
    pass

  def save(self):
    pass

  def load(self, data):
    pass
