import warnings

import dreamerv3
from dreamerv3 import embodied
from dreamerv3.embodied.envs import color_dmc
from dreamerv3.embodied.envs import from_gym
from wrappers import color_grid_utils


def main(argv=None):
    warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

    parsed, other = embodied.Flags(configs=['defaults']).parse_known(argv)
    config = embodied.Config(dreamerv3.configs['defaults'])
    config = config.update(dreamerv3.configs['medium'])
    config = config.update(dreamerv3.configs['dmc_vision'])
    for name in parsed.configs:
        config = config.update(dreamerv3.configs[name])
    config = embodied.Flags(config).parse(other)
    args = embodied.Config(
        **config.run, logdir=config.logdir,
        batch_steps=config.batch_size * config.batch_length)
    print(config)

    logdir = embodied.Path(config.logdir)
    logdir.mkdirs()
    config.save(logdir / 'config.yaml')
    step = embodied.Counter()
    logger = embodied.Logger(step, [
        embodied.logger.TerminalOutput(),
        embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
        embodied.logger.TensorBoardOutput(logdir),
    ])

    env = color_dmc.DMC(
        'cheetah_run',
        repeat=config.env.dmc.repeat,
        size=config.env.dmc.size,
        camera=config.env.dmc.camera,
        num_cells_per_dim=config.evil.num_cells_per_dim,
        num_colors_per_cell=config.evil.num_colors_per_cell,
        evil_level=color_grid_utils.EVIL_CHOICE_CONVENIENCE_MAPPING[
            config.evil.evil_level
        ],
        action_dims_to_split=config.evil.action_dims_to_split,
        action_power=(
            config.evil.action_power if config.evil.action_power >= 0
            else None),
        action_splits=(
            config.evil.action_splits if config.evil.action_power < 0
            else None),
    )
    env = dreamerv3.wrap_env(env, config)
    env = embodied.BatchEnv([env], parallel=False)

    agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
    replay = embodied.replay.Uniform(
        config.batch_length, config.replay_size, logdir / 'replay')

    embodied.run.train(agent, env, replay, logger, args)


if __name__ == '__main__':
    main()