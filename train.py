import argparse
import logging
import sys

import chainer
import chainerrl
import gym
import numpy as np

from model import QFunction


def main(args):
    args.out = chainerrl.experiments.prepare_output_dir(
        args, args.out, argv=sys.argv)
    print('Output files are saved in {}'.format(args.out))

    # Setup a logger
    gym.undo_logger_setup()
    logging.basicConfig(level=logging.INFO, format='')

    # Make an environment
    env = gym.make(args.env_id)

    # Setup a model(q_func)
    q_func = QFunction(env.action_space.n)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        q_func.to_gpu()

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(q_func)

    # Create an agent
    replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)
    explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(
        args.start_epsilon, args.end_epsilon, args.decay_steps,
        env.action_space.sample)
    agent = chainerrl.agents.DQN(
        q_func, optimizer, replay_buffer, args.gamma, explorer,
        gpu=args.gpu,
        replay_start_size=500, update_interval=1, target_update_interval=100,
        phi=lambda x: x.astype(np.float32, copy=False))

    # Train
    chainerrl.experiments.train_agent_with_evaluation(
        agent, env,
        steps=args.steps,
        eval_n_runs=args.eval_n_runs,
        max_episode_len=env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps'),
        eval_interval=args.eval_interval,
        outdir=args.out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='Breakout-v0',
                        help='Select the environment to run.')
    parser.add_argument('--steps', '-s', type=int, default=1000000,
                        help='Number of total time steps for training.')
    parser.add_argument('--eval_n_runs', type=int, default=10)
    parser.add_argument('--eval_interval', type=int, default=10000)
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--start_epsilon', type=float, default=1.0)
    parser.add_argument('--end_epsilon', type=float, default=0.1)
    parser.add_argument('--decay_steps', type=int, default=1000000)
    parser.add_argument('--gamma', type=float, default=0.99)
    main(parser.parse_args())
