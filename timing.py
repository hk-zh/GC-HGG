# DEPRECATED, use baselines.common.plot_util instead

import os
import matplotlib.pyplot as plt
import numpy as np
import json
import seaborn as sns; sns.set()
import glob2
import argparse


def smooth_reward_curve(x, y):
    halfwidth = int(np.ceil(len(x) / 60))  # Halfwidth of our smoothing convolution
    k = halfwidth
    xsmoo = x
    ysmoo = np.convolve(y, np.ones(2 * k + 1), mode='same') / np.convolve(np.ones_like(y), np.ones(2 * k + 1),
        mode='same')
    return xsmoo, ysmoo


def load_results(file):
    if not os.path.exists(file):
        return None
    with open(file, 'r') as f:
        lines = [line for line in f]
    if len(lines) < 2:
        return None
    keys = [name.strip() for name in lines[0].split(',')]
    data = np.genfromtxt(file, delimiter=',', skip_header=1, filling_values=0.)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    assert data.ndim == 2
    assert data.shape[-1] == len(keys)
    result = {}
    for idx, key in enumerate(keys):
        result[key] = data[:, idx]
    return result


def pad(xs, value=np.nan):
    maxlen = np.max([len(x) for x in xs])

    padded_xs = []
    for x in xs:
        if x.shape[0] >= maxlen:
            padded_xs.append(x)

        padding = np.ones((maxlen - x.shape[0],) + x.shape[1:]) * value
        x_padded = np.concatenate([x, padding], axis=0)
        assert x_padded.shape[1:] == x.shape[1:]
        assert x_padded.shape[0] == maxlen
        padded_xs.append(x_padded)
    return np.array(padded_xs)



if __name__ == "__main__":
    # call plot.py to plot success stored in progress.csv files

    # get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str)
    parser.add_argument('env_id', type=str)
    parser.add_argument('--smooth', type=int, default=1)
    parser.add_argument('--naming', type=int, default=0)
    parser.add_argument('--e_per_c', type=int, default=50)
    parser.add_argument('--save_path', type=str, default=None)
    args = parser.parse_args()
    env_id = args.env_id

    # Load all data.
    data = {}
    paths = [os.path.abspath(os.path.join(path, '..')) for path in glob2.glob(os.path.join(args.dir, '**', 'progress.csv'))]
    location = 1
    for curr_path in paths:
        if not os.path.isdir(curr_path):
            continue
        if not args.dir in curr_path:
            continue
        clean_path = curr_path.replace(env_id, '')
        clean_path = os.path.basename(os.path.normpath(clean_path))
        clean_path = ''.join([i for i in clean_path if not i.isdigit()])
        # divide path into run (number in the beginning) and config (information on configuration, included in the path name)
        if args.naming == 0:
            config = clean_path
        elif args.naming == 1:
            if (("graph" in clean_path) or ("mesh" in clean_path)):
                config = "G-HGG"
            elif "hgg" in clean_path:
                config = "HGG"
            elif "normal" in clean_path:
                config = "HER"
            else:
                raise Exception("Naming failed!")


        # Test:
        run = config
        print('Config / run: {} / {}'.format(config, run))

        results = load_results(os.path.join(curr_path, 'progress.csv'))
        if not results:
            print('skipping {}'.format(curr_path))
            continue
        print('loading {} ({})'.format(curr_path, len(results['Episodes'])))

        # Filter out success rates from results
        for key, value in results.items():
            if 'TimeCost' in key:
                success_rate = np.array(results[key])
                for i in range(len(success_rate)):
                    if success_rate[i] > 1000:
                        success_rate[i] = success_rate[i-1]
                iteration = (np.array(results['Episodes'])) / args.e_per_c

                # Process and smooth data.
                assert success_rate.shape == iteration.shape
                x = iteration
                y = success_rate
                if args.smooth:
                    x, y = smooth_reward_curve(iteration, success_rate)
                assert x.shape == y.shape

                # store everything in an array
                if config not in data:
                    data[config] = {}
                if run not in data[config]:
                    data[config][run] = []
                data[config][run].append((x, y))


    # Plot data.
    print('exporting {}'.format(env_id))
    plt.clf()

    # new curve for each config
    for config in sorted(data.keys(), key=len):
        print("Config: {}".format(config))
        # merge curves from runs of one config
        for run in sorted(data[config].keys()):
            print("\tRun: {}".format(run))
            xs, ys = zip(*data[config][run])
            xs, ys = pad(xs), pad(ys)
            assert xs.shape == ys.shape
            plt.plot(xs[0], np.nanmedian(ys, axis=0), label=config)
            plt.fill_between(xs[0], np.nanpercentile(ys, 25, axis=0), np.nanpercentile(ys, 75, axis=0), alpha=0.25)

    plt.title(env_id)
    plt.xlabel('Iteration')
    plt.ylabel('Computation Time per Iteration (seconds)')
    plt.ylim(bottom=0)
    plt.legend(loc=location)
    plt.savefig(os.path.join(args.dir, 'timefig_{}.pdf'.format(env_id)), format='pdf')
    if args.save_path:
        plt.savefig(args.save_path)