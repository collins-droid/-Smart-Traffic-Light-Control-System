import matplotlib.pyplot as plt
import numpy as np
import glob
import seaborn as sns


def get_file_names():
    qmodel_file_name = glob.glob('./qmodel*')
    stats_file_name = glob.glob('./stats*')

    if not qmodel_file_name:
        qmodel_file_name = ''
    else:
        qmodel_file_name = qmodel_file_name[0]

    if not stats_file_name:
        stats_file_name = ''
    else:
        stats_file_name = stats_file_name[0]

    return qmodel_file_name, stats_file_name


def get_init_epoch(filename, total_episodes):
    if filename:
        index = filename.find('_')
        exp_start = index + 1
        exp_end = int(filename.find('_', exp_start))
        exp = int(filename[exp_start:exp_end])
        epoch_start = exp_end + 1
        epoch_end = int(filename.find('.', epoch_start))
        epoch = int(filename[epoch_start:epoch_end])
        if epoch < total_episodes - 1:
            epoch += 1
        else:
            epoch = 0
            exp += 1

    else:
        exp = 0
        epoch = 0
    return exp, epoch


def get_stats(stats_filename, num_experiments, total_episodes, learn=True):
    if stats_filename and learn:
        stats = np.load(stats_filename, allow_pickle=True)[()]

    else:
        reward_store = np.zeros((num_experiments, total_episodes))
        intersection_queue_store = np.zeros((num_experiments, total_episodes))
        stats = {'rewards': reward_store, 'intersection_queue': intersection_queue_store}

    return stats


def plot_sample(sample, title, xlabel, legend_label, show=True, subplot=False):
    if not subplot:
        plt.figure()

    ax = sns.distplot(sample, kde=True, label=legend_label)
    ax.set(xlabel=xlabel, title=title)
    ax.legend()

    if show:
        plt.show()



def plot_rewards(reward_store, label):
    x = np.mean(reward_store, axis=0)
    plt.plot(x, label=label)
    plt.xlabel('Episodes')
    plt.ylabel('Cummulative negative wait times')
    plt.title('Cummulative negative wait times across episodes')
    plt.legend()


def plot_intersection_queue_size(intersection_queue_store, label):
    x = np.mean(intersection_queue_store, axis=0)
    plt.plot(x, label=label, color='m')
    plt.xlabel('Episodes')
    plt.ylabel('Cummulative intersection queue size')
    plt.title('Cummulative intersection queue size across episodes')
    plt.legend()


def show_plots():
    plt.show()
