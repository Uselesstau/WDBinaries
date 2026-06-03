import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
import itertools

def sort_binary_mass(data):
    m1 = data[:,8]
    m2 = data[:,9]
    id1 = data[:,6]
    id2 = data[:,7]
    k1 = data[:,10]
    k2 = data[:,11]
    for i in range(0, len(m1)):
        if m1[i] < m2[i]:
            m1[i], m2[i] = m2[i], m1[i]
            id1[i], id2[i] = id2[i], id1[i]
            k1[i], k2[i] = k2[i], k1[i]
    return data

dataOld = np.loadtxt("WD_mergers_old.dat")
dataNew = np.loadtxt("WD_mergers_new.dat")

sorted_old_data = sort_binary_mass(dataOld)
sorted_new_data = sort_binary_mass(dataNew)

columns = ["modelID", "rv(pc)", "Rgc(kpc)", "Z", "N(x10^5)", "t_merger(Myr)", "ID1", "ID2", "M1(Msun)", "M2(Msun)", "k1", "k2", "r(pc)"]

def plot_masses():
    x = np.linspace(0, 1.4, 100)
    y = np.linspace(0, 1.4, 100)
    chandrasekhar_limit = 1.4

    m1_old = sorted_old_data[:, 8]
    m2_old = sorted_old_data[:, 9]
    k1_old = sorted_old_data[:, 10]
    k2_old = sorted_old_data[:, 11]

    m1_new = sorted_new_data[:, 8]
    m2_new = sorted_new_data[:, 9]
    k1_new = sorted_new_data[:, 10]
    k2_new = sorted_new_data[:, 11]

    color_indices_old = (k1_old - 10) * 3 + (k2_old - 10)
    color_indices_new = (k1_new - 10) * 3 + (k2_new - 10)

    # Plot the data
    fig, ax = plt.subplots()

    ax.scatter(m1_old, m2_old, c=color_indices_old, cmap='tab10', vmin=0, vmax=8, marker='o', alpha=0.8)
    ax.scatter(m1_new, m2_new, c=color_indices_new, cmap='tab10', vmin=0, vmax=8, marker='x', alpha=0.8)

    ax.plot(x[x >= 0.69], chandrasekhar_limit - x[x >=0.69], label="Chandrasekhar Limit", color="black")
    ax.plot(x, y, label="Boundry", color="black")
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_xlabel("M1 (Msun)")
    ax.set_ylabel("M2 (Msun)")

    # Create the Color Legend manually to ensure all 9 show up
    cmap = plt.get_cmap('tab10')
    norm = mcolors.Normalize(vmin=0, vmax=8)

    color_handles = []
    legend_labels = []

    # Generate all 9 combinations in the exact same order as our math equation
    combinations = list(itertools.product([10, 11, 12], [10, 11, 12]))

    for idx, (k1_val, k2_val) in enumerate(combinations):
        color = cmap(norm(idx))

        handle = mlines.Line2D([], [], color=color, marker='o', linestyle='None', markersize=8)

        color_handles.append(handle)
        legend_labels.append(f"k1={k1_val}, k2={k2_val}")

    # Add the color legend
    color_legend = ax.legend(handles=color_handles, labels=legend_labels, title="Combinations",
                             bbox_to_anchor=(0, 1), loc='upper left')
    ax.add_artist(color_legend)

    old_marker = mlines.Line2D([], [], color='black', marker='o', linestyle='None', label='5% Binary Fraction')
    new_marker = mlines.Line2D([], [], color='black', marker='x', linestyle='None', label='50% Binary Fraction')
    line_sum = mlines.Line2D([], [], color='black', label='Chandrasekhar Limit')

    ax.legend(handles=[old_marker, new_marker, line_sum], loc='lower right')

    plt.tight_layout()

    plt.show()

def plot_merger_time():
    merge_time_old = dataOld[:, 5]
    merge_time_new = dataNew[:, 5]

    fig, ax = plt.subplots(1, 2)
    ax[0].hist(merge_time_old, bins=100)
    ax[0].title.set_text("5% Merger Times")
    ax[0].set_xlabel("Merger Times (Myr)")
    ax[0].set_ylabel("Count")

    ax[1].hist(merge_time_new, bins=100)
    ax[1].title.set_text("50% Merger Times")
    ax[1].set_xlabel("Merger Times (Myr)")
    ax[1].set_ylabel("Count")

    plt.tight_layout()
    plt.show()

def plot_average_mass_with_time():
    merge_time_old = dataOld[:, 5]
    m1_old = sorted_old_data[:, 8]
    m2_old = sorted_old_data[:, 9]
    mtotal_old = m1_old + m2_old
    m_ratio_old = m2_old / m1_old


    merge_time_new = dataNew[:, 5]
    m1_new = sorted_new_data[:, 8]
    m2_new = sorted_new_data[:, 9]
    mtotal_new = m1_new + m2_new
    m_ratio_new = m2_new / m1_new

    fig, ax = plt.subplots(2, 2)

    ax[0, 0].scatter(merge_time_old, m_ratio_old, c="red")
    ax[0, 0].scatter(merge_time_new, m_ratio_new, c="blue")
    ax[0, 0].set_xlabel("Merger Times (Myr)")
    ax[0, 0].set_ylabel("Average Merger Mass (Msun)")

    ax[0, 1].scatter(merge_time_old, mtotal_old, c="red")
    ax[0, 1].scatter(merge_time_new, mtotal_new, c="blue")
    ax[0, 1].set_xlabel("Merger Times (Myr)")
    ax[0, 1].set_ylabel("Total Merger Mass (Msun)")

    bins = np.linspace(0, 14000, 20)
    bin_indices_old = np.digitize(merge_time_old, bins)
    bin_means_old = [merge_time_old[bin_indices_old == i].mean() for i in bins]
    bin_indices_new = np.digitize(merge_time_new, bins)
    bin_means_new = [merge_time_new[bin_indices_new == i].mean() for i in bins]

    ax[1, 0].hist(bin_means_old, m_ratio_old, c="red")
    ax[1, 0].hist(bin_means_new, m_ratio_new, c="blue")
    ax[1, 0].set_xlabel("Merger Times (Myr)")
    ax[1, 0].set_ylabel("Average Merger Mass (Msun)")

    ax[1, 1].hist(bin_means_old, mtotal_old, c="red")
    ax[1, 1].hist(bin_means_new, mtotal_new, c="blue")
    ax[1, 1].set_xlabel("Merger Times (Myr)")
    ax[1, 1].set_ylabel("Total Merger Mass (Msun)")

    plt.tight_layout()
    plt.show()

#plot_masses()
#plot_merger_time()
plot_average_mass_with_time()

