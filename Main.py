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

def compare_masses(sorted_data_new, sorted_data_old):
    x = np.linspace(0, 1.4, 100)
    y = np.linspace(0, 1.4, 100)
    chandrasekhar_limit = 1.4

    m1_old = sorted_data_old[:, 8]
    m2_old = sorted_data_old[:, 9]
    k1_old = sorted_data_old[:, 10]
    k2_old = sorted_data_old[:, 11]

    m1_new = sorted_data_new[:, 8]
    m2_new = sorted_data_new[:, 9]
    k1_new = sorted_data_new[:, 10]
    k2_new = sorted_data_new[:, 11]

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

def compare_merger_time(data_new, data_old):
    merge_time_old = data_old[:, 5]
    merge_time_new = data_new[:, 5]

    fig, ax = plt.subplots(1, 2)
    ax[0].hist(merge_time_old, bins=20)
    ax[0].title.set_text("5% Merger Times")
    ax[0].set_xlabel("Merger Times (Myr)")
    ax[0].set_ylabel("Count")

    ax[1].hist(merge_time_new, bins=20)
    ax[1].title.set_text("50% Merger Times")
    ax[1].set_xlabel("Merger Times (Myr)")
    ax[1].set_ylabel("Count")

    plt.tight_layout()
    plt.show()

def compare_average_mass_with_time(data_new, sorted_data_new, data_old, sorted_data_old):
    merge_time_old = data_old[:, 5]
    m1_old = sorted_data_old[:, 8]
    m2_old = sorted_data_old[:, 9]
    mtotal_old = m1_old + m2_old
    m_ratio_old = m2_old / m1_old


    merge_time_new = data_new[:, 5]
    m1_new = sorted_data_new[:, 8]
    m2_new = sorted_data_new[:, 9]
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



def plot_m1_m2(data):
    m1_all = data[:, 8]
    m2_all = data[:, 9]

    # Define mass bins and shared color logic
    bins = [0.5, 1.1, 1.4]
    cat1_all = np.digitize(m1_all, bins)
    cat2_all = np.digitize(m2_all, bins)
    color_indices_all = cat1_all * 4 + cat2_all

    # Setup for legends
    cmap = plt.get_cmap('tab20')
    norm = mcolors.Normalize(vmin=0, vmax=15)
    range_labels = ["<0.5", "0.5-1.1", "1.1-1.4", ">1.4"]

    def create_pruned_legend(ax, present_indices):
        handles = []
        labels = []
        # Sort indices to keep the legend order consistent
        for idx in sorted(present_indices):
            i, j = divmod(idx, 4)  # i is M1 cat, j is M2 cat
            color = cmap(norm(idx))
            handle = mlines.Line2D([], [], color=color, marker='o',
                                   linestyle='None', markersize=8)
            handles.append(handle)
            labels.append(f"M1:{range_labels[i]}, M2:{range_labels[j]}")

        return ax.legend(handles=handles, labels=labels, title="Mass Combinations ($M_\\odot$)",
                         loc='upper left', fontsize='small', bbox_to_anchor=(1, 1))

    # PLOT 1: M1 < 1.4 Msun
    mask1 = m1_all < 1.4
    m1_1 = m1_all[mask1]
    m2_1 = m2_all[mask1]
    idx_1 = color_indices_all[mask1]

    fig1, ax1 = plt.subplots(figsize=(11, 7))
    ax1.scatter(m1_1, m2_1, c=idx_1, cmap='tab20', vmin=0, vmax=15, alpha=0.8)

    # Add reference lines for Plot 1
    x_lin = np.linspace(0, 1.4, 100)
    ax1.plot(x_lin, x_lin, color="black", linewidth=1, label="Boundary")
    ax1.plot(x_lin[x_lin >= 0.7], 1.4 - x_lin[x_lin >= 0.7], color="black", linestyle='--', label="Chandra Limit")

    ax1.set_title("Systems with $M_1 < 1.4 M_\\odot$")
    ax1.set_xlabel("M1 (Msun)")
    ax1.set_ylabel("M2 (Msun)")

    create_pruned_legend(ax1, np.unique(idx_1))
    plt.tight_layout()

    # PLOT 2: M1 > 1.4 Msun
    mask2 = m1_all >= 1.4
    m1_2 = m1_all[mask2]
    m2_2 = m2_all[mask2]
    idx_2 = color_indices_all[mask2]

    fig2, ax2 = plt.subplots(figsize=(11, 7))
    ax2.scatter(m1_2, m2_2, c=idx_2, cmap='tab20', vmin=0, vmax=15, alpha=0.8)

    # Add reference lines for Plot 2 (Boundary only as it goes above 1.4)
    x_high = np.linspace(1.4, max(m1_all), 100)
    ax2.plot(x_high, x_high, color="black", linewidth=1)

    ax2.set_title("Systems with $M_1 > 1.4 M_\\odot$")
    ax2.set_xlabel("M1 (Msun)")
    ax2.set_ylabel("M2 (Msun)")

    create_pruned_legend(ax2, np.unique(idx_2))
    plt.tight_layout()

    plt.show()

def plot_log_log_mass(data):
    m1 = data[:, 8]
    m2 = data[:, 9]

    # Define mass bins and categories
    # <0.5 (0), 0.5-1.1 (1), 1.1-1.4 (2), >1.4 (3)
    bins = [0.5, 1.1, 1.4]
    cat1 = np.digitize(m1, bins)
    cat2 = np.digitize(m2, bins)
    color_indices = cat1 * 4 + cat2

    # Setup Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter plot with log-log scale
    scatter = ax.scatter(m1, m2, c=color_indices, cmap='tab20',
                         vmin=0, vmax=15, marker='o', alpha=0.7, s=20)

    ax.set_xscale('log')
    ax.set_yscale('log')

    # Add Reference Lines
    # Boundary line (y = x)
    x_range = np.logspace(np.log10(min(m1)), np.log10(max(m1)), 100)
    ax.plot(x_range, x_range, color="black", linewidth=1, label="Boundary ($M_1=M_2$)")

    # Chandrasekhar Limit line (M1 + M2 = 1.4)
    # This is only relevant for masses < 1.4. In log-log, this is a curve.
    m1_chandra = np.linspace(0.01, 1.39, 200)
    ax.plot(m1_chandra, 1.4 - m1_chandra, color="black", linestyle='--', label="Chandra Limit")

    cmap = plt.get_cmap('tab20')
    norm = mcolors.Normalize(vmin=0, vmax=15)
    range_labels = ["<0.5", "0.5-1.1", "1.1-1.4", ">1.4"]

    present_indices = np.unique(color_indices)
    color_handles = []
    legend_labels = []

    for idx in sorted(present_indices):
        i, j = divmod(idx, 4)  # Get row (M1) and col (M2) indices
        color = cmap(norm(idx))
        handle = mlines.Line2D([], [], color=color, marker='o', linestyle='None', markersize=8)
        color_handles.append(handle)
        legend_labels.append(f"M1:{range_labels[i]}, M2:{range_labels[j]}")

    # Add mass combination legend
    color_legend = ax.legend(handles=color_handles, labels=legend_labels,
                             title="Mass Combinations ($M_\\odot$)",
                             bbox_to_anchor=(0, 1), loc='upper left', fontsize='small')
    ax.add_artist(color_legend)

    # Formatting
    ax.set_xlabel("Log $M_1$ ($M_\\odot$)")
    ax.set_ylabel("Log $M_2$ ($M_\\odot$)")
    ax.set_title("Binary Masses (Log-Log Scale)")
    ax.grid(True, which="both", ls="-", alpha=0.2)

    plt.tight_layout()
    plt.show()

def mergers_vs_time(data):
    merge_time = data[:, 5]

    fig, ax = plt.subplots()
    ax.hist(merge_time, bins=20)
    ax.title.set_text("Number of Mergers Over Time")
    ax.set_xlabel("Merger Times (Myr)")
    ax.set_ylabel("Count")

    plt.tight_layout()
    plt.show()

def z_overtime(data):
    merge_Z = data[:, 3]
    merge_time = data[:, 5]

    plt.figure(figsize=(10, 7))

    # Create a 2D histogram with 20 bins on each axis
    # x is time, y is Z
    plt.hist2d(merge_time, merge_Z, bins=20, cmap='viridis')

    # Add a colorbar to show what the colors represent (count of mergers)
    cb = plt.colorbar()
    cb.set_label('Number of Mergers')

    plt.xlabel('Merger Time (Gyr)')
    plt.ylabel('Metallicity (Z)')
    plt.title('Metallicity Distribution Over Time')

    plt.show()



# Collisions of at least 1 compact object (WD, NS, or BH)
CO_collisions = np.loadtxt("data/CO_collisions_binaryfraction.dat")
# Collisions of at least 1 WD
WD_mergers = np.loadtxt("data/WD_mergers_binaryfraction.dat")

sorted_CO = sort_binary_mass(CO_collisions)
sorted_WD = sort_binary_mass(WD_mergers)

columns = ["modelID", "rv(pc)", "Rgc(kpc)", "Z", "N(x10^5)", "t_merger(Myr)", "ID1", "ID2", "M1(Msun)", "M2(Msun)", "k1", "k2", "r(pc)"]

plot_m1_m2(sorted_WD)
mergers_vs_time(sorted_WD)
z_overtime(sorted_WD)

plot_m1_m2(sorted_CO)
plot_log_log_mass(sorted_CO)
mergers_vs_time(sorted_CO)
z_overtime(sorted_CO)

#For each rv value, plot mergers vs time
#Make M1 M2 plots for each variant of k plots
#Number of mergers over rv and N (as well as late time scales)

