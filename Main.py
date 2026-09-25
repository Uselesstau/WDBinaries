import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
from astropy.cosmology import Planck18
import astropy.units as u
import ClusterWeightCalculator as cwc
from ClusterWeightCalculator import inspect_model_weights


def sort_binary_mass(data):
    """Rearranges each data entry such that mass 1 > mass 2

    Arguments:
    data - the raw data of Mergers or Collisions

    Returns:
    Rearranged data
    """
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

def _normalize_k_values(k_array):
    # Return k values as-is to get unique color for each k value
    return k_array.astype(int)

def _k_label(k_normalized):
    # Display the actual k value
    return f"k={int(k_normalized)}"

def plot_m1_m2_combined(data, title_context=""):
    """Plots m1 vs m2 for a given data set, coloring based off of mass regions and k-value combinations

    Arguments:
    data - the raw data of Mergers or Collisions
    title_context - the title of the plot
    """
    # Filter out masses less than 0.1
    mask = (data[:, 8] >= 0.1) & (data[:, 9] >= 0.1)
    data = data[mask]
    
    m1_all = data[:, 8]
    m2_all = data[:, 9]
    k1_all = data[:, 10]
    k2_all = data[:, 11]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # LEFT PLOT: Mass Combinations
    bins = [0.5, 1.1, 1.4]
    cat1_all = np.digitize(m1_all, bins)
    cat2_all = np.digitize(m2_all, bins)
    color_indices_mass = cat1_all * 4 + cat2_all

    cmap_mass = plt.get_cmap('tab20')
    norm_mass = mcolors.Normalize(vmin=0, vmax=15)
    range_labels = ["<0.5", "0.5-1.1", "1.1-1.4", ">1.4"]

    ax1.scatter(m1_all, m2_all, c=color_indices_mass, cmap='tab20', vmin=0, vmax=15, alpha=0.8)
    
    x_range = np.linspace(0, max(m1_all), 100)
    ax1.plot(x_range, x_range, color="black", linewidth=1, label="Boundary")
    ax1.plot(x_range[x_range >= 0.7], 1.4 - x_range[x_range >= 0.7], color="black", linestyle='--', label="Chandra Limit")
    ax1.axvline(x=1.4, color="red", linewidth=2, linestyle=':', label="M1 = 1.4 Msun")
    
    ax1.set_xlabel("M1 (Msun)")
    ax1.set_ylabel("M2 (Msun)")
    ax1.set_title(f"Binary Masses (M1 vs M2){f' - {title_context}' if title_context else ''}")
    ax1.grid(alpha=0.3)

    # Create legend for mass plot
    present_indices_mass = np.unique(color_indices_mass)
    handles_mass = []
    labels_mass = []
    for idx in sorted(present_indices_mass):
        i, j = divmod(idx, 4)
        color = cmap_mass(norm_mass(idx))
        handle = mlines.Line2D([], [], color=color, marker='o', linestyle='None', markersize=8)
        handles_mass.append(handle)
        labels_mass.append(f"M1:{range_labels[i]}, M2:{range_labels[j]}")
    
    ax1.legend(handles=handles_mass, labels=labels_mass, title="Mass Combinations ($M_\\odot$)", loc='upper left', fontsize='small')

    # RIGHT PLOT: k-type Combinations
    k1_normalized = _normalize_k_values(k1_all)
    k2_normalized = _normalize_k_values(k2_all)
    
    unique_combinations = sorted(set(zip(k1_normalized, k2_normalized)))
    k_combo_to_idx = {combo: idx for idx, combo in enumerate(unique_combinations)}
    color_indices_k = np.array([k_combo_to_idx[(k1, k2)] for k1, k2 in zip(k1_normalized, k2_normalized)])

    n_combos = len(unique_combinations)
    cmap_k = plt.get_cmap('tab20')
    norm_k = mcolors.Normalize(vmin=0, vmax=n_combos - 1)

    ax2.scatter(m1_all, m2_all, c=color_indices_k, cmap=cmap_k, vmin=0, vmax=n_combos - 1, alpha=0.8)
    
    ax2.plot(x_range, x_range, color="black", linewidth=1, label="Boundary")
    ax2.plot(x_range[x_range >= 0.7], 1.4 - x_range[x_range >= 0.7], color="black", linestyle='--', label="Chandra Limit")
    ax2.axvline(x=1.4, color="red", linewidth=2, linestyle=':', label="M1 = 1.4 Msun")
    
    ax2.set_xlabel("M1 (Msun)")
    ax2.set_ylabel("M2 (Msun)")
    ax2.set_title(f"Binary Masses by k Types{f' - {title_context}' if title_context else ''}")
    ax2.grid(alpha=0.3)

    # Create legend for k-type plot
    present_indices_k = np.unique(color_indices_k)
    handles_k = []
    labels_k = []
    for idx in sorted(present_indices_k):
        idx_int = int(idx)
        color = cmap_k(norm_k(idx_int))
        k1_val, k2_val = unique_combinations[idx_int]
        handle = mlines.Line2D([], [], color=color, marker='o', linestyle='None', markersize=8)
        handles_k.append(handle)
        labels_k.append(f"k1={_k_label(k1_val)}, k2={_k_label(k2_val)}")
    
    ax2.legend(handles=handles_k, labels=labels_k, title="k1/k2 Combinations", loc='upper left', fontsize='small')

    plt.tight_layout()
    plt.show()

def plot_log_log_mass_combined(data, title_context=""):
    """Plots m1 vs m2 in log-log scale for a given data set, coloring based off of mass regions and k-value combinations

    Arguments:
    data - the raw data of Mergers or Collisions
    title_context - the title of the plot
    """
    # Filter out masses less than 0.1
    mask = (data[:, 8] >= 0.1) & (data[:, 9] >= 0.1)
    data = data[mask]
    
    m1 = data[:, 8]
    m2 = data[:, 9]
    k1 = data[:, 10]
    k2 = data[:, 11]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # LEFT PLOT: Mass Combinations (Log-Log)
    bins = [0.5, 1.1, 1.4]
    cat1 = np.digitize(m1, bins)
    cat2 = np.digitize(m2, bins)
    color_indices_mass = cat1 * 4 + cat2

    ax1.scatter(m1, m2, c=color_indices_mass, cmap='tab20', vmin=0, vmax=15, marker='o', alpha=0.7, s=20)
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')

    x_range = np.logspace(np.log10(np.min(m1)), np.log10(np.max(m1)), 100)
    ax1.plot(x_range, x_range, color="black", linewidth=1, label="Boundary ($M_1=M_2$)")
    
    m1_chandra = np.linspace(0.01, 1.39, 200)
    ax1.plot(m1_chandra, 1.4 - m1_chandra, color="black", linestyle='--', label="Chandra Limit")
    
    ax1.set_xlabel("Log $M_1$ ($M_\\odot$)")
    ax1.set_ylabel("Log $M_2$ ($M_\\odot$)")
    ax1.set_title(f"Binary Masses (Log-Log Scale){f' - {title_context}' if title_context else ''}")
    ax1.grid(True, which="both", ls="-", alpha=0.2)

    # Create legend for mass plot
    cmap_mass = plt.get_cmap('tab20')
    norm_mass = mcolors.Normalize(vmin=0, vmax=15)
    range_labels = ["<0.5", "0.5-1.1", "1.1-1.4", ">1.4"]
    
    present_indices_mass = np.unique(color_indices_mass)
    handles_mass = []
    labels_mass = []
    for idx in sorted(present_indices_mass):
        i, j = divmod(idx, 4)
        color = cmap_mass(norm_mass(idx))
        handle = mlines.Line2D([], [], color=color, marker='o', linestyle='None', markersize=8)
        handles_mass.append(handle)
        labels_mass.append(f"M1:{range_labels[i]}, M2:{range_labels[j]}")
    
    ax1.legend(handles=handles_mass, labels=labels_mass, title="Mass Combinations ($M_\\odot$)", loc='upper left', fontsize='small')

    # RIGHT PLOT: k-type Combinations (Log-Log)
    k1_normalized = _normalize_k_values(k1)
    k2_normalized = _normalize_k_values(k2)
    
    unique_combinations = sorted(set(zip(k1_normalized, k2_normalized)))
    k_combo_to_idx = {combo: idx for idx, combo in enumerate(unique_combinations)}
    color_indices_k = np.array([k_combo_to_idx[(k1_val, k2_val)] for k1_val, k2_val in zip(k1_normalized, k2_normalized)])

    n_combos = len(unique_combinations)
    cmap_k = plt.get_cmap('tab20')
    norm_k = mcolors.Normalize(vmin=0, vmax=n_combos - 1)

    ax2.scatter(m1, m2, c=color_indices_k, cmap=cmap_k, vmin=0, vmax=n_combos - 1, marker='o', alpha=0.7, s=20)
    
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    
    ax2.plot(x_range, x_range, color="black", linewidth=1, label="Boundary ($M_1=M_2$)")
    ax2.plot(m1_chandra, 1.4 - m1_chandra, color="black", linestyle='--', label="Chandra Limit")
    
    ax2.set_xlabel("Log $M_1$ ($M_\\odot$)")
    ax2.set_ylabel("Log $M_2$ ($M_\\odot$)")
    ax2.set_title(f"Binary Masses by k Types (Log-Log Scale){f' - {title_context}' if title_context else ''}")
    ax2.grid(True, which="both", ls="-", alpha=0.2)

    # Create legend for k-type plot
    present_indices_k = np.unique(color_indices_k)
    handles_k = []
    labels_k = []
    for idx in sorted(present_indices_k):
        idx_int = int(idx)
        color = cmap_k(norm_k(idx_int))
        k1_val, k2_val = unique_combinations[idx_int]
        handle = mlines.Line2D([], [], color=color, marker='o', linestyle='None', markersize=8)
        handles_k.append(handle)
        labels_k.append(f"k1={_k_label(k1_val)}, k2={_k_label(k2_val)}")
    
    ax2.legend(handles=handles_k, labels=labels_k, title="k1/k2 Combinations", loc='upper left', fontsize='small')

    plt.tight_layout()
    plt.show()

def mergers_vs_time(data, title_context=""):
    """Plots the evolution of the number of mergers over the time (bins=20)

    Arguments:
    data - the raw data of Mergers or Collisions
    title_context - the title of the plot
    """
    merge_time = data[:, 5]

    fig, ax = plt.subplots()
    ax.hist(merge_time, bins=20)
    
    title = f"Number of Mergers Over Time{f' - {title_context}' if title_context else ''}"
    ax.title.set_text(title)
    ax.set_xlabel("Merger Times (Myr)")
    ax.set_ylabel("Count")

    plt.tight_layout()
    plt.show()

def filter_k_values(data, k_min=10, k_max=12):
    """Removes k values smaller than k_min (k<10 = not a WD)

    Arguments:
    data - the raw data of Mergers or Collisions
    k_min - the minimum k value

    Returns:
    Masked data with k values smaller than k_min (k<10 = not a WD)
    """
    k1 = data[:, 10]
    k2 = data[:, 11]
    mask = (k1 >= k_min) & (k2 >= k_min) & (k1 < k_max) & (k2 < k_max)
    return data[mask]

def separate_by_column(data, col_idx):
    """Separates the data into a number of datasets equal to the number of unique values in a given column

    Arguments:
    data - the raw data of Mergers or Collisions
    col_idx - the index of the column to separate by (e.g. 1 for rv, 3 for Z, 4 for N)

    Returns:
    grouped_datasets - dictionary of the datasets split by unique column values
    unique_values - the unique values in the column
    """
    column_data = data[:, col_idx]
    unique_values = np.unique(column_data)
    grouped_datasets = {}

    for val in unique_values:
        mask = column_data == val
        grouped_datasets[val] = data[mask]

    return grouped_datasets, unique_values

def filter_extreme_models(data, model_col_idx=0, percentile=95):
    """Checks the data set and removes any models with an extreme number (default=95% tile)

    Arguments:
    data - the raw data of Mergers or Collisions
    model_col_idx - the index of the model to check
    percentile - the percentile to check

    Returns:
    filtered_data - the filtered data
    """
    if len(data) == 0:
        return data

    model_ids = data[:, model_col_idx]

    # Get unique models and count the number of entries (rows) for each
    unique_models, counts = np.unique(model_ids, return_counts=True)

    # Determine and filter thresholds
    threshold = np.percentile(counts, percentile)
    valid_models = unique_models[counts <= threshold]

    mask = np.isin(model_ids, valid_models)
    filtered_data = data[mask]

    # Calculate stats for printing
    removed_models_count = len(unique_models) - len(valid_models)
    removed_entries_count = len(data) - len(filtered_data)

    if removed_models_count > 0:
        print(f"    - Threshold: {threshold:.0f} entries per model")
        print(f"    - Removed {removed_entries_count} entries from {removed_models_count} extreme model(s)")
    else:
        print("    - No extreme models found.")

    return filtered_data

def remove_specific_model_group(data, target_rv=0.5, target_z=0.03, target_n=4.0):
    """Removes the models with specific values due to their extreme results

    Arguments:
    data - the raw data of Mergers or Collisions
    target_rv - the model(s) with this rv
    target_z - the model(s) with this z
    target_n - the model(s) with this n

    Returns:
    filtered_data - the filtered data
    """
    if len(data) == 0:
        return data

    # 1. Create conditions for each parameter
    mask_rv = np.isclose(data[:, 1], target_rv)
    mask_z = np.isclose(data[:, 3], target_z)
    mask_n = np.isclose(data[:, 4], target_n)

    # 2. Combine conditions to find the exact rows to remove
    # (True where ALL three parameters match)
    mask_to_remove = mask_rv & mask_z & mask_n

    # 3. Invert the mask to KEEP everything else
    # The ~ operator flips True to False and False to True
    mask_to_keep = ~mask_to_remove

    filtered_data = data[mask_to_keep]

    # Calculate and print stats
    removed_count = len(data) - len(filtered_data)
    print(f"    - Excluded {removed_count} mergers matching rv={target_rv}, Z={target_z}, N={target_n}")

    return filtered_data

def overlay_mergers_by_group(datasets, group_name="RV", data_type="WD"):
    """Plots the mergers over time, overlaying each group value (rv, Z, N)

    Arguments:
    datasets - dictionary of grouped datasets
    group_name - the string name of the column grouped by
    data_type - the type of merger to plot
    """
    # Create overlay plot of merger/collision times for each group value
    fig, ax = plt.subplots(figsize=(12, 7))

    # Define colors for each group value
    colors = plt.cm.tab10(np.linspace(0, 1, len(datasets)))

    # Calculate global min and max across all datasets for equal bins
    all_merge_times = []
    for group_data in datasets.values():
        all_merge_times.extend(group_data[:, 5])

    if len(all_merge_times) == 0:
        return

    global_min = np.min(all_merge_times)
    global_max = np.max(all_merge_times)

    # Create bins that span the entire range equally
    bins = np.linspace(global_min, global_max, 21)  # 20 bins (21 edges)

    # Calculate the center of each bin for plotting the lines
    bin_centers = (bins[:-1] + bins[1:]) / 2

    for (val, group_data), color in zip(sorted(datasets.items()), colors):
        merge_times = group_data[:, 5]  # Column 5 is t_merger(Myr)

        # Calculate histogram counts manually
        counts, _ = np.histogram(merge_times, bins=bins)

        # Plot the counts at the center of each bin
        ax.plot(bin_centers, counts, marker='o', linestyle='-',
                label=f"{group_name} = {val}", color=color, linewidth=2,
                markersize=6, alpha=0.8)

    ax.set_xlabel("Merger Times (Myr)")
    ax.set_ylabel("Count")
    ax.set_title(f"Overlaid Merger/Collision Times by {group_name} - {data_type}")
    ax.legend()
    ax.grid(axis='both', alpha=0.3)

    plt.tight_layout()
    plt.show()

def mergers_over_time_loglog_by_group(datasets, group_name="RV", data_type="WD", start_time=150.0):
    """Plots mergers over time in log-log scale, overlaying each group value

    Arguments:
    datasets - dictionary of grouped datasets
    group_name - the string name of the column grouped by
    data_type - the type of merger to plot
    start_time - the start time to plot (in Myr)
    """
    # Plot merger/collision counts over time in log-log scale (non-cumulative)
    fig, ax = plt.subplots(figsize=(12, 7))

    # Define colors for each group value
    colors = plt.cm.tab10(np.linspace(0, 1, len(datasets)))

    # Calculate global min and max for equal bin edges across all datasets
    all_merge_times = []
    for group_data in datasets.values():
        valid = group_data[:, 5][group_data[:, 5] >= start_time]
        all_merge_times.extend(valid)

    if len(all_merge_times) > 0:
        global_min = np.max([np.min(all_merge_times), start_time])
        global_max = np.max(all_merge_times)

        # Create equal-sized bins for all datasets
        num_bins = 20
        bins = np.logspace(np.log10(global_min), np.log10(global_max), num_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        for (val, group_data), color in zip(sorted(datasets.items()), colors):
            merge_times = group_data[:, 5]  # Column 5 is t_merger(Myr)

            # Filter times >= start_time
            valid_times = merge_times[merge_times >= start_time]

            if len(valid_times) > 0:
                # Create histogram (counts per bin, not cumulative)
                counts, _ = np.histogram(valid_times, bins=bins)

                # Remove zero counts for log scale plotting
                nonzero_mask = counts > 0
                nonzero_times = bin_centers[nonzero_mask]
                nonzero_counts = counts[nonzero_mask]

                if len(nonzero_counts) > 0:
                    ax.loglog(nonzero_times, nonzero_counts, marker='o', linestyle='-', label=f"{group_name} = {val}",
                              color=color, linewidth=2, markersize=6, alpha=0.7)

    ax.set_xlabel("Time (Myr)")
    ax.set_ylabel("Merger/Collision Count per Bin")
    ax.set_title(f"Merger/Collision Distribution Over Time (Log-Log) by {group_name} - {data_type}")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.show()

def generate_mass_groups(data):
    """Separates the data into 3 mass groups (All M_total, M_total > 1.4, M_total <= 1.4)

    Arguments:
    data - the raw data of Mergers or Collisions

    Returns:
    mass_datasets - dictionary of the datasets split by total mass values
    """
    m_total_data = data[:, 8] + data[:, 9]
    # Create the dictionary of datasets
    mass_datasets = {
        "All": data,
        "M < 1.4": data[m_total_data < 1.4],
        "M >= 1.4": data[m_total_data >= 1.4]
    }

    return mass_datasets

def overlay_mass_fractions(datasets, model_name="", bin_size=1000):
    """Find the fraction of masses in each mass group that has merged at a given age (All M_total, M_total > 1.4, M_total <= 1.4)

    Arguments:
    datasets - the datasets of Mergers or Collisions separated by total mass values
    model_name - the string name of the model
    bin_size - the size of each bin in Myr
    """
    # Create overlay plot of merger/collision times for each group value
    fig, ax = plt.subplots(figsize=(12, 7))

    # Define colors for each group value
    colors = plt.cm.tab10(np.linspace(0, 1, len(datasets)))

    # Calculate global max across all datasets to figure out how far our bins must go
    all_merge_times = []
    for group_data in datasets.values():
        if len(group_data) > 0:
            all_merge_times.extend(group_data[:, 5])

    global_max = np.max(all_merge_times)

    # Create a bin every Gyr
    bins = np.arange(0, global_max + bin_size, bin_size)

    # Calculate the center of each bin for plotting the lines
    bin_edges_right = bins[1:]

    for (val, group_data), color in zip(sorted(datasets.items()), colors):
        merge_times = group_data[:, 5]  # Column 5 is t_merger(Myr)
        total_count = len(merge_times)

        if total_count == 0:
            continue

        # Calculate the raw histogram counts per 1000 Myr bin
        counts, _ = np.histogram(merge_times, bins=bins)

        # Plot the percentages at the center of each bin
        cumulative_counts = np.cumsum(counts)
        percentages = (cumulative_counts / total_count) * 100

        ax.plot(bin_edges_right, percentages, marker='o', linestyle='-',
                label=f"{val} (n={total_count})", color=color, linewidth=2,
                markersize=6, alpha=0.8)

    ax.set_xlabel(f"Merger Times (Myr) - {int(bin_size)} Myr Bins")
    ax.set_ylabel(f"Cumulative Percentage (%)")
    ax.set_title(f"Percentage of Mergers/Collisions Over Time by {model_name}")
    ax.legend()
    ax.grid(axis='both', alpha=0.3)

    ax.set_ylim(0, 105)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1000))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(500))

    plt.tight_layout()
    plt.show()

def plot_merger_rate_density(datasets, group_name="", z_max=5.0, num_bins=50, cluster_density_mpc3=1.0):
    """
    Plots the merger rate density R(z) in Gpc^-3 yr^-1 vs redshift z.
    Includes a secondary top axis showing the Age of the Universe in Gyr.

    Arguments:
    datasets - dictionary of grouped datasets (e.g., separated by mass).
    group_name - the string name of the column grouped by
    z_max - maximum redshift to plot (default 10.0).
    num_bins - number of redshift bins.
    cluster_density_mpc3 - Assumed density of clusters in the universe in Mpc^-3.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define colors for each group value
    colors = plt.cm.tab10(np.linspace(0, 1, len(datasets)))

    # Find the total number of unique models simulated across all groups
    all_model_ids = set()
    for data in datasets.values():
        if len(data) > 0:
            all_model_ids.update(data[:, 0])
    unique_models = list(all_model_ids)
    total_models = len(all_model_ids) if len(all_model_ids) > 0 else 1

    # Provide a birth time for the cluster.
    combined_data = np.vstack(list(datasets.values()))

    # Call our new script to get the dictionaries
    model_stats = cwc.assign_model_weights_and_times(combined_data)

    inspect_model_weights(model_stats)

    # Pre-compute a high-resolution grid for instant interpolation from time to z
    z_grid = np.linspace(0, 100, 10000)
    t_grid = Planck18.age(z_grid).to(u.Myr).value
    t_grid_gyr = Planck18.age(z_grid).to(u.Gyr).value  # Added for secondary axis

    # Reverse arrays because np.interp expects the t_grid to be strictly increasing
    t_grid_rev = t_grid[::-1]
    t_grid_gyr_rev = t_grid_gyr[::-1]  # Added for secondary axis
    z_grid_rev = z_grid[::-1]

    # Define the redshift bins
    z_edges = np.linspace(0, z_max, num_bins + 1)
    z_centers = (z_edges[:-1] + z_edges[1:]) / 2
    t_edges_yr = Planck18.age(z_edges).to(u.yr).value
    dt_yr = np.abs(t_edges_yr[1:] - t_edges_yr[:-1])

    density_gpc3 = cluster_density_mpc3 * 1e9

    # Get the sum of all independent weights across ALL models in your dataset
    total_grid_weight = sum([info['weight'] for info in model_stats.values()])

    for (label, group_data), color in zip(sorted(datasets.items()), colors):
        if len(group_data) == 0:
            continue

        model_ids = group_data[:, 0]
        t_merger_myr = group_data[:, 5]

        # Retrieve the assigned birth time for every model in this group
        birth_times = np.array([model_stats[mid]['birth_time_myr'] for mid in model_ids])
        model_weights = np.array([model_stats[mid]['weight'] for mid in model_ids])

        # Absolute Cosmic Time = Birth Time + Delay Time
        t_merger_cosmic_myr = t_merger_myr + birth_times

        # Map the absolute cosmic merger times to redshift
        z_mergers = np.interp(t_merger_cosmic_myr, t_grid_rev, z_grid_rev, right=0.0)

        # Pass the model weights into the histogram
        counts, _ = np.histogram(z_mergers, bins=z_edges, weights=model_weights)

        # Calculate the Volumetric Rate Density R(z)
        rate_density = (counts / total_grid_weight) / dt_yr * density_gpc3

        # Mask out bins with 0 mergers
        mask = rate_density > 0

        # Plot
        ax.plot(z_centers[mask], rate_density[mask], marker='', linestyle='-',
                label=label, color=color, linewidth=2)

        # Plot the Individual Mergers as ticks on the top axis (Rug Plot)
        # transform=ax.get_xaxis_transform() locks the y-coordinate to the axis fraction (1.0 = exactly the top spine)
        ax.plot(z_mergers, np.full_like(z_mergers, 1.0), marker='|', color=color,
                linestyle='None', alpha=0.4, markersize=8,
                transform=ax.get_xaxis_transform(), clip_on=False)

    # === Formatting the plot to match the reference image ===
    ax.set_yscale('log')
    ax.set_xlim(0, z_max)

    ax.set_xlabel("Redshift $z$", fontsize=14)
    ax.set_ylabel(r"$\mathcal{R}(z) \, [\mathrm{Gpc}^{-3} \, \mathrm{yr}^{-1}]$", fontsize=14)
    ax.set_title(f"Look back Time of Merger/Collision Distribution Density by {group_name}", pad=20)

    # === Setting up the Secondary Axis (Top) ===
    def z_to_age(z):
        z_clipped = np.clip(z, 0, 100) # Prevents out-of-bounds calculations
        return np.interp(z_clipped, z_grid, t_grid_gyr)

    def age_to_z(age):
        age_clipped = np.clip(age, 0, t_grid_gyr[0])
        return np.interp(age_clipped, t_grid_gyr_rev, z_grid_rev)

    sec_ax = ax.secondary_xaxis('top', functions=(z_to_age, age_to_z))
    sec_ax.set_xlabel("Age of the Universe (Gyr)", fontsize=14)

    # Adding the clean major/minor gridlines from your reference image
    ax.grid(True, which='major', linestyle='-', alpha=0.4)
    ax.grid(True, which='minor', linestyle=':', alpha=0.3)

    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.show()

# === Main functions for starting plots ===
def setup_data():
    # Collisions of at least 1 compact object (WD, NS, or BH)
    co_collisions = np.loadtxt("data/CO_collisions_binaryfraction.dat")
    # Collisions of at least 1 WD
    wd_mergers = np.loadtxt("data/WD_mergers_binaryfraction.dat")

    sorted_co = sort_binary_mass(co_collisions)
    sorted_wd = sort_binary_mass(wd_mergers)

    # Filter to keep only systems with both k >= 10 (WD, NS, BH, or combinations)
    sorted_co = filter_k_values(sorted_co)
    sorted_co = remove_specific_model_group(sorted_co)
    sorted_co = remove_specific_model_group(sorted_co, target_rv=1.0)
    sorted_wd = filter_k_values(sorted_wd)
    sorted_wd = remove_specific_model_group(sorted_wd)
    sorted_wd = remove_specific_model_group(sorted_wd, target_rv=1.0)

    # Columns reference:
    # [0:"modelID", 1:"rv(pc)", 2:"Rgc(kpc)", 3:"Z", 4:"N(x10^5)", 5:"t_merger(Myr)", ...]

    # plot WD data
    plots(sorted_wd, "WD Mergers")

    # plot CO data
    plots(sorted_co, "CO Collisions")

    # Combine data
    combined_data = np.concatenate([sorted_co, sorted_wd[:, :12]], axis=0)
    plots(combined_data, "All Mergers")

    print("Finished plotting all data.")

def plots(data, model=""):
    # === Generic WD Plots ===
    #plot_m1_m2_combined(data, title_context=model)
    #mergers_vs_time(data, title_context=model)

    # === Overlay Plots for rv, Z, and N ===
    groups_to_plot = [(1, "rv(pc)"), (3, "Z"), (4, "N(x10^5)")]

    for col_idx, group_name in groups_to_plot:
        datasets, unique_values = separate_by_column(data, col_idx)
        print(f"\nFound {len(unique_values)} unique {group_name} values in " + model + f": {unique_values}")

        # Filter Extremes
        filtered_datasets = datasets
        for unique_value in unique_values:
            print(f"\n--- Checking {group_name} = {unique_value} ---")

            filtered_data = datasets[unique_value]
            print(f"  - Initial count: {len(filtered_data)} systems")
            filtered_data = filter_extreme_models(filtered_data, 0, percentile=95)
            print(f"  - After filtering: {len(filtered_data)} systems")
            filtered_datasets[unique_value] = filtered_data

        #!overlay_mergers_by_group(filtered_datasets, group_name=group_name, data_type=model)
        #!mergers_over_time_loglog_by_group(filtered_datasets, group_name=group_name, data_type=model)

    mass_datasets = generate_mass_groups(data)

    #!overlay_mass_fractions(mass_datasets, model_name=model)

    # === Test Model with r=0.5, z=0.3, N=8.0 ===
    mask_rv = np.isclose(data[:, 1], 0.5)
    mask_z = np.isclose(data[:, 3], 0.3)
    mask_n = np.isclose(data[:, 4], 8.0)

    # Combine conditions
    specific_model_mask = mask_rv & mask_z & mask_n

    # Extract the filtered data
    specific_data = data[specific_model_mask]

    print(f"\n--- Testing Specific Model (rv=0.5, Z=0.3, N=8) ---")
    print(f"Found {len(specific_data)} total mergers for these parameters.")

    if len(specific_data) > 0:
        # Plot the volumetric rate density
        specific_mass_datasets = generate_mass_groups(specific_data)
        plot_merger_rate_density(
            specific_mass_datasets,
            group_name="Model where (rv=0.5, Z=0.3, N=8)",
            z_max=2.0,
            num_bins=50,
            cluster_density_mpc3=1.0
        )
    else:
        print("No data found for this specific combination of parameters.")

    plot_merger_rate_density(mass_datasets, group_name=model, z_max=2.0, num_bins=50,
                             cluster_density_mpc3=1.0)


setup_data()
