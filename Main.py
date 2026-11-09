import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
from astropy.cosmology import Planck18
import astropy.units as u
import ClusterWeightCalculator as cwc
import MLAnalysis as ml
from scipy.ndimage import gaussian_filter1d


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

def mergers_vs_time_by_model(data, title_context="", max_figs=None, target_models=None, model_stats=None):
    """
    Plots a global overview of all models, then generates an independent 2-pane figure
    for individual models showing their isolated behavior and their overlaid context.

    Arguments:
    data - the raw data of Mergers or Collisions
    title_context - the title of the plot
    max_figs - safety limit for the number of individual figures to generate (prevents crashes).
               Set to "None" to plot ALL models.
    target_models - specific list of model IDs to plot (e.g., the anomalies array).
    model_stats - dictionary containing calculated birth times and weights (optional).
    """
    if len(data) == 0:
        print("    - No data to plot.")
        return

    model_ids = data[:, 0]
    unique_models = np.unique(model_ids)

    # Use target_models if provided, otherwise use all unique models
    if target_models is not None:
        models_to_plot = target_models
    else:
        models_to_plot = unique_models

    merge_times_all = data[:, 5]

    # Calculate global bins so all plots share the exact same x-axis
    global_min = np.min(merge_times_all)
    global_max = np.max(merge_times_all)
    bins = np.linspace(global_min, global_max, 21)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Pre-compute histograms and extract properties for ALL models for efficiency
    model_data_dict = {}
    for mid in unique_models:
        m_data = data[model_ids == mid]
        counts, _ = np.histogram(m_data[:, 5], bins=bins)

        # Grab properties from the first row of this model
        model_data_dict[mid] = {
            'counts': counts,
            'rv': m_data[0, 1],
            'Z': m_data[0, 3],
            'N': m_data[0, 4]
        }

    # --- All model count ---
    fig_main, ax_main = plt.subplots(figsize=(12, 7))
    colors_all = plt.cm.viridis(np.linspace(0, 1, len(unique_models)))

    for (mid, m_info), color in zip(model_data_dict.items(), colors_all):
        ax_main.plot(bin_centers, m_info['counts'], marker='', linestyle='-',
                     color=color, alpha=0.3, linewidth=1.5)

    ax_main.set_title(
        f"Mergers Over Time (All {len(unique_models)} Models){f' - {title_context}' if title_context else ''}",
        fontsize=14)
    ax_main.set_xlabel("Merger Times (Myr)", fontsize=12)
    ax_main.set_ylabel("Count per Bin", fontsize=12)
    ax_main.grid(axis='both', alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --- Individual model inspection ---
    # Apply safety limit to avoid crashing Matplotlib memory
    if max_figs is not None and len(models_to_plot) > max_figs:
        print(f"\n[Note] Limiting individual plots to {max_figs} to prevent IDE crashes.")
        print("Pass `max_figs=None` to plot all of them, or pass a specific list to `target_models`.\n")
        models_to_plot = models_to_plot[:max_figs]

    for mid in models_to_plot:
        # Failsafe if a target model isn't in the dataset
        if mid not in model_data_dict:
            continue

        m_info = model_data_dict[mid]

        # Fetch birth time and weight strings if model_stats was provided
        xtra_info_str = ""
        if model_stats is not None and mid in model_stats:
            birth_myr = model_stats[mid]['birth_time_myr']
            birth_z = model_stats[mid]['birth_z']
            weight = model_stats[mid]['weight']

            # Formatted with .2e for scientific notation
            xtra_info_str = f"\nBirth: {birth_myr:.1f} Myr (z={birth_z:.2f})  |  Weight: {weight:.2e}"

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Format the title with the physical properties
        fig.suptitle(
            f"Model ID: {mid:.0f}  |  $r_v$ = {m_info['rv']} pc  |  $Z$ = {m_info['Z']}  |  $N$ = {m_info['N']} $\\times"
            f" 10^5${xtra_info_str}",
            fontsize=16, fontweight='bold')

        # Isolated View
        ax1.plot(bin_centers, m_info['counts'], marker='o', linestyle='-', color='tab:red', linewidth=2.5)
        ax1.set_title("Isolated Model View", fontsize=14)
        ax1.set_xlabel("Merger Times (Myr)")
        ax1.set_ylabel("Count per Bin")
        ax1.grid(axis='both', alpha=0.3)

        # Overlaid view
        # First, draw the background "spaghetti" of all other models
        for other_mid, other_info in model_data_dict.items():
            if other_mid != mid:
                ax2.plot(bin_centers, other_info['counts'], color='gray', alpha=0.1, linewidth=1.5)

        # Then, draw the specific model bolded on top
        ax2.plot(bin_centers, m_info['counts'], marker='o', linestyle='-', color='tab:blue', linewidth=3.5,
                 label=f"Model {mid:.0f}")
        ax2.set_title("Context (Overlaid against all others)", fontsize=14)
        ax2.set_xlabel("Merger Times (Myr)")
        ax2.set_ylabel("Count per Bin")
        ax2.grid(axis='both', alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        plt.show()

def filter_k_values(data, k_min=10, k_max=12):
    """Removes k values smaller than k_min (k<10 = not a WD) and values greater than k_max (k>12 = not a WD) from the data

    Arguments:
    data - the raw data of Mergers or Collisions
    k_min - the minimum k value
    k_max - the maximum k value

    Returns:
    Masked data with k values smaller than k_min (k<10 = not a WD)
    """
    k1 = data[:, 10]
    k2 = data[:, 11]
    mask = (k1 >= k_min) & (k2 >= k_min) & (k1 <= k_max) & (k2 <= k_max)
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

def remove_specific_model_group(data, target_model_ids):
    """Removes the models with specific values due to their extreme results

    Arguments:
    data - the raw data of Mergers or Collisions
    target_model_id - the ID of the target model

    Returns:
    filtered_data - the filtered data
    """
    if len(data) == 0:
        return data

    # Create mask for each id
    mask_id = np.isin(data[:, 0], target_model_ids)

    # Invert the mask to KEEP everything else
    mask_to_keep = ~mask_id

    filtered_data = data[mask_to_keep]

    # Calculate and print stats
    removed_count = len(data) - len(filtered_data)
    print(f"    - Excluded {removed_count} mergers matching model ids={target_model_ids}")

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

def plot_merger_rate_density(datasets, group_name="", z_max=5.0, num_bins=50,
                             cluster_density_mpc3=2.1, model_stats=None):
    """
    Plots the merger rate density R(z) in Gpc^-3 yr^-1 vs redshift z.
    Includes a secondary top axis showing the Age of the Universe in Gyr.

    Arguments:
    datasets - dictionary of grouped datasets (e.g., separated by mass).
    group_name - the string name of the column grouped by
    z_max - maximum redshift to plot (default 10.0).
    num_bins - number of redshift bins.
    cluster_density_mpc3 - Assumed density of clusters in the universe in Mpc^-3.
    model_stats - dictionary of the models statistics for cluster birthtimes and weights
    """

    if model_stats is None:
        print ("Model statistics not provided!")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Define colors for each group value
    colors = plt.cm.tab10(np.linspace(0, 1, len(datasets)))

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

    # Get the sum of all independent weights across ALL models in dataset
    total_grid_weight = 0.0
    for info in model_stats.values():
        z_array = info['z_array']
        weight_dist = info['weight_dist']
        # Calculate the delta-z (width of the probability bins)
        dz = z_array[1] - z_array[0] if len(z_array) > 1 else 1.0
        # The true weight of the model is the integral (area) of its probability curve
        total_grid_weight += np.sum(weight_dist) * dz

    for (label, group_data), color in zip(sorted(datasets.items()), colors):
        if len(group_data) == 0:
            continue

        unique_mids = np.unique(group_data[:, 0])

        # Accumulator for the whole group
        group_counts = np.zeros(num_bins)
        rug_z_mergers = []  # Holds peak values for the visual top-ticks

        # Process model by model to apply unique weight distributions
        for mid in unique_mids:

            # Skip if model isn't in stats
            if mid not in model_stats: continue

            model_data = group_data[group_data[:, 0] == mid]
            tau = model_data[:, 5]  # Delay times (Myr)

            # === Peak Estimate ===
            # Save the "most likely" merger time just for the visual ticks on top
            t_birth_peak = model_stats[mid]['birth_time_myr']
            z_peak = np.interp(tau + t_birth_peak, t_grid_rev, z_grid_rev, right=-1.0)
            valid_z_peak = z_peak[z_peak >= 0]
            rug_z_mergers.extend(valid_z_peak)

            # === Cosmological Convolution ===
            z_array = model_stats[mid]['z_array']
            weight_dist = model_stats[mid]['weight_dist']
            dz = z_array[1] - z_array[0] if len(z_array) > 1 else 1.0

            # Convert the model's z_array into birth times
            t_birth_array = np.interp(z_array, z_grid, t_grid)

            # Create a 2D matrix: [Every Merger] evaluated at [Every Birth Redshift]
            # Shape: (N_mergers) x (N_z_bins)
            t_cosmic_2d = tau[:, None] + t_birth_array[None, :]

            # Convert the entire 2D matrix of cosmic times back into redshift
            z_cosmic_2d = np.interp(t_cosmic_2d, t_grid_rev, z_grid_rev, right=-1.0)

            # The true weight assigned to each point is its probability slice,
            # divided by mc_draws so we don't accidentally multiply the universe's density
            bin_weights = (weight_dist * dz)
            weights_2d = np.tile(bin_weights, (len(tau), 1))

            # Flatten the 2D arrays and drop them into the group's overall histogram
            counts, _ = np.histogram(z_cosmic_2d.flatten(), bins=z_edges, weights=weights_2d.flatten())
            group_counts += counts

        # Calculate the Volumetric Rate Density R(z) using the convolved counts
        rate_density = (group_counts / total_grid_weight) / dt_yr * density_gpc3

        mask = rate_density > 0

        # Plot main line
        ax.plot(z_centers[mask], rate_density[mask], marker='', linestyle='-',
                label=label, color=color, linewidth=2, alpha=0.75)  # Increased alpha slightly so it's readable

        # Plot the Individual Mergers as ticks on the top axis
        if len(rug_z_mergers) > 0:
            rug_z_mergers = np.array(rug_z_mergers)
            ax.plot(rug_z_mergers, np.full_like(rug_z_mergers, 1.0), marker='|', color=color,
                    linestyle='None', alpha=0.3, markersize=8,
                    transform=ax.get_xaxis_transform(), clip_on=False)

    # === Plot Reference curve - Cosmic Star Formation Rate ===
    # Create a smooth, high-resolution array of redshifts for the curve
    z_sfr = np.linspace(0, z_max, 200)

    # Madau & Fragos (2017) formula
    psi_z = 0.01 * ((1 + z_sfr) ** 2.6) / (1 + ((1 + z_sfr) / 3.2) ** 6.2)

    # Scale it so the peak sits exactly at 1.0 (10^0)
    psi_z_normalized = psi_z / np.max(psi_z)

    # Plot as a dashed reference line
    ax.plot(z_sfr, psi_z_normalized, linestyle='--', color='tab:blue',
            linewidth=1.5, alpha=0.8, label='Star Formation Rate')

    # === Formatting the plot to match the reference image ===
    ax.set_yscale('log')
    ax.set_xlim(0, z_max)
    ax.set_ylim(bottom=1e-1)

    ax.set_xlabel("Redshift $z$", fontsize=14)
    ax.set_ylabel(r"$\mathcal{R}(z) \, [\mathrm{Gpc}^{-3} \, \mathrm{yr}^{-1}]$", fontsize=14)
    ax.set_title(f"Look back Time of Merger/Collision Distribution Density by {group_name}", pad=20)

    # === Setting up the Secondary Axis (Top) ===
    def z_to_age(z):
        z_clipped = np.clip(z, 0, 100)  # Prevents out-of-bounds calculations
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


def plot_model_weight_distributions(model_stats, normalize=False, max_models_in_legend=20):
    """
    Plots the weight distribution over redshift for EVERY INDIVIDUAL MODEL.

    Arguments:
    model_stats: Dictionary output from assign_model_weights_and_times.
    normalize: If True, scales each model's curve so its peak touches 1.0.
               If False, plots the absolute raw weight values on a log scale.
    max_models_in_legend: Caps the legend size to prevent Matplotlib from crashing.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Sort models by their peak weight so the heaviest ones are plotted on top
    sorted_models = sorted(model_stats.items(), key=lambda item: item[1]['weight'])

    # Give every individual model a unique color
    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_models)))

    count = 0
    for (mid, stats), color in zip(sorted_models, colors):
        z_array = stats['z_array']
        weight_dist = stats['weight_dist']  # The unique weight value for every z bin!

        # Determine if we are plotting raw weights or normalized shapes
        if normalize and np.max(weight_dist) > 0:
            y_data = weight_dist / np.max(weight_dist)
        else:
            y_data = weight_dist

        # Only add a label for the legend up to the limit
        label = f"Model {mid:.0f}" if count < max_models_in_legend else ""

        # Plot the model's weight distribution curve
        ax.plot(z_array, y_data, color=color, alpha=0.6, linewidth=2, label=label)

        # Plot a small dot exactly at the peak (the estimated birth time)
        best_idx = np.argmax(y_data)
        ax.plot(z_array[best_idx], y_data[best_idx], marker='o', color=color, markersize=5, alpha=0.8)

        count += 1

    # === Formatting ===
    if normalize:
        ax.set_title("Normalized Weight Distribution per Model Over Redshift", fontsize=14)
        ax.set_ylabel("Relative Probability (Peak = 1.0)", fontsize=12)
    else:
        ax.set_title("Absolute Weight Distribution per Model Over Redshift", fontsize=14)
        ax.set_ylabel("Absolute Cosmological Weight", fontsize=12)
        ax.set_yscale('log')  # Log scale is required because raw weights vary by orders of magnitude

    ax.set_xlabel("Redshift $z$", fontsize=12)

    # Handle the legend safely
    if len(sorted_models) > max_models_in_legend:
        ax.legend(title=f"Top {max_models_in_legend} Models", loc='upper right', bbox_to_anchor=(1.15, 1))
        # Add a text box letting you know how many lines are actually plotted
        ax.text(1.02, 0.02, f"Total Models\nPlotted: {len(sorted_models)}",
                transform=ax.transAxes, fontsize=11,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    else:
        ax.legend(title="Models", loc='upper right', bbox_to_anchor=(1.15, 1))

    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

# === Main functions for starting plots ===
def setup_data():

    # Collision data
    co_collisions = np.loadtxt("data/CO_collisions_binaryfraction.dat")

    # WD data
    wd_mergers = np.loadtxt("data/WD_mergers_binaryfraction.dat")

    sorted_co = sort_binary_mass(co_collisions)
    sorted_wd = sort_binary_mass(wd_mergers)

    # Filter to keep only systems with both k >= 10 (WD, NS, BH, or combinations)
    sorted_co = filter_k_values(sorted_co)
    sorted_wd = filter_k_values(sorted_wd)

    # Columns reference:
    # [0:"modelID", 1:"rv(pc)", 2:"Rgc(kpc)", 3:"Z", 4:"N(x10^5)", 5:"t_merger(Myr)", ...]

    # Combine data
    combined_data = np.concatenate([sorted_co, sorted_wd[:, :12]], axis=0)

    # Find Anomalies with Isolation Forest
    clean_data, flagged_models = ml.find_anomalous_models(combined_data)
    sorted_co = remove_specific_model_group(sorted_co, flagged_models)
    sorted_wd = remove_specific_model_group(sorted_wd, flagged_models)

    # Call Cluster Weight Calculator across all the datasets (normalized all together)
    model_stats_all = cwc.assign_model_weights_and_times(combined_data, num_z_bins=1000, z_max=20.0)
    model_stats_filtered = cwc.assign_model_weights_and_times(clean_data, num_z_bins=1000, z_max=20.0)

    # Check Anomalies
    # mergers_vs_time_by_model(combined_data, model_stats=model_stats_all)

    # Compare WD, CO, and combined data
    datasets = {
        "CO Data": sorted_co,
        "WD Data": sorted_wd[:,:12],
        "Combined Data": clean_data,
    }
    #plot_merger_rate_density(datasets, group_name="WD, CO, and Combined Data", z_max=2.0, num_bins=50,
    #                         cluster_density_mpc3=2.1, model_stats=model_stats_all)

    # Print Earliest merger time
    print(f"\nEarliest merger time in Myrs: {clean_data[5].min()}\n")

    # Plot the data
    mass_datasets = generate_mass_groups(clean_data)
    plot_merger_rate_density(mass_datasets, group_name="Filtered Mass-Separated Data", z_max=6.5, num_bins=100,
                            cluster_density_mpc3=2.1, model_stats=model_stats_filtered)

    #cwc.inspect_model_weights(model_stats_filtered)
    #plot_model_weight_distributions(model_stats_filtered, normalize=True, max_models_in_legend=24)
    #plot_model_weight_distributions(model_stats_filtered, normalize=False, max_models_in_legend=24)


    print("Finished plotting all data.")

def plots(data, model=""):
    # === Generic WD Plots ===
    #plot_m1_m2_combined(data, title_context=model)
    #mergers_vs_time(data, title_context=model)

    # === Overlay Plots for rv, Z, and N ===
    groups_to_plot = [(1, "rv(pc)"), (3, "Z"), (4, "N(x10^5)")]

    # for col_idx, group_name in groups_to_plot:
        # datasets, unique_values = separate_by_column(data, col_idx)
        # print(f"\nFound {len(unique_values)} unique {group_name} values in " + model + f": {unique_values}")

        # Filter Extremes
        # filtered_datasets = datasets
        # for unique_value in unique_values:
            # print(f"\n--- Checking {group_name} = {unique_value} ---")

            # filtered_data = datasets[unique_value]
            # print(f"  - Initial count: {len(filtered_data)} systems")
            # filtered_data = filter_extreme_models(filtered_data, 0, percentile=95)
            # print(f"  - After filtering: {len(filtered_data)} systems")
            # filtered_datasets[unique_value] = filtered_data

        #!overlay_mergers_by_group(filtered_datasets, group_name=group_name, data_type=model)
        #!mergers_over_time_loglog_by_group(filtered_datasets, group_name=group_name, data_type=model)

    #!overlay_mass_fractions(mass_datasets, model_name=model)

setup_data()

# core radius
