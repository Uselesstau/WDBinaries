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

def _normalize_k_values(k_array):
    # Return k values as-is to get unique color for each k value
    return k_array.astype(int)

def _k_label(k_normalized):
    # Display the actual k value
    return f"k={int(k_normalized)}"

def plot_m1_m2_combined(data, title_context=""):
    #Plot m1 vs m2 (by mass) and by k-types in a single figure with 2 subplots
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
    #Plot log-log masses (by mass) and by k-types in a single figure with 2 subplots
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
    merge_time = data[:, 5]

    fig, ax = plt.subplots()
    ax.hist(merge_time, bins=20)
    
    title = f"Number of Mergers Over Time{f' - {title_context}' if title_context else ''}"
    ax.title.set_text(title)
    ax.set_xlabel("Merger Times (Myr)")
    ax.set_ylabel("Count")

    plt.tight_layout()
    plt.show()

def filter_k_values(data, k_min=10):
    #Remove points with any k<k_min (WD, NS, BH have k >= 10)
    k1 = data[:, 10]
    k2 = data[:, 11]
    mask = (k1 >= k_min) & (k2 >= k_min)
    return data[mask]

def separate_by_rv(data):
    rv_column = data[:, 1]
    unique_rv_values = np.unique(rv_column)
    rv_datasets = {}
      
    for rv in unique_rv_values:
        mask = rv_column == rv
        rv_datasets[rv] = data[mask]
      
    return rv_datasets, unique_rv_values

def filter_extreme_models(data, model_col_idx=0, percentile=95):
    """
    Takes a dataset (already grouped by RV) and removes any model
    that has an extreme number of entries compared to the other models.
    """
    if len(data) == 0:
        return data

    model_ids = data[:, model_col_idx]

    # 1. Get unique models and count the number of entries (rows) for each
    unique_models, counts = np.unique(model_ids, return_counts=True)

    # 2. Determine the threshold for entry counts
    threshold = np.percentile(counts, percentile)

    # 3. Identify models that fall at or below the threshold
    valid_models = unique_models[counts <= threshold]

    # 4. Keep ONLY the rows belonging to the valid models
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

def overlay_mergers_by_rv(rv_datasets, data_type="WD"):
    #Create overlay plot of merger/collision times for each RV value
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Define colors for each RV value
    colors = plt.cm.tab10(np.linspace(0, 1, len(rv_datasets)))
    
    # Calculate global min and max across all RV datasets for equal bins
    all_merge_times = []
    for rv_data in rv_datasets.values():
        all_merge_times.extend(rv_data[:, 5])
    
    global_min = np.min(all_merge_times)
    global_max = np.max(all_merge_times)
    
    # Create bins that span the entire range equally
    bins = np.linspace(global_min, global_max, 21)  # 20 bins (21 edges)
    
    for (rv, rv_data), color in zip(sorted(rv_datasets.items()), colors):
        merge_times = rv_data[:, 5]  # Column 5 is t_merger(Myr)
        ax.hist(merge_times, bins=bins, alpha=1.0, label=f"RV = {rv} pc", 
                color=color, edgecolor=color, histtype='step', linewidth=2)
    
    ax.set_xlabel("Merger Times (Myr)")
    ax.set_ylabel("Count")
    ax.set_title(f"Overlaid Merger/Collision Times by RV - {data_type}")
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def mergers_over_time_loglog(rv_datasets, data_type="WD", start_time=150.0):
    # Plot merger/collision counts over time in log-log scale (non-cumulative)
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Define colors for each RV value
    colors = plt.cm.tab10(np.linspace(0, 1, len(rv_datasets)))
    
    # Calculate global min and max for equal bin edges across all datasets
    all_merge_times = []
    for rv_data in rv_datasets.values():
        valid = rv_data[:, 5][rv_data[:, 5] >= start_time]
        all_merge_times.extend(valid)
    
    if len(all_merge_times) > 0:
        global_min = np.max([np.min(all_merge_times), start_time])
        global_max = np.max(all_merge_times)
        
        # Create equal-sized bins for all datasets
        num_bins = 20
        bins = np.logspace(np.log10(global_min), np.log10(global_max), num_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        for (rv, rv_data), color in zip(sorted(rv_datasets.items()), colors):
            merge_times = rv_data[:, 5]  # Column 5 is t_merger(Myr)
            
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
                    ax.loglog(nonzero_times, nonzero_counts, marker='o', linestyle='-', label=f"RV = {rv} pc", color=color, linewidth=2, markersize=6, alpha=0.7)
    
    ax.set_xlabel("Time (Myr)")
    ax.set_ylabel("Merger/Collision Count per Bin")
    ax.set_title(f"Merger/Collision Distribution Over Time (Log-Log) - {data_type}")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    
    plt.tight_layout()
    plt.show()




# Collisions of at least 1 compact object (WD, NS, or BH)
CO_collisions = np.loadtxt("data/CO_collisions_binaryfraction.dat")
# Collisions of at least 1 WD
WD_mergers = np.loadtxt("data/WD_mergers_binaryfraction.dat")

sorted_CO = sort_binary_mass(CO_collisions)
sorted_WD = sort_binary_mass(WD_mergers)

# Filter to keep only systems with both k >= 10 (WD, NS, BH, or combinations)
sorted_CO = filter_k_values(sorted_CO)
sorted_WD = filter_k_values(sorted_WD)

columns = ["modelID", "rv(pc)", "Rgc(kpc)", "Z", "N(x10^5)", "t_merger(Myr)", "ID1", "ID2", "M1(Msun)", "M2(Msun)", "k1", "k2", "r(pc)"]

# === Process WD mergers by RV values ===
wd_rv_datasets, wd_rv_values = separate_by_rv(sorted_WD)
print(f"Found {len(wd_rv_values)} unique RV values in WD mergers: {wd_rv_values}")

# === Process CO collisions by RV values ===
co_rv_datasets, co_rv_values = separate_by_rv(sorted_CO)
print(f"Found {len(co_rv_values)} unique RV values in CO collisions: {co_rv_values}")

# === WD Plots ===
plot_m1_m2_combined(sorted_WD, title_context="WD Mergers")
mergers_vs_time(sorted_WD, title_context="WD Mergers")
overlay_mergers_by_rv(wd_rv_datasets, data_type="WD Mergers")
mergers_over_time_loglog(wd_rv_datasets, data_type="WD Mergers")

print("\n=== WD Mergers Analysis by RV ===")
for rv in wd_rv_values:
    print(f"\n--- Checking RV = {rv} ---")
    # Filter Extremes
    rv_data = wd_rv_datasets[rv]
    print(f"  - Initial count: {len(rv_data)} systems")
    rv_data_filtered_WD = filter_extreme_models(rv_data, 0, percentile=95)
    print(f"  - After filtering: {len(rv_data_filtered_WD)} systems")

    # Plots
    context = f"WD Mergers, RV = {rv} pc"
    plot_m1_m2_combined(rv_data_filtered_WD, title_context=context)

# === CO Plots ===
plot_log_log_mass_combined(sorted_CO, title_context="CO collisions")
mergers_vs_time(sorted_CO, title_context="CO collisions")
overlay_mergers_by_rv(co_rv_datasets, data_type="CO Collisions")
mergers_over_time_loglog(co_rv_datasets, data_type="CO Collisions")

print("\n=== CO Collisions Analysis by RV ===")
for rv in co_rv_values:
    print(f"\n--- Checking RV = {rv} ---")
    # Filter Extremes
    rv_data = co_rv_datasets[rv]
    print(f"  - Initial count: {len(rv_data)} systems")
    rv_data_filtered_CO = filter_extreme_models(rv_data, 0, percentile=95)
    print(f"  - After filtering: {len(rv_data_filtered_CO)} systems")\

    # Plots
    context = f"CO Collisions, RV = {rv} pc"
    plot_log_log_mass_combined(rv_data_filtered_CO, title_context=context)

# overlay mergers/collisions for each rv, N, and Z value.
