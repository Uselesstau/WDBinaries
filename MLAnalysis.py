import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def find_anomalous_models(data, contamination=0.05):
    """
    Uses an Isolation Forest to find truly weird models.
    Calculates global averages for the model features and compares
    the anomalies against these averages using Z-scores.
    """
    print("\n" + "=" * 70)
    print("--- ML Anomaly Detection & Feature Diagnostics ---")
    print("=" * 70)

    unique_models = np.unique(data[:, 0])
    model_features = []

    for mid in unique_models:
        model_data = data[data[:, 0] == mid]

        # Features describing the "behavior" of the model
        n_mergers = len(model_data)
        mean_t = np.mean(model_data[:, 5])

        # Protect against models with only 1 merger (std would be 0)
        std_t = np.std(model_data[:, 5]) if n_mergers > 1 else 0.0

        mean_z_density = np.mean(n_mergers / (model_data[:, 3] + 1e-10))
        mean_m_tot = np.mean(model_data[:, 8] + model_data[:, 9])
        mean_m_ratio = np.mean(model_data[:, 8] / (model_data[:, 9] + 1e-10))

        model_features.append([mid, n_mergers, mean_t, std_t, mean_z_density, mean_m_tot, mean_m_ratio])

    model_features = np.array(model_features)
    feature_names = ["Count", "Mean T", "Std T", "Z-Density", "Mean M_tot", "Mean M_Ratio"]

    # Isolate the feature columns (ignore modelID for stats)
    X = model_features[:, 1:]

    # Calculate Global Averages and Standard Deviations
    global_means = np.mean(X, axis=0)
    global_stds = np.std(X, axis=0)

    print("\n[GLOBAL AVERAGES ACROSS ALL MODELS]")
    print(f"{'Feature':<15} | {'Global Mean':<12} | {'Global Std Dev':<12}")
    print("-" * 45)
    for name, mean, std in zip(feature_names, global_means, global_stds):
        print(f"{name:<15} | {mean:<12.2f} | ± {std:<12.2f}")

    # Standardize features for ML training
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train Isolation Forest
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    predictions = iso_forest.fit_predict(X_scaled)

    # Extract Anomalies
    anomalies = model_features[predictions == -1]

    print(f"\nTotal Models Evaluated: {len(unique_models)}")
    print(f"Anomalies Detected: {len(anomalies)} (Contamination set to {contamination * 100}%)")

    # Detailed Anomaly Report using Z-scores
    print("\n[FLAGGED ANOMALIES DIAGNOSTIC REPORT]")
    print("Values shown in brackets [ ] are Z-scores (Standard Deviations away from Global Mean).")

    header = f"{'ModelID':<8} | {'Count':<15} | {'Mean T (Myr)':<18} | {'Std T (Myr)':<18} | {'Z-Density':<15} | {'Mean M_tot':<15} | {'Mean Ratio'}"
    print(header)
    print("-" * len(header))

    for row in anomalies:
        mid = row[0]
        vals = row[1:]

        # Calculate Z-scores for this specific model
        # Z = (Value - Mean) / Std
        z_scores = (vals - global_means) / (global_stds + 1e-10)

        # Format the output strings combining raw values and Z-scores
        s_cnt = f"{vals[0]:.0f} [{z_scores[0]:+5.1f}]"
        s_mt = f"{vals[1]:.0f} [{z_scores[1]:+5.1f}]"
        s_std = f"{vals[2]:.0f} [{z_scores[2]:+5.1f}]"
        s_zden = f"{vals[3]:.1f} [{z_scores[3]:+5.1f}]"
        s_mtot = f"{vals[4]:.2f} [{z_scores[4]:+5.1f}]"
        s_mrat = f"{vals[5]:.2f} [{z_scores[5]:+5.1f}]"

        print(f"{mid:<8.0f} | {s_cnt:<15} | {s_mt:<18} | {s_std:<18} | {s_zden:<15} | {s_mtot:<15} | {s_mrat}")

    print("=" * 70 + "\n")

    anomaly_ids = anomalies[:, 0]
    clean_data = data[~np.isin(data[:, 0], anomaly_ids)]

    return clean_data, anomaly_ids