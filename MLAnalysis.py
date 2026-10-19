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

        # Features describing the behavior of the model
        n_mergers = len(model_data)
        mean_t = np.mean(model_data[:, 5])

        # Protect against models with only 1 merger
        std_t = np.std(model_data[:, 5]) if n_mergers > 1 else 0.0

        mean_m_tot = np.mean(model_data[:, 8] + model_data[:, 9])
        mean_m_ratio = np.mean(model_data[:, 8] / (model_data[:, 9] + 1e-10))

        model_features.append([mid, mean_t, std_t, mean_m_tot, mean_m_ratio])

    model_features = np.array(model_features)
    feature_names = ["Mean T", "Std T", "Mean M_tot", "Mean M_Ratio"]

    # Isolate the feature columns
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

    header = f"{'ModelID':<8} | {'Mean T (Myr)':<18} | {'Std T (Myr)':<18} | {'Mean M_tot':<15} | {'Mean Ratio'}"
    print(header)
    print("-" * len(header))

    for row in anomalies:
        mid = row[0]
        vals = row[1:]

        # Calculate Z-scores for this specific model
        z_scores = (vals - global_means) / (global_stds + 1e-10)

        # Format the output strings combining raw values and Z-scores
        s_mt = f"{vals[0]:.0f} [{z_scores[0]:+5.1f}]"
        s_std = f"{vals[1]:.0f} [{z_scores[1]:+5.1f}]"
        s_mtot = f"{vals[2]:.2f} [{z_scores[2]:+5.1f}]"
        s_mrat = f"{vals[3]:.2f} [{z_scores[3]:+5.1f}]"

        print(f"{mid:<8.0f} | {s_mt:<18} | {s_std:<18} | {s_mtot:<15} | {s_mrat}")

    print("=" * 70 + "\n")

    anomaly_ids = anomalies[:, 0]
    clean_data = data[~np.isin(data[:, 0], anomaly_ids)]

    return clean_data, anomaly_ids