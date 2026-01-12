# main.py
# Joshua Boquiren (@thejoshbq)
# boquiren@musc.edu

import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde, chi2_contingency
import pandas as pd

from pynapse.config.events import LEGACY_HER
from pynapse.analysis.preprocessing.epoch.pipelines import *
from pynapse.analysis.peri_event import *


def procure_data(basedir: str):
    sample_names = [s for s in os.listdir(basedir) if os.path.isdir(os.path.join(basedir, s))]
    samples = []
    for s in sample_names:
        sample_dir = os.path.join(basedir, s)
        FOVs = [f for f in os.listdir(sample_dir) if os.path.isdir(os.path.join(sample_dir, f))]
        for f in FOVs:
            FOV_dir = os.path.join(sample_dir, f)
            mat_files = [os.path.join(FOV_dir, m) for m in os.listdir(FOV_dir) if
                         m.endswith(".mat") and "popevents" not in m]
            npy_files = [os.path.join(FOV_dir, n) for n in os.listdir(FOV_dir) if
                         n.endswith(".npy") and "extracted" in n]
            try:
                sample = Sample(
                    event_data=mat_files,
                    signal_data=npy_files,
                    name=s,
                    fps=30,
                    frame_averaging=4,
                    frame_correction=False,
                    correction_file=None,
                    event_dict=LEGACY_HER
                )
                samples.append(sample)
            except Exception as e:
                print(f"  Sample {s} failed ({e}); skipping")
                continue
    if len(samples) == 0:
        return None
    population = Population(name=basedir, samples=samples)
    population_tensor = PopulationEventTensor(
        population,
        event_id=[22, 222],
        pre_event=10,
        post_event=11.6,
        min_trials=3,
        buffer_ms=1000,
        trace_preprocess=None,
        window_preprocess=OTIS_PIPE,
    )
    event_windows = population_tensor.get_event_windows()
    if len(event_windows) == 0:
        return None
    processed_neurons = []
    for sample_tensor in event_windows:
        if sample_tensor.shape[0] == 0:  # no valid trials
            continue
        mean_per_sample = np.nanmean(sample_tensor, axis=0)
        processed_neurons.append(mean_per_sample)
    if not processed_neurons:
        return None
    all_neurons = np.vstack(processed_neurons)
    return all_neurons

def plot_analysis(population_data: dict):
    n_pops = len(population_data)
    stages = sorted(population_data.keys())
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(4 * n_pops, 16))
    gs = GridSpec(5, n_pops, figure=fig, height_ratios=[1, 3, 1, 3, 1], hspace=0.4, wspace=0.3)
    event_frame = 75
    fps = 7.5
    post_seconds = 10.0
    custom_cmap = LinearSegmentedColormap.from_list("GreenBlackMagenta", ["green", "black", "magenta"])
    mean_traces = {}
    exc_kde_data = {}
    inh_kde_data = {}
    global_mean_min = np.inf
    global_mean_max = -np.inf
    global_exc_density_max = 0
    global_inh_density_max = 0
    for stage_name, data in population_data.items():
        mean_trace = np.nanmean(data, axis=0)
        time_axis = (np.arange(data.shape[1]) - event_frame) / fps
        mean_traces[stage_name] = (time_axis, mean_trace)
        global_mean_min = min(global_mean_min, mean_trace.min())
        global_mean_max = max(global_mean_max, mean_trace.max())
        post_slice = data[:, event_frame:event_frame + int(post_seconds * fps)]
        peak_frames_rel = np.argmax(post_slice, axis=1)
        peak_times_s = peak_frames_rel / fps
        exc_responsive = np.max(post_slice, axis=1) > 0.1
        exc_times = peak_times_s[exc_responsive]
        exc_kde_data[stage_name] = exc_times
        if len(exc_times) > 0:
            kde = gaussian_kde(exc_times)
            x = np.linspace(0, post_seconds, 300)
            global_exc_density_max = max(global_exc_density_max, kde(x).max())
        trough_frames_rel = np.argmin(post_slice, axis=1)
        trough_times_s = trough_frames_rel / fps
        inh_responsive = np.min(post_slice, axis=1) < -0.1
        inh_times = trough_times_s[inh_responsive]
        inh_kde_data[stage_name] = inh_times
        if len(inh_times) > 0:
            kde = gaussian_kde(inh_times)
            x = np.linspace(0, post_seconds, 300)
            global_inh_density_max = max(global_inh_density_max, kde(x).max())
    for col, stage_name in enumerate(stages):
        data = population_data[stage_name]
        ax_mean = fig.add_subplot(gs[0, col])
        for other_stage, (t, m) in mean_traces.items():
            if other_stage != stage_name:
                ax_mean.plot(t, m, color='green', alpha=0.3, linewidth=1)
        t_curr, m_curr = mean_traces[stage_name]
        ax_mean.plot(t_curr, m_curr, color='magenta', linewidth=3)
        ax_mean.axvline(0, color='white', linestyle='--', linewidth=2)
        ax_mean.axhline(0, color='gray', linestyle=':', linewidth=1)
        ax_mean.set_ylim(global_mean_min * 1.1, global_mean_max * 1.1)  # Shared scale + padding
        ax_mean.set_title(f"{stage_name}\nMean Activity", color='white')
        ax_mean.set_xlabel("Time (s)", color='white')
        ax_mean.tick_params(colors='white')
        ax_mean.grid(True, alpha=0.3)
        ax_exc_heat = fig.add_subplot(gs[1, col])
        post_slice_exc = data[:, event_frame:event_frame + int(8 * fps)]
        peak_frames = np.argmax(post_slice_exc, axis=1)
        sort_idx_exc = np.argsort(peak_frames)
        sorted_exc = data[sort_idx_exc]
        ax_exc_heat.imshow(sorted_exc, aspect='auto', cmap=custom_cmap,
                           vmin=-0.4, vmax=0.4, interpolation='nearest')
        ax_exc_heat.axvline(event_frame, color='white', linestyle='--', linewidth=2)
        ax_exc_heat.set_ylabel("Neurons (exc sort)", color='white')
        ax_exc_heat.set_xlabel("Frames", color='white')
        ax_exc_heat.tick_params(colors='white')
        ax_exc_kde = fig.add_subplot(gs[2, col])
        for other_stage, times in exc_kde_data.items():
            if other_stage != stage_name and len(times) > 0:
                kde = gaussian_kde(times)
                x = np.linspace(0, post_seconds, 300)
                y = kde(x)
                ax_exc_kde.plot(x, y, color='green', alpha=0.3, linewidth=1)
                ax_exc_kde.fill_between(x, y, alpha=0.1, color='green')
        curr_times = exc_kde_data[stage_name]
        if len(curr_times) > 0:
            kde_curr = gaussian_kde(curr_times)
            x_curr = np.linspace(0, post_seconds, 300)
            y_curr = kde_curr(x_curr)
            ax_exc_kde.plot(x_curr, y_curr, color='magenta', linewidth=3)
            ax_exc_kde.fill_between(x_curr, y_curr, alpha=0.3, color='magenta')
        ax_exc_kde.set_ylim(0, global_exc_density_max * 1.1)  # Shared scale
        ax_exc_kde.set_title(f"Exc Peaks", color='white', fontsize=10)
        ax_exc_kde.set_xlabel("Latency (s)", color='white')
        ax_exc_kde.tick_params(colors='white')
        ax_exc_kde.grid(True, alpha=0.3)
        ax_inh_heat = fig.add_subplot(gs[3, col])
        trough_frames = np.argmin(post_slice_exc, axis=1)
        sort_idx_inh = np.argsort(trough_frames)
        sorted_inh = data[sort_idx_inh]
        ax_inh_heat.imshow(sorted_inh, aspect='auto', cmap=custom_cmap,
                           vmin=-0.4, vmax=0.4, interpolation='nearest')
        ax_inh_heat.axvline(event_frame, color='white', linestyle='--', linewidth=2)
        ax_inh_heat.set_ylabel("Neurons (inh sort)", color='white')
        ax_inh_heat.set_xlabel("Frames", color='white')
        ax_inh_heat.tick_params(colors='white')
        ax_inh_kde = fig.add_subplot(gs[4, col])
        for other_stage, times in inh_kde_data.items():
            if other_stage != stage_name and len(times) > 0:
                kde = gaussian_kde(times)
                x = np.linspace(0, post_seconds, 300)
                y = kde(x)
                ax_inh_kde.plot(x, y, color='magenta', alpha=0.3, linewidth=1)
                ax_inh_kde.fill_between(x, y, alpha=0.1, color='magenta')
        curr_times = inh_kde_data[stage_name]
        if len(curr_times) > 0:
            kde_curr = gaussian_kde(curr_times)
            x_curr = np.linspace(0, post_seconds, 300)
            y_curr = kde_curr(x_curr)
            ax_inh_kde.plot(x_curr, y_curr, color='green', linewidth=3)
            ax_inh_kde.fill_between(x_curr, y_curr, alpha=0.3, color='green')
        ax_inh_kde.set_ylim(0, global_inh_density_max * 1.1)  # Shared scale
        ax_inh_kde.set_title(f"Inh Troughs", color='white', fontsize=10)
        ax_inh_kde.set_xlabel("Latency (s)", color='white')
        ax_inh_kde.tick_params(colors='white')
        ax_inh_kde.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def responder_proportions(population_data: dict, threshold: float = 0.15):
    results = []
    for stage, data in sorted(population_data.items()):
        post_slice = data[:, 75:75 + int(5 * 7.5)]
        pre_mean = np.nanmean(data[:, :75], axis=1)
        post_mean = np.nanmean(post_slice, axis=1)
        delta = post_mean - pre_mean
        n = len(delta)
        excited = np.sum(delta > threshold) / n
        inhibited = np.sum(delta < -threshold) / n
        se_exc = np.sqrt(excited * (1 - excited) / n) * 100
        se_inh = np.sqrt(inhibited * (1 - inhibited) / n) * 100
        results.append({
            'Stage': stage,
            'Excited_%': excited * 100,
            'Excited_SE': se_exc,
            'Inhibited_%': inhibited * 100,
            'Inhibited_SE': se_inh
        })
    df = pd.DataFrame(results)
    counts = []
    for _, row in df.iterrows():
        n = population_data[row['Stage']].shape[0]  # total neurons
        exc_count = int(row['Excited_%'] / 100 * n)
        inh_count = int(row['Inhibited_%'] / 100 * n)
        non_count = n - exc_count - inh_count
        counts.append([exc_count, inh_count, non_count])
    chi2, p, dof, expected = chi2_contingency(counts)
    print(f"\nChi-Square Test: χ²={chi2:.1f}, df={dof}, p={p:.2e}")
    if p < 0.05:
        print("Significant differences in responder proportions across stages!")
    ax = df.plot(x='Stage', y=['Excited_%', 'Inhibited_%'], kind='bar', stacked=False,
                 yerr=df[['Excited_SE', 'Inhibited_SE']].values.T,
                 capsize=5, color=['magenta', 'green'], figsize=(10, 6),
                 title='% Responders Post-Press (±SEM)')
    ax.set_ylabel('% Neurons')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()
    return df


from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


def cluster_post_event_subpopulations(population_data: dict, n_clusters: int = 6, post_seconds: float = 8.0):
    """
    GMM clustering on post-press traces per stage.
    - Standardize for shape focus.
    - Sort clusters by mean peak time.
    - Plot clustered heatmap + mean traces per stage.
    - Return proportions for stats.
    """
    event_frame = 75
    fps = 7.5
    post_frames = int(post_seconds * fps)

    prop_results = []
    for stage_name, data in sorted(population_data.items()):
        print(f"\nClustering post-event in {stage_name}...")
        post_slice = data[:, event_frame:event_frame + post_frames]

        # Optional responsive filter (focus active post-press)
        responsive = np.ptp(post_slice, axis=1) > 0.2  # range > threshold
        cluster_data = post_slice[responsive]

        if cluster_data.shape[0] < n_clusters:
            print(f"  Too few responsive neurons — skipping {stage_name}")
            continue

        # Standardize (shape over amplitude)
        scaler = StandardScaler()
        normalized = scaler.fit_transform(cluster_data.T).T

        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        labels = gmm.fit_predict(normalized)

        # Sort clusters by mean post-peak time
        cluster_peaks = [np.mean(np.argmax(cluster_data[labels == c], axis=1)) for c in range(n_clusters)]
        order = np.argsort(cluster_peaks)

        # Reorder
        sorted_labels = np.zeros_like(labels)
        for new, old in enumerate(order):
            sorted_labels[labels == old] = new
        sorted_data = cluster_data[np.argsort(sorted_labels)]

        # Plot
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), facecolor='black')
        fig.suptitle(f"{stage_name} Post-Event Clusters (n={cluster_data.shape[0]} responsive)", color='white')

        # Heatmap
        axes[0].imshow(sorted_data, aspect='auto', cmap='RdBu_r', vmin=-0.4, vmax=0.4)
        axes[0].axvline(0, color='white', linestyle='--')  # relative to post-start
        axes[0].set_ylabel("Neurons (clustered)", color='white')
        axes[0].set_xlabel("Frames Post-Press", color='white')
        axes[0].tick_params(colors='white')

        # Cluster means
        for c in range(n_clusters):
            cluster_mean = np.nanmean(cluster_data[labels == order[c]], axis=0)
            axes[1].plot(cluster_mean, label=f"Cluster {c + 1} ({np.sum(labels == order[c]) / len(labels) * 100:.1f}%)")
        axes[1].legend(facecolor='black', framealpha=0.8)
        axes[1].axvline(0, color='white', linestyle='--')
        axes[1].set_ylabel("Mean Activity", color='white')
        axes[1].set_xlabel("Frames Post-Press", color='white')
        axes[1].tick_params(colors='white')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Proportions
        props = np.bincount(labels, minlength=n_clusters) / len(labels) * 100
        for c, p in enumerate(props):
            prop_results.append({
                'Stage': stage_name,
                'Cluster': c + 1,
                'Proportion_%': p
            })

    prop_df = pd.DataFrame(prop_results)
    print("\n=== Post-Event Cluster Proportions ===")
    print(prop_df.round(1))

    # Bar plot proportions across stages
    prop_df.pivot(index='Stage', columns='Cluster', values='Proportion_%').plot(kind='bar', stacked=True)
    plt.ylabel('% Neurons')
    plt.title('Cluster Proportions Across Stages')
    plt.show()

    return prop_df


if __name__ == "__main__":
    basedir = "../data/"
    population_data = {}
    for population in sorted(os.listdir(basedir)):
        if not os.path.isdir(os.path.join(basedir, population)):
            continue
        if population.startswith('.'):
            continue
        result = procure_data(os.path.join(basedir, population))
        if result is not None:
            population_data[population] = result
    if population_data:
        plot_analysis(population_data)
        responder_proportions(population_data)
        cluster_post_event_subpopulations(population_data, n_clusters=6)

