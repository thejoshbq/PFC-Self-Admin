# main.py

# System I/O
import os
import warnings

# Custom Python library
from pynapse.config.events import LEGACY_HER
from pynapse.core import SignalRecording, EventLog, Sample, Population, Project

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning) # suppress DeprecationWarnings

    def create_population(dir: str, name: str = None):
        sample_names = [s for s in os.listdir(dir) if os.path.isdir(os.path.join(dir, s))]
        samples = []
        for s in sample_names:
            sample_dir = os.path.join(dir, s)
            FOVs = [f for f in os.listdir(sample_dir) if os.path.isdir(os.path.join(sample_dir, f))]
            for f in FOVs:
                FOV_dir = os.path.join(sample_dir, f)
                mat_files = [os.path.join(FOV_dir, m) for m in os.listdir(FOV_dir) if
                             m.endswith(".mat") and "popevents" not in m]
                npy_files = [os.path.join(FOV_dir, n) for n in os.listdir(FOV_dir) if
                             n.endswith(".npy") and "extracted" in n]

                sig_rec = SignalRecording(npy_files, f"{s}_{f}_trace")
                print(f"{'*' * 10} SignalRecording {'*' * 10}")
                print(sig_rec)
                print()

                event_log = EventLog(mat_files, f"{s}_{f}_event_log", LEGACY_HER)
                print(f"{'*' * 10} EventLog {'*' * 10}")
                print(event_log)
                print()

                try:
                    sample = Sample(
                        event_data=event_log,
                        signal_data=sig_rec,
                        name=s,
                        fps=30,
                        frame_averaging=4,
                        frame_correction=False,
                        correction_file=None,
                    )
                    print(f"{'*' * 10} Sample {'*' * 10}")
                    print(sample)
                    samples.append(sample)
                    print()
                except Exception as e:
                    print(f"  Sample {s} failed ({e}); skipping")
                    continue
        if len(samples) == 0:
            return None

        population = Population(name=name, samples=samples)
        return population


    data_dir = "./data"
    populations = []
    for population_dir in sorted(os.listdir(data_dir)):
        if not os.path.isdir(os.path.join(data_dir, population_dir)):
            continue
        if population_dir.startswith('.'):
            continue

        population = create_population(os.path.join(data_dir, population_dir), population_dir)
        print(f"{'*' * 10} Population {'*' * 10}")
        print(population)
        print()
        if population is not None:
            populations.append(population)

    print(f"{'*' * 10} Project {'*' * 10}")
    project = Project(
        name="PFC Self-Admin Analysis",
        populations=populations,
        authors=["Jim Otis", "Beth Doncheck", "Josh Boquiren"],
        description="A peri-event response latency analysis of PFC self-administration in mouse visual cortex."
    )
    print(project)
    print("\n\n")