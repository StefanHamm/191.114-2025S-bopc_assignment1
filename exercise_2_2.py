# Script for scalability analysis of Julia set computation

import subprocess
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Parameters for the scalability analysis
problem_sizes = [165, 1020]
processor_counts = [1, 2, 4, 8, 16, 24, 32]
patch_size = 22
repetitions = 3

# Store results
cflags = ["--benchmark",""]
results = []
raw_results = []
srun_precommand = "srun -p q_student -t 1 -N 1 -c 32"


for cflag in cflags:
    
    print(f"Starting scalability analysis for {cflag}...")
    print("Problem size | Processors | Mean runtime (s) | Speedup | Efficiency")
    print("-" * 70)
    for size in problem_sizes:
        # For each problem size, measure sequential runtime first (1 processor)
        sequential_runtimes = []

        mean_sequential_runtime = 0

        # Now measure for all processor counts
        for nprocs in processor_counts:
            runtimes = []

            for rep in range(repetitions):
                cmd = f"{srun_precommand} python3 julia_set/julia_par.py --size {size} --nprocs {nprocs} --patch {patch_size} {cflag}"
                output = subprocess.check_output(cmd, shell=True, text=True)
                _, _, _, runtime_str = output.strip().split(';')
                runtimes.append(float(runtime_str))
                if nprocs == 1:
                    sequential_runtimes.append(float(runtime_str))
                
                raw_results.append({
                'problem_size': size,
                'nprocs': nprocs,
                'runtime': float(runtime_str),
                'c': "cb" if cflag == "--benchmark" else "cs",
                "run_rep": rep
            })
                
            if nprocs == 1:
                mean_sequential_runtime = np.mean(runtimes)

            # Calculate statistics
            mean_runtime = np.mean(runtimes)
            std_runtime = np.std(runtimes)

            # Calculate speedup and efficiency
            speedup = mean_sequential_runtime / mean_runtime
            efficiency = speedup / nprocs

            print(f"{size:11} | {nprocs:10} | {mean_runtime:.6f} ± {std_runtime:.6f} | {speedup:.2f} | {efficiency:.2f}")
                
                
            results.append({
                'problem_size': size,
                'nprocs': nprocs,
                'mean_runtime': mean_runtime,
                'std_runtime': std_runtime,
                'speedup': speedup,
                'efficiency': efficiency,
                'c': "cb" if cflag == "--benchmark" else "cs"
            })

# Convert results to DataFrame
df_raw = pd.DataFrame(raw_results)

df = pd.DataFrame(results)


for workload_type in df['c'].unique():
    print(f"\n{'='*60}")
    print(f"Analysis for Workload: {workload_type}")
    print(f"{'='*60}")

    # Filter data for the current workload
    workload_df = df[df['c'] == workload_type].sort_values(by=['problem_size', 'nprocs'])

    # --- 1. Generate Table ---
    print(f"\nTable for Workload '{workload_type}': Mean Runtime, Speedup, and Efficiency")
    print("-" * 75)
    print(f"{'Size':>6} | {'Processors':>10} | {'Runtime (s)':>15} | {'Speedup':>10} | {'Efficiency':>12}")
    print("-" * 75)
    for _, row in workload_df.iterrows():
        print(
            f"{int(row['problem_size']):>6} | {int(row['nprocs']):>10} | {row['mean_runtime']:>15.6f} | "
            f"{row['speedup']:>10.2f} | {row['efficiency']:>12.2f}"
        )
    print("-" * 75)


    # --- 2. Generate Plots (Runtime, Speedup, Efficiency) ---
    fig, axes = plt.subplots(1, 3, figsize=(21, 6)) # 1 row, 3 columns for the plots
    fig.suptitle(f'Performance Analysis for Workload: {workload_type}', fontsize=16)

    # a) Absolute Running Time Plot
    ax = axes[0]
    for size in problem_sizes:
        size_df = workload_df[workload_df['problem_size'] == size]
        ax.plot(size_df['nprocs'], size_df['mean_runtime'],
                marker='o', linestyle='-', label=f'Size {size}')
    ax.set_xlabel('Number of Cores')
    ax.set_ylabel('Runtime (seconds)')
    ax.set_title('Absolute Running Time')
    ax.set_xticks(processor_counts) # Ensure ticks match actual core counts

    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend()


    # b) Relative Speed-up Plot
    ax = axes[1]
    ideal_speedup_line_plotted = False
    for size in problem_sizes:
        size_df = workload_df[workload_df['problem_size'] == size]
        ax.plot(size_df['nprocs'], size_df['speedup'],
                marker='s', linestyle='-', label=f'Size {size}')

        # Add ideal speedup line (only once)
        if not ideal_speedup_line_plotted:
             # Ensure the line covers the full range of processor counts
            ax.plot(processor_counts, processor_counts, 'r--', label='Ideal Speedup')
            ideal_speedup_line_plotted = True

    ax.set_xlabel('Number of Cores')
    ax.set_ylabel('Relative Speed-up (T_1 / T_N)')
    ax.set_title('Relative Speed-up')
    ax.set_xticks(processor_counts)
    # Set y-limit maybe slightly larger than max procs for ideal line visibility
    ax.set_ylim(bottom=0, top=32 * 1.1)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend()


    # c) Parallel Efficiency Plot
    ax = axes[2]
    ideal_efficiency_line_plotted = False
    for size in problem_sizes:
        size_df = workload_df[workload_df['problem_size'] == size]
        ax.plot(size_df['nprocs'], size_df['efficiency'],
                marker='^', linestyle='-', label=f'Size {size}')

        # Add ideal efficiency line (only once)
        if not ideal_efficiency_line_plotted:
            ax.axhline(y=1.0, color='r', linestyle='--', label='Ideal Efficiency (1.0)')
            ideal_efficiency_line_plotted = True

    ax.set_xlabel('Number of Cores')
    ax.set_ylabel('Parallel Efficiency (Speed-up / N)')
    ax.set_title('Parallel Efficiency')
    ax.set_xticks(processor_counts)
    ax.set_ylim(bottom=0, top=1.1) # Efficiency typically between 0 and 1
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend()
    #  # Add discussion point placeholder - replace with actual findings

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust rect to make space for suptitle
    plt.savefig(f'plots/scalability_analysis_{workload_type}_2_2.png')
    plt.show()


# Save results to CSV
df.to_csv('results/scalability_analysis_runs_2_2_mean.csv', index=False)
df_raw.to_csv('results/scalability_analysis_runs_2_2_raw.csv', index=False)

print("\nAnalysis complete. Results saved to scalability_analysis.csv")
print("Plot saved to scalability_analysis.png")
