# Script for scalability analysis of Julia set computation

import subprocess
import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd

# Parameters for the scalability analysis
problem_sizes = [165, 1020]
processor_counts = [1, 2, 4, 8, 16, 24, 32]
patch_size = 22
repetitions = 3

# Store results
results = []

print("Starting scalability analysis...")
print("Problem size | Processors | Mean runtime (s) | Speedup | Efficiency")
print("-" * 70)

for size in problem_sizes:
    # For each problem size, measure sequential runtime first (1 processor)
    sequential_runtimes = []

    for _ in range(repetitions):
        cmd = f"python julia_set.py --size {size} --nprocs 1 --patch {patch_size} --benchmark"
        output = subprocess.check_output(cmd, shell=True, text=True)
        _, _, _, runtime_str = output.strip().split(';')
        sequential_runtimes.append(float(runtime_str))

    mean_sequential_runtime = np.mean(sequential_runtimes)

    # Now measure for all processor counts
    for nprocs in processor_counts:
        runtimes = []

        for _ in range(repetitions):
            cmd = f"python julia_set.py --size {size} --nprocs {nprocs} --patch {patch_size} --benchmark"
            output = subprocess.check_output(cmd, shell=True, text=True)
            _, _, _, runtime_str = output.strip().split(';')
            runtimes.append(float(runtime_str))

        # Calculate statistics
        mean_runtime = np.mean(runtimes)
        std_runtime = np.std(runtimes)

        # Calculate speedup and efficiency
        speedup = mean_sequential_runtime / mean_runtime
        efficiency = speedup / nprocs

        print(f"{size:11} | {nprocs:10} | {mean_runtime:.6f} ± {std_runtime:.6f} | {speedup:.2f} | {efficiency:.2f}")

        # Store results
        results.append({
            'problem_size': size,
            'nprocs': nprocs,
            'mean_runtime': mean_runtime,
            'std_runtime': std_runtime,
            'speedup': speedup,
            'efficiency': efficiency
        })

# Convert results to DataFrame
df = pd.DataFrame(results)

# # Create tables for the report
# for size in problem_sizes:
#     size_df = df[df['problem_size'] == size]
#     print(f"\nTable for problem size {size}:")
#     print("Processors | Runtime (s) | Speedup | Efficiency")
#     print("-" * 50)
#     for _, row in size_df.iterrows():
#         print(
#             f"{row['nprocs']:10} | {row['mean_runtime']:.6f} | {row['speedup']:.2f} | {row['efficiency']:.2f}")

# # Create plots
# fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# # Plot runtimes
# for i, size in enumerate(problem_sizes):
#     size_df = df[df['problem_size'] == size]
#     axes[0, 0].plot(size_df['nprocs'], size_df['mean_runtime'],
#                     marker='o', linestyle='-', label=f'Size {size}')

# axes[0, 0].set_xlabel('Number of Processors')
# axes[0, 0].set_ylabel('Runtime (seconds)')
# axes[0, 0].set_title('Runtime vs. Number of Processors')
# axes[0, 0].grid(True)
# axes[0, 0].legend()

# # Plot speedup
# for i, size in enumerate(problem_sizes):
#     size_df = df[df['problem_size'] == size]
#     axes[0, 1].plot(size_df['nprocs'], size_df['speedup'],
#                     marker='o', linestyle='-', label=f'Size {size}')

#     # Add ideal speedup line
#     max_procs = max(processor_counts)
#     axes[0, 1].plot([1, max_procs], [1, max_procs], 'r--',
#                     label='Ideal Speedup' if i == 0 else "")

# axes[0, 1].set_xlabel('Number of Processors')
# axes[0, 1].set_ylabel('Speedup')
# axes[0, 1].set_title('Speedup vs. Number of Processors')
# axes[0, 1].grid(True)
# axes[0, 1].legend()

# # Plot efficiency
# for i, size in enumerate(problem_sizes):
#     size_df = df[df['problem_size'] == size]
#     axes[1, 0].plot(size_df['nprocs'], size_df['efficiency'],
#                     marker='o', linestyle='-', label=f'Size {size}')

#     # Add ideal efficiency line
#     axes[1, 0].axhline(y=1.0, color='r', linestyle='--',
#                        label='Ideal Efficiency' if i == 0 else "")

# axes[1, 0].set_xlabel('Number of Processors')
# axes[1, 0].set_ylabel('Parallel Efficiency')
# axes[1, 0].set_title('Parallel Efficiency vs. Number of Processors')
# axes[1, 0].grid(True)
# axes[1, 0].legend()

# # Plot efficiency vs problem size for different processor counts
# for nprocs in processor_counts:
#     nprocs_df = df[df['nprocs'] == nprocs]
#     if len(nprocs_df) >= 2:  # Only if we have at least 2 problem sizes
#         axes[1, 1].plot(nprocs_df['problem_size'], nprocs_df['efficiency'],
#                         marker='o', linestyle='-', label=f'{nprocs} procs')

# axes[1, 1].set_xlabel('Problem Size')
# axes[1, 1].set_ylabel('Parallel Efficiency')
# axes[1, 1].set_title('Parallel Efficiency vs. Problem Size')
# axes[1, 1].grid(True)
# axes[1, 1].legend()

# plt.tight_layout()
# plt.savefig('scalability_analysis.png')
# plt.show()

# Save results to CSV
df.to_csv('scalability_analysis.csv', index=False)

print("\nAnalysis complete. Results saved to scalability_analysis.csv")
print("Plot saved to scalability_analysis.png")
