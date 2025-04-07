import subprocess
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Parameters for the scalability analysis
patch_sizes = range(1,41)
#processor_counts = [1, 2, 4, 8, 16, 24, 32]
#fixed proc count
nprocs = 24
#patch_size = 22
repetitions = 3
#Fixed size
size = 850
# Store results
cflags = ["--benchmark",""]
results = []
raw_results = []
srun_precommand = "srun -p q_student -t 1 -N 1 -c 32"



    
print(f"Starting scalability analysis for cs ...")
print("patch size| Processors | Mean runtime (s) | Std. dev. (s)")
print("-" * 70)
for patch_size in patch_sizes:

    runtimes = []

    for rep in range(repetitions):
        cmd = f"{srun_precommand} python3 julia_set/julia_par.py --size {size} --nprocs {nprocs} --patch {patch_size}"
        output = subprocess.check_output(cmd, shell=True, text=True)
        _, _, _, runtime_str = output.strip().split(';')
        runtimes.append(float(runtime_str))
        raw_results.append({
        'problem_size': size,
        'nprocs': nprocs,
        'runtime': float(runtime_str),
        'patch_size': patch_size,
        'c': "cs",
        "run_rep": rep
    })
        
    # if nprocs == 1:
    #     mean_sequential_runtime = np.mean(runtimes)

    # Calculate statistics
    mean_runtime = np.mean(runtimes)
    std_runtime = np.std(runtimes)

    # Calculate speedup and efficiency
    #speedup = mean_sequential_runtime / mean_runtime
    #efficiency = speedup / nprocs

    print(f"{patch_size:11} | {nprocs:10} | {mean_runtime:.6f} ± {std_runtime:.6f} ")
        
        
    results.append({
        'problem_size': size,
        'nprocs': nprocs,
        'mean_runtime': mean_runtime,
        'std_runtime': std_runtime,
        'patch_size': patch_size,
        'c': "cs"
    })
# Convert results to DataFrame
df = pd.DataFrame(results)
df_raw = pd.DataFrame(raw_results)

#plot x patchsize 
#plot y mean_runtime
plt.figure(figsize=(10, 6))
plt.errorbar(df['patch_size'], df['mean_runtime'], yerr=df['std_runtime'], fmt='o', label='Mean Runtime')
plt.title('Scalability Analysis: Mean Runtime vs Patch Size')
plt.xlabel('Patch Size')
plt.ylabel('Mean Runtime (s)')
plt.xticks(df['patch_size'])
#anotate patchsize where mean_runtime is minimum
min_index = df['mean_runtime'].idxmin()
min_patch_size = df['patch_size'][min_index]
min_runtime = df['mean_runtime'][min_index]

plt.annotate(f'Minimum runtime at patch size {min_patch_size} with runtime {min_runtime:.3f}s',
             xy=(min_patch_size, min_runtime), 
             xytext=(min_patch_size + 5, min_runtime + 0.5),
             arrowprops=dict(facecolor='black', arrowstyle='->'),
             fontsize=10,
             color='red')

# display error bars on the plot for each point std_runtime
#plt.errorbar(df['patch_size'], df['mean_runtime'], yerr=df['std_runtime'], fmt='o', label='Mean Runtime')

plt.grid()
plt.legend()
plt.savefig('plots/scalability_analysis_cs_2_4.png')


# Save results to CSV
df.to_csv('results/scalability_analysis_runs_2_4_mean.csv', index=False)
df_raw.to_csv('results/scalability_analysis_runs_2_4_raw.csv', index=False)

print("\nAnalysis complete. Results saved to scalability_analysis.csv")
print("Plot saved to scalability_analysis.png")
