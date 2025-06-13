import numpy as np
from distance_metrics import experiments, lane_invasion

def generate_latex_per_class_ed_table(exp_num, experiments, output_file=None):
    """
    Generate a LaTeX longtable for per-class Euclidean Distance (ED) statistics for one experiment.
    
    Args:
        exp_num (int): Experiment number to look up in the dictionary.
        experiments (list): List of dictionaries with experiment details (Exp, Label, Centroid_File, Data_File, Bins, Noise_Level).
        output_file (str, optional): File path to save the LaTeX code (e.g., 'exp_263_table.tex'). If None, prints to console.
    
    Returns:
        str: LaTeX code for the table.
    """
    try:
        # Find experiment
        exp = next((e for e in experiments if e["Exp"] == exp_num), None)
        if not exp:
            raise ValueError(f"Experiment {exp_num} not found in dictionary")
        
        # Extract details
        model = "CNN" if "CNN" in exp["Label"] else "ViT"
        num_bins = exp["Bins"]
        noise_pct = exp["Noise_Level"]
        # Create human-readable suffix
        noise_str = str(noise_pct).replace('.', 'p')  # e.g., 0 -> 0, 0.5 -> 0p5
        suffix = f"{model}_{num_bins}bins_{noise_str}noise"
        # Check for lane invasion
        invasion = lane_invasion(exp["Distance_File"])
        invasion_text = ", Lane Invasion Occurred" if invasion else ", No Lane Invasion"        
        # Load data and centroids
        data = np.load(exp["Data_File"], allow_pickle=True)
        centroids = np.load(exp["Centroid_File"])
        
        if centroids.shape[0] != num_bins:
            raise ValueError(f"Centroids shape {centroids.shape} mismatch with {num_bins} bins")
        
        # Initialize per-class lists
        class_ed = [[] for _ in range(num_bins)]
        total_frames = len(data)
        
        # Compute EDs per class
        for entry in data:
            softmax = entry[0]
            predicted_class = int(entry[1])
            
            # Handle CNN (1, n) or ViT (n,) softmax
            softmax = softmax[0] if softmax.ndim == 2 else softmax
            if softmax.shape[0] != num_bins:
                continue
            
            # Normalize softmax
            softmax_sum = np.sum(softmax)
            if not np.isclose(softmax_sum, 1.0, rtol=1e-5):
                softmax = softmax / softmax_sum
            
            # Compute ED
            ed = np.sqrt(np.sum((softmax - centroids[predicted_class]) ** 2))
            class_ed[predicted_class].append(ed)
        
        # Build table rows
        rows = []
        for cls in range(num_bins):
            num_frames = len(class_ed[cls])
            pct_frames = (num_frames / total_frames * 100) if total_frames else 0
            avg_ed = np.mean(class_ed[cls]) if class_ed[cls] else None
            rows.append({
                "Class": cls,
                "NumFrames": num_frames,
                "PctPercent": f"{pct_frames:.2f}",
                "AvgED": f"{avg_ed:.4f}" if avg_ed is not None else "N/A"
            })
        
        # Generate LaTeX comment
        comment = r"""%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Experiment """ + str(exp_num) + f""": {model} Model, {num_bins} Bins, {noise_pct}% Noise %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

        # Generate LaTeX table
        table = r"""\begin{longtable}{@{}llll@{}}
\toprule
Class & Number of Frames & Percentage (\%) & Average ED \\
\midrule
\endfirsthead
\toprule
Class & Number of Frames & Percentage (\%) & Average ED \\
\midrule
\endhead
"""
        for row in rows:
            table += f"{row['Class']} & {row['NumFrames']} & {row['PctPercent']} & {row['AvgED']} \\\\\n"
        table += r"""\bottomrule
\caption{Per-Class ED Statistics: """ + f"{model}, {num_bins} Bins, {noise_pct}\% Noise, Experiment {exp_num}{invasion_text}" + r"""}
\label{tab:exp""" + str(exp_num) + f"_{suffix}" + r"""}
\end{longtable}
        """

        # Combine comment, blank line, table, and blank line
        latex = comment + "\n" + table + "\n"
        
        # Output LaTeX code
        if output_file:
            # Modify file name to include suffix
            base_name = output_file.rsplit('.', 1)[0] if '.' in output_file else output_file
            new_output_file = f"{base_name}_{suffix}.tex"
            with open(new_output_file, 'w') as f:
                f.write(latex)
            print(f"LaTeX table saved to {new_output_file}")
        # else:
        #     print("\nLaTeX Table for Experiment", exp_num)
        #     print(latex)
        # return in any case
        return latex
    
    except Exception as e:
        error_msg = f"Error for Exp {exp_num}: {str(e)}"
        print(error_msg)
        return error_msg

# latex_code = generate_latex_per_class_ed_table(263, experiments, output_file="exp_263_table.tex")    


# Test for Exp 262 and 257
# experiments = [...]  # Your provided dictionary
# for exp_num in [262, 257]:
#     test_lane_invasion(exp_num, experiments)

# 262
# distances_file = "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_5_cnn_balanced_pep_0_250611_1936.txt"
# print("Testing lane invasion for Exp 262")
# print(lane_invasion(distances_file))
# # 257
# distances_file = "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_5_vit_balanced_pep_20_250610_1559.txt"
# print("Testing lane invasion for Exp 257")
# print(lane_invasion(distances_file))

import numpy as np

def generate_all_latex_ed_tables(experiments, output_file="all_experiments_ed_tables.tex"):
    """
    Generate LaTeX longtables for per-class ED statistics for all experiments, sorted by Model, NumBins, NoisePct.
    
    Args:
        experiments (list): List of dictionaries with experiment details (Exp, Label, Centroid_File, Data_File, Distance_File, Bins, Noise_Level).
        output_file (str): File path to save the concatenated LaTeX code (default: 'all_experiments_ed_tables.tex').
    
    Returns:
        str: Concatenated LaTeX code for all tables.
    """
    try:
        # Sort experiments by Model (CNN, ViT), NumBins (3, 5), NoisePct (ascending)
        sorted_experiments = sorted(experiments, key=lambda x: (
            "CNN" if "CNN" in x["Label"] else "ViT",  # Alphabetical: CNN first
            x["Bins"],  # Ascending: 3, 5
            x["Noise_Level"]  # Ascending: 0, 1, 2, ...
        ))
        
        # Concatenate LaTeX tables
        latex_all = ""
        for exp in sorted_experiments:
            latex_table = generate_latex_per_class_ed_table(exp["Exp"], experiments)
            latex_all += latex_table + "\n"  # Add blank line after each table
        
        # Save to file
        with open(output_file, 'w') as f:
            f.write(latex_all)
        print(f"All LaTeX tables saved to {output_file}")
        
        return latex_all
    
    except Exception as e:
        error_msg = f"Error generating all tables: {str(e)}"
        print(error_msg)
        return error_msg
    
generate_all_latex_ed_tables(experiments, output_file="all_experiments_ed_tables.tex")    