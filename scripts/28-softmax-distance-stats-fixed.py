import numpy as np
import pandas as pd
from distance_metrics import compute_average_ed, get_per_class_ed
from distance_metrics import experiments, lane_invasion
# Experiment data (manually copied from your list for simplicity)

def generate_stats_table_v1(experiments):
    """
    Generate a table with experiment statistics for path distances and softmax metrics.
    
    Args:
        experiments (list): List of dictionaries with experiment details (Exp, Label, 
                           Centroid_File, Data_File, Distance_File, Noise_Level, Bins).
    
    Returns:
        tuple: (pandas.DataFrame, str) - DataFrame with statistics and LaTeX code.
    """
    try:
        # Initialize results list
        results = []
        
        for exp in experiments:
            exp_id = exp["Exp"]
            try:
                # Extract experiment details
                model = "CNN" if "CNN" in exp["Label"] else "ViT"
                num_bins = exp["Bins"]
                noise_pct = exp["Noise_Level"]
                # Check lane invasion
                invasion = lane_invasion(exp["Distance_File"])
                li = "T" if invasion else "F"
                
                # Compute distance metrics from Distance_File
                with open(exp["Distance_File"], 'r') as f:
                    distances = [float(line.strip()) for line in f if line.strip()]
                if not distances:
                    raise ValueError("Empty Distance_File")
                distances = np.array(distances)
                avg_path_dist = np.mean(distances)
                path_dist_count = len(distances)
                
                # Load Data_File and Centroid_File for softmax metrics
                data = np.load(exp["Data_File"], allow_pickle=True)
                centroids = np.load(exp["Centroid_File"])
                frame_count = len(data)
                
                if centroids.shape[0] != num_bins:
                    raise ValueError(f"Centroids shape {centroids.shape} mismatch with {num_bins} bins")
                
                # Compute softmax metrics
                ed_list, bd_list, hi_list, kl_list = [], [], [], []
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
                    
                    centroid = centroids[predicted_class]
                    
                    # Euclidean Distance
                    ed = np.sqrt(np.sum((softmax - centroid) ** 2))
                    ed_list.append(ed)
                    
                    # Bhattacharyya Distance
                    bd = -np.log(np.sum(np.sqrt(softmax * centroid)))
                    bd_list.append(bd)
                    
                    # Histogram Intersection
                    hi = np.sum(np.minimum(softmax, centroid))
                    hi_list.append(hi)
                    
                    # Kullback-Leibler Divergence
                    kl = np.sum(centroid * np.log10((centroid + 1e-10) / (softmax + 1e-10)))
                    kl_list.append(kl)
                
                if not ed_list:
                    raise ValueError("No valid frames processed")
                
                # Compute averages
                avg_ed = np.mean(ed_list)
                avg_bd = np.mean(bd_list)
                avg_hi = np.mean(hi_list)
                avg_kl = np.mean(kl_list)
                
                # Append results
                results.append({
                    "Exp": exp_id,
                    "Net": model,
                    "Bins": num_bins,
                    "Noise": noise_pct,
                    "AvgPD": avg_path_dist,
                    "PDCnt": path_dist_count,
                    "AvgED": avg_ed,
                    "AvgBD": avg_bd,
                    "AvgHI": avg_hi,
                    "AvgKL": avg_kl,
                    "FrCnt": frame_count,
                    "LI": li
                })
                
            except Exception as e:
                print(f"Error for Exp {exp_id}: {str(e)}")
                results.append({
                    "Exp": exp_id,
                    "Net": "N/A",
                    "Bins": 0,
                    "Noise": 0,
                    "AvgPD": None,
                    "PDCnt": 0,
                    "AvgED": None,
                    "AvgBD": None,
                    "AvgHI": None,
                    "AvgKL": None,
                    "FrCnt": 0,
                    "LI": "N/A"
                })
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Sort DataFrame by Model, NumBins, NoisePct
        df = df.sort_values(by=["Net", "Bins", "Noise"])
        
        # Safely print DataFrame
        pd.set_option('display.float_format', '{:.4f}'.format)
        print(df.to_string(index=False, justify='center'))
        
   
        # Generate LaTeX table
        latex = r"""
\begin{longtable}{@{}cllrrrrrrrrrc@{}}
\toprule
Exp & Net & Bins & Noise & AvgPD & PDCnt & AvgED & AvgBD & AvgHI & AvgKL & FrCnt & LI \\
\midrule
\endfirsthead
\toprule
Exp & Net & Bins & Noise & AvgPD & PDCnt & AvgED & AvgBD & AvgHI & AvgKL & FrCnt & LI \\
\midrule
\endhead
"""
        for _, row in df.iterrows():
            # Handle None/NaN values
            avg_path_dist = "N/A" if pd.isna(row['AvgPD']) else f"{row['AvgPD']:.4f}"
            avg_ed = "N/A" if pd.isna(row['AvgED']) else f"{row['AvgED']:.4f}"
            avg_bd = "N/A" if pd.isna(row['AvgBD']) else f"{row['AvgBD']:.4f}"
            avg_hi = "N/A" if pd.isna(row['AvgHI']) else f"{row['AvgHI']:.4f}"
            avg_kl = "N/A" if pd.isna(row['AvgKL']) else f"{row['AvgKL']:.4f}"
            latex += f"{int(row['Exp'])} & {row['Net']} & {int(row['Bins'])} & {int(row['Noise'])} & {avg_path_dist} & {int(row['PDCnt'])} & {avg_ed} & {avg_bd} & {avg_hi} & {avg_kl} & {int(row['FrCnt'])} & {row['LI']} \\\\\n"
        latex += r"""\bottomrule
\caption{Experiment Statistics with Lane Invasion}
\label{tab:experiment_stats}
\end{longtable}
"""
        header = r"""%%%%%%%%%%%%%%%%%%%%%%
% MAIN RESULTS TABLE %
%%%%%%%%%%%%%%%%%%%%%%

% table generated by experiment 268"""
        latex = header + "\n" + latex

        return df, latex
    
    except Exception as e:
        print(f"Table generation failed: {str(e)}")
        return pd.DataFrame(), ""


df, latex = generate_stats_table_v1(experiments)
# print(latex)
# print(df.to_string(index=False))
# Optionally save to CSV
# df.to_csv("experiment_stats_v1.csv", index=False)
# print(df)
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def plot_noise_vs_metrics(df, output_dir='plots'):
    """
    Plot Noise vs normalized metrics (AvgPD, AvgED, AvgBD, AvgHI, AvgKL) for each Net+Bins combination.
    
    Args:
        df (pandas.DataFrame): DataFrame with columns Exp, Net, Bins, Noise, AvgPD, AvgED, AvgBD, AvgHI, AvgKL, etc.
        output_dir (str): Directory to save plots (created if not exists).
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter for Noise values: 0, 10, 20, 30, 40, 50
    df = df[df['Noise'].isin([0, 10, 20, 30, 40, 50])]
    
    # Metrics to plot
    metrics = ['AvgPD', 'AvgED', 'AvgBD', 'AvgHI', 'AvgKL']
    
    # Normalize metrics to [0, 1]
    scaler = MinMaxScaler()
    df[metrics] = scaler.fit_transform(df[metrics])
    
    # Unique Net+Bins combinations
    groups = df.groupby(['Net', 'Bins'])
    
    for (net, bins), group in groups:
        # Sort by Noise for plotting
        group = group.sort_values('Noise')
        
        # Create plot
        plt.figure(figsize=(8, 6))
        colors = ['b', 'g', 'r', 'c', 'm']
        markers = ['o', 's', '^', 'd', '*']

        # Add shading for LI = T
        li_true = group[group['LI'] == 'T']
        if not li_true.empty:
            first_li_true_noise = li_true['Noise'].min()
            plt.axvspan(first_li_true_noise, 50, alpha=0.2, color='red', label='Lane Invasion')
                
        for metric, color, marker in zip(metrics, colors, markers):
            plt.plot(group['Noise'], group[metric], color=color, marker=marker, linestyle='-', label=metric)
        
        plt.xlabel('Noise (%)')
        plt.ylabel('Normalized Metric Value')
        plt.title(f'{net}, {bins} Bins: Noise vs Metrics')
        plt.grid(True)
        plt.legend()
        plt.xticks([0, 10, 20, 30, 40, 50])
        
        # Save plot
        filename = f"{net.lower()}_{bins}bins_noise_metrics.png"
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

# plot_noise_vs_metrics(df, output_dir='plots')  

def generate_youtube_table(experiments, df):
    """
    Generate a LaTeX longtable for experiments with YouTube links, including Exp, Net, Bins, Noise, LI, and YouTube link.
    
    Args:
        experiments (list): List of experiment dictionaries with keys Exp, Label, Noise_Level, Bins, optional Youtube.
        df (pandas.DataFrame): DataFrame with columns Exp, Net, Bins, Noise, LI, etc.
    
    Returns:
        str: LaTeX code for the longtable.
    """
    # Filter experiments with YouTube links
    youtube_exps = [exp for exp in experiments if 'Youtube' in exp]
    
    # Create DataFrame from YouTube experiments
    youtube_data = [
        {
            'Exp': exp['Exp'],
            'Bins': exp['Bins'],
            'Noise': exp['Noise_Level'],
            'YouTube': exp['Youtube']
        }
        for exp in youtube_exps
    ]
    youtube_df = pd.DataFrame(youtube_data)
    
    # Merge with main DataFrame to get Net and LI
    merged_df = youtube_df.merge(df[['Exp', 'Net', 'LI']], on='Exp', how='left')
    
    # Sort by Net, Bins, Noise
    merged_df = merged_df.sort_values(['Net', 'Bins', 'Noise'])
    
    # Generate LaTeX table
    latex = r"""
\begin{longtable}{@{}clcrcc@{}}
\toprule
Exp & Net & Bins & Noise & LI & YouTube \\
\midrule
\endfirsthead
\toprule
Exp & Net & Bins & Noise & LI & YouTube \\
\midrule
\endhead
"""
    for _, row in merged_df.iterrows():
        youtube_link = r"\href{" + row['YouTube'] + r"}{Video}"
        latex += f"{int(row['Exp'])} & {row['Net']} & {int(row['Bins'])} & {int(row['Noise'])} & {row['LI']} & {youtube_link} \\\\\n"
    
    latex += r"""\bottomrule
\caption{Experiments with YouTube Video Links}
\label{tab:youtube_links}
\end{longtable}
"""
    return latex

youtube_latex = generate_youtube_table(experiments, df)
print(youtube_latex)
# Save to file
# with open('youtube_table.tex', 'w') as f:
#     f.write(youtube_latex)

import pandas as pd
import numpy as np

def generate_li_ed_table(experiments, df):
    """
    Generate a LaTeX longtable for LI = T experiments, comparing AvgED (overall) and AvgED-LI (last 10, 20, 30 predictions).
    
    Args:
        experiments (list): List of experiment dictionaries with keys Exp, Data_File, Centroid_File, Noise_Level, Bins.
        df (pandas.DataFrame): DataFrame with columns Exp, Net, Bins, Noise, AvgED, FrCnt, LI, etc.
    
    Returns:
        str: LaTeX code for the longtable.
    """
    # Filter for LI = T experiments
    li_true_df = df[df['LI'] == 'T'][['Exp', 'Net', 'Bins', 'Noise', 'AvgED', 'FrCnt', 'LI']]
    
    # Create dictionary mapping Exp to Data_File and Centroid_File
    exp_to_files = {
        exp['Exp']: {'Data_File': exp['Data_File'], 'Centroid_File': exp['Centroid_File']}
        for exp in experiments
    }
    
    # Compute AvgED-LI-10, AvgED-LI-20, AvgED-LI-30
    avg_ed_li_10, avg_ed_li_20, avg_ed_li_30 = [], [], []
    for exp_id in li_true_df['Exp']:
        files = exp_to_files.get(exp_id)
        if files and files['Data_File'] and files['Centroid_File']:
            try:
                # Load data and centroids
                distances = np.load(files['Data_File'], allow_pickle=True)
                centroids = np.load(files['Centroid_File'])
                
                # Select last 10, 20, 30 predictions
                last_10 = distances[-10:] if len(distances) >= 10 else distances
                last_20 = distances[-20:] if len(distances) >= 20 else distances
                last_30 = distances[-30:] if len(distances) >= 30 else distances
                
                # Compute Euclidean distances
                for last_n, avg_list in [(last_10, avg_ed_li_10), (last_20, avg_ed_li_20), (last_30, avg_ed_li_30)]:
                    ed_values = []
                    for entry in last_n:
                        softmax = entry[0].flatten()  # Shape: (1, num_bins) -> (num_bins,)
                        centroid_idx = entry[1]  # Integer index
                        centroid = centroids[centroid_idx].flatten()  # Shape: (num_bins,)
                        # Compute Euclidean distance
                        ed = np.sqrt(np.sum((softmax - centroid) ** 2))
                        ed_values.append(ed)
                    # Average Euclidean distances
                    avg_list.append(np.mean(ed_values) if ed_values else np.nan)
            except Exception as e:
                print(f"Error processing Exp {exp_id}: {e}")
                avg_ed_li_10.append(np.nan)
                avg_ed_li_20.append(np.nan)
                avg_ed_li_30.append(np.nan)
        else:
            print(f"Missing files for Exp {exp_id}")
            avg_ed_li_10.append(np.nan)
            avg_ed_li_20.append(np.nan)
            avg_ed_li_30.append(np.nan)
    
    # Add AvgED-LI columns to DataFrame
    li_true_df['AvgED-LI-10'] = avg_ed_li_10
    li_true_df['AvgED-LI-20'] = avg_ed_li_20
    li_true_df['AvgED-LI-30'] = avg_ed_li_30
    
    # Sort by Net, Bins, Noise
    li_true_df = li_true_df.sort_values(['Net', 'Bins', 'Noise'])
    
    # Generate LaTeX table
    latex = r"""
\begin{longtable}{@{}clcrrrrrc@{}}
\toprule
Exp & Net & Bins & Noise & AvgED & AvgED-LI-10 & AvgED-LI-20 & AvgED-LI-30 & FrCnt & LI \\
\midrule
\endfirsthead
\toprule
Exp & Net & Bins & Noise & AvgED & AvgED-LI-10 & AvgED-LI-20 & AvgED-LI-30 & FrCnt & LI \\
\midrule
\endhead
"""
    for _, row in li_true_df.iterrows():
        avg_ed = "N/A" if pd.isna(row['AvgED']) else f"{row['AvgED']:.4f}"
        avg_ed_li_10 = "N/A" if pd.isna(row['AvgED-LI-10']) else f"{row['AvgED-LI-10']:.4f}"
        avg_ed_li_20 = "N/A" if pd.isna(row['AvgED-LI-20']) else f"{row['AvgED-LI-20']:.4f}"
        avg_ed_li_30 = "N/A" if pd.isna(row['AvgED-LI-30']) else f"{row['AvgED-LI-30']:.4f}"
        latex += f"{int(row['Exp'])} & {row['Net']} & {int(row['Bins'])} & {int(row['Noise'])} & {avg_ed} & {avg_ed_li_10} & {avg_ed_li_20} & {avg_ed_li_30} & {int(row['FrCnt'])} & {row['LI']} \\\\\n"
    
    latex += r"""\bottomrule
\caption{Comparison of Overall and Last 10, 20, 30 Predictions Euclidean Distance for Lane Invasion Experiments}
\label{tab:li_ed_comparison}
\end{longtable}
"""
    return latex

li_ed_latex = generate_li_ed_table(experiments, df)
with open('li_ed_table.tex', 'w') as f:
    f.write(li_ed_latex)

print(li_ed_latex)
