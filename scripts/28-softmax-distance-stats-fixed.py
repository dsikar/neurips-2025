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
                std_path_dist = np.std(distances, ddof=1)  # Sample standard deviation
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
                    "StdPD": std_path_dist,
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
                    "StdPD": None,
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
\begin{longtable}{@{}cllrrrrrrrrrrc@{}}
\toprule
Exp & Net & Bins & Noise & AvgPD & StdPD & PDCnt & AvgED & AvgBD & AvgHI & AvgKL & FrCnt & LI \\
\midrule
\endfirsthead
\toprule
Exp & Net & Bins & Noise & AvgPD & StdPD & PDCnt & AvgED & AvgBD & AvgHI & AvgKL & FrCnt & LI \\
\midrule
\endhead
"""
        for _, row in df.iterrows():
            # Handle None/NaN values
            avg_path_dist = "N/A" if pd.isna(row['AvgPD']) else f"{row['AvgPD']:.4f}"
            std_path_dist = "N/A" if pd.isna(row['StdPD']) else f"{row['StdPD']:.4f}"
            avg_ed = "N/A" if pd.isna(row['AvgED']) else f"{row['AvgED']:.4f}"
            avg_bd = "N/A" if pd.isna(row['AvgBD']) else f"{row['AvgBD']:.4f}"
            avg_hi = "N/A" if pd.isna(row['AvgHI']) else f"{row['AvgHI']:.4f}"
            avg_kl = "N/A" if pd.isna(row['AvgKL']) else f"{row['AvgKL']:.4f}"
            latex += f"{int(row['Exp'])} & {row['Net']} & {int(row['Bins'])} & {int(row['Noise'])} & {avg_path_dist} & {std_path_dist} & {int(row['PDCnt'])} & {avg_ed} & {avg_bd} & {avg_hi} & {avg_kl} & {int(row['FrCnt'])} & {row['LI']} \\\\\n"
        latex += r"""\bottomrule
\caption{Experiment Statistics with Lane Invasion}
\label{tab:experiment_stats}
\end{longtable}
"""
        
        return df, latex
    
    except Exception as e:
        print(f"Table generation failed: {str(e)}")
        return pd.DataFrame(), ""


df, latex = generate_stats_table_v1(experiments)
print(latex)
# print(df.to_string(index=False))
# Optionally save to CSV
# df.to_csv("experiment_stats_v1.csv", index=False)
# print(df)
