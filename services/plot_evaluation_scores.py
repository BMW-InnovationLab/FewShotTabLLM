import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict
import os

class EvaluationVisualizer:
    
    def __init__(self, evaluation_reports: list[str], output_dir: str = None):
        self.evaluation_reports = evaluation_reports
        self.output_dir = output_dir

    def visualize_evaluation_reports(self):
        for report in self.evaluation_reports:
            self._visualize_evaluation_report(report, output_dir=self.output_dir)

    def _visualize_evaluation_report(self, evaluation_report: str, output_dir: str = None):
        """
        Plot scores from a JSON file or dictionary with flexible score structures.
        
        Handles cases where "Scores" can be:
        - A single value (int/float)
        - A dictionary of column-value pairs
        - Empty/None
        - Missing metric entirely
        
        Args:
            json_file_path (str): Path to JSON file (optional if json_data provided)
            json_data (Dict): JSON data as dictionary (optional if json_file_path provided)
            figsize (tuple): Figure size for matplotlib
        """
        try:
            with open(evaluation_report, 'r') as f:
                data = json.load(f)

            if "Report" not in data:
                print("Warning: No 'Report' key found in JSON data")
                return None

            report = data["Report"]

            single_value_metrics = []
            dict_metrics = []
            correlation_metrics = []
            empty_metrics = []

            correlation_metric_names = ["Correlation_Similarity", "Pairwise_Correlation_Distance", "Contingency_Similarity"]

            # Extract Metrics and their scores
            for metric_name, metric_value in report.items():
                if not isinstance(metric_value, dict) or "Scores" not in metric_value:
                    empty_metrics.append(metric_name)
                    continue

                scores = metric_value["Scores"]

                if scores is None or (isinstance(scores, dict) and len(scores) == 0):
                    empty_metrics.append(metric_name)
                    continue

                if metric_name in correlation_metric_names:
                    correlation_metrics.append((metric_name, scores))
                elif isinstance(scores, (int, float)):
                    single_value_metrics.append((metric_name, scores))
                elif isinstance(scores, dict):
                    if any(isinstance(v, dict) for v in scores.values()):
                        continue
                    dict_metrics.append((metric_name, scores))
                else:
                    empty_metrics.append(metric_name)

            print(f"Found {len(single_value_metrics)} single-value metrics")
            print(f"Found {len(dict_metrics)} column metrics")
            print(f"Found {len(correlation_metrics)} correlation metrics")
            print(f"Found {len(empty_metrics)} empty/invalid metrics")

            figures = []

            # Plot correlation heatmaps for correlation metrics
            if correlation_metrics:
                for metric_name, scores in correlation_metrics:

                    if isinstance(scores, dict):
                        # Check if it's nested (has keys like 'statistic', 'pvalue')
                        first_value = next(iter(scores.values())) if scores else None

                        if isinstance(first_value, dict):
                            for sub_metric_name, sub_scores in scores.items():
                                fig = self._plot_correlation_heatmap(metric_name, sub_metric_name, sub_scores)
                                if fig:
                                    figures.append(fig)
                        else:
                            # Direct pairwise format
                            fig = self._plot_correlation_heatmap(metric_name, None, scores)
                            if fig:
                                figures.append(fig)

            # Plot single-value metrics in 3x3 subplots with max 7 metrics per subplot
            if single_value_metrics:
                metrics_per_subplot = 7
                subplots_per_figure = 9  # 3x3 grid

                # Calculate how many subplots we need in total
                num_subplots_needed = int(np.ceil(len(single_value_metrics) / metrics_per_subplot))
                num_figures_needed = int(np.ceil(num_subplots_needed / subplots_per_figure))

                subplot_idx = 0

                for fig_idx in range(num_figures_needed):
                    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
                    axes = axes.flatten()

                    for ax_idx in range(subplots_per_figure):
                        if subplot_idx >= num_subplots_needed:
                            axes[ax_idx].axis('off')
                            continue

                        # Get the metrics for this subplot
                        start_idx = subplot_idx * metrics_per_subplot
                        end_idx = min(start_idx + metrics_per_subplot, len(single_value_metrics))

                        metrics_subset = single_value_metrics[start_idx:end_idx]
                        metric_names = [m[0] for m in metrics_subset]
                        values = [m[1] for m in metrics_subset]

                        ax = axes[ax_idx]
                        bars = ax.bar(range(len(metric_names)), values, color='steelblue', alpha=0.8)
                        ax.set_title(f'Single-Value Metrics (Group {subplot_idx + 1})', fontsize=8, fontweight='bold')
                        ax.set_xlabel('Metrics', fontsize=6)
                        ax.set_ylabel('Score', fontsize=6)
                        ax.set_xticks(range(len(metric_names)))
                        ax.set_xticklabels(metric_names, rotation=45, ha='right', fontsize=6)
                        ax.grid(True, alpha=0.3)

                        for bar, value in zip(bars, values):
                            height = bar.get_height()

                            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{value:.3f}', ha='center', va='bottom', fontsize=6)

                        subplot_idx += 1

                    fig.suptitle(f'Single-Value Metrics - Page {fig_idx + 1}/{num_figures_needed}', fontsize=8, fontweight='bold')
                    plt.show()
                    figures.append(fig)



            if len(dict_metrics) == 0:
                if empty_metrics:
                    print(f"\nEmpty/Invalid metrics (not plotted): {empty_metrics}")
                return figures

            # Create figures with 3x3 subplots for dictionary metrics (max 9 plots per figure)
            plots_per_figure = 9
            num_figures = int(np.ceil(len(dict_metrics) / plots_per_figure))

            for fig_idx in range(num_figures):
                start_idx = fig_idx * plots_per_figure
                end_idx = min(start_idx + plots_per_figure, len(dict_metrics))


                fig, axes = plt.subplots(3, 3, figsize=(15, 12))
                axes = axes.flatten()

                figures.append(fig)

                plot_idx_in_figure = 0

                for i in range(start_idx, end_idx):
                    metric_name, scores_dict = dict_metrics[i]

                    ax = axes[plot_idx_in_figure]
                    columns = list(scores_dict.keys())
                    values = list(scores_dict.values())

                    values = [0 if v is None or (isinstance(v, float) and np.isnan(v)) else v for v in values]


                    bars = ax.bar(range(len(columns)), values, color='coral', alpha=0.8)
                    ax.set_title(f'{metric_name}', fontsize=10, fontweight='bold')
                    ax.set_xlabel('Columns', fontsize=8)
                    ax.set_ylabel('Score', fontsize=8)
                    ax.set_xticks(range(len(columns)))
                    ax.set_xticklabels(columns, rotation=45, ha='right', fontsize=6)
                    ax.grid(True, alpha=0.3)

                    for bar, value in zip(bars, values):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{value:.3f}', ha='center', va='bottom', fontsize=6)

                    plot_idx_in_figure += 1

                for i in range(plot_idx_in_figure, plots_per_figure):
                    axes[i].axis('off')

                dict_page_num = fig_idx + 1
                total_dict_pages = num_figures
                if single_value_metrics:
                    fig.suptitle(f'Column Metrics - Page {dict_page_num}/{total_dict_pages}', fontsize=8, fontweight='bold')
                else:
                    fig.suptitle(f'JSON Scores Visualization - Page {dict_page_num}/{total_dict_pages}', fontsize=8, fontweight='bold')
                plt.show()

            self._save_figures(figures, output_dir=output_dir, prefix=evaluation_report.split("/")[-1].replace(".json", ""))
            return figures

        except Exception as e:
            print(f"Error in _visualize_evaluation_report {evaluation_report}:\n{str(e)}")
            return None

    def _save_figures(self, figures, output_dir: str = None, prefix: str = "evaluation"):
        """
        Save all generated figures to files.
        
        Args:
            output_dir: Directory to save figures. If None, creates 'evaluation_figures' directory
            prefix: Prefix for figure filenames
            dpi: Resolution for saved figures (default: 300)
        
        Returns:
            List of saved file paths
        """
        try:
            if not figures:
                print("No figures to save. Please run visualize_evaluation_report() first.")
                return []

            # Create output directory if not specified
            if output_dir is None:
                output_dir = "evaluation_figures"

            output_dir = os.path.join(output_dir, prefix)
            os.makedirs(output_dir, exist_ok=True)

            saved_paths = []

            for idx, fig in enumerate(figures, start=1):
                filename = f"{prefix}_figure_{idx}.png"
                filepath = os.path.join(output_dir, filename)

                try:
                    # Use lower DPI for very large figures to prevent size issues
                    # Maximum image dimension in matplotlib is 2^23 pixels
                    fig_width, fig_height = fig.get_size_inches()
                    max_dim = max(fig_width, fig_height)
                    
                    # Check for unreasonable figure sizes (likely a matplotlib bug)
                    if fig_width > 1000 or fig_height > 1000:
                        print(f"Warning: Figure {idx} has unreasonable size {fig_width:.1f}x{fig_height:.1f} inches, skipping...")
                        continue
                    
                    # Calculate safe DPI, ensuring it's at least 50 and at most 300
                    if max_dim > 0:
                        calculated_dpi = int((2**23) / max_dim) - 100
                        safe_dpi = max(50, min(300, calculated_dpi))
                    else:
                        safe_dpi = 300
                    
                    # Double-check that the resulting image won't be too large
                    max_pixels_width = int(fig_width * safe_dpi)
                    max_pixels_height = int(fig_height * safe_dpi)
                    
                    # If still too large, reduce further
                    while max(max_pixels_width, max_pixels_height) > 2**23 and safe_dpi > 50:
                        safe_dpi = max(50, safe_dpi - 10)
                        max_pixels_width = int(fig_width * safe_dpi)
                        max_pixels_height = int(fig_height * safe_dpi)
                    
                    print(f"Saving figure {idx} with size {fig_width:.1f}x{fig_height:.1f} inches at {safe_dpi} DPI ({max_pixels_width}x{max_pixels_height} pixels)")
                    
                    # Try without bbox_inches='tight' first to avoid matplotlib size recalculation bugs
                    try:
                        fig.savefig(filepath, dpi=safe_dpi, bbox_inches='tight')
                    except (ValueError, TypeError) as e:
                        # If tight layout fails, try without it
                        print(f"Tight bbox failed for figure {idx}, trying standard save...")
                        fig.savefig(filepath, dpi=safe_dpi)
                    
                    saved_paths.append(filepath)
                    print(f"Saved figure {idx}/{len(figures)}: {filepath}")
                except Exception as e:
                    print(f"Error saving figure {idx}: {str(e)}")
                    print(f"Figure size: {fig.get_size_inches()[0]:.1f}x{fig.get_size_inches()[1]:.1f} inches")
                    # Try to save with minimal DPI and no tight bbox as last resort
                    try:
                        fig.savefig(filepath, dpi=50)
                        saved_paths.append(filepath)
                        print(f"Saved figure {idx}/{len(figures)} with reduced quality: {filepath}")
                    except:
                        print(f"Failed to save figure {idx}, skipping...")

            print(f"\nAll {len(figures)} figures saved to: {output_dir}")
            return saved_paths
        except Exception as e:
            raise Exception(f"Error saving figures: {str(e)}")

    def _plot_correlation_heatmap(self, metric_name: str, sub_metric_name: str, scores: Dict[str, float]):
        """
        Plot a correlation heatmap from pairwise column scores.
        
        Args:
            metric_name: Name of the main metric
            sub_metric_name: Name of the sub-metric (e.g., 'statistic', 'pvalue') or None
            scores: Dictionary with keys like "Column_1_Column_2" and correlation values
        
        Returns:
            matplotlib figure or None if plotting fails
        """
        try:
            # Extract unique column names from the score keys by splitting on "__"
            columns_set = set()
            for key in scores.keys():
                if "__" in key:
                    parts = key.split("__")
                    if len(parts) == 2:
                        columns_set.add(parts[0])
                        columns_set.add(parts[1])
            
            # Convert to sorted list for consistent ordering
            columns = sorted(list(columns_set))
            
            # Create a mapping from column name to index
            col_to_idx = {col: idx for idx, col in enumerate(columns)}
            
            # Create correlation matrix
            n = len(columns)
            corr_matrix = np.ones((n, n))
            
            # Fill the matrix with pairwise correlations
            for key, value in scores.items():
                # Handle None, NaN, or string values
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    value = 0
                elif isinstance(value, str):
                    value = 0.0
                
                # Split the key by "__" to extract the two column names
                if "__" in key:
                    parts = key.split("__")
                    if len(parts) == 2:
                        col1, col2 = parts
                        
                        # Find indices for both columns
                        if col1 in col_to_idx and col2 in col_to_idx:
                            i = col_to_idx[col1]
                            j = col_to_idx[col2]
                            
                            # Set both symmetric positions in the matrix
                            corr_matrix[i, j] = value
                            corr_matrix[j, i] = value

            
            # Create the heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            
            im = ax.imshow(corr_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
            
            # Set ticks and labels
            ax.set_xticks(np.arange(n))
            ax.set_yticks(np.arange(n))
            ax.set_xticklabels(columns, rotation=45, ha='right')
            ax.set_yticklabels(columns)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Correlation Score', rotation=270, labelpad=20)
            
            # Add text annotations
            for i in range(n):
                for j in range(n):
                    text = ax.text(j, i, f'{corr_matrix[i, j]:.3f}', ha="center", va="center", color="black", fontsize=9)
            
            # Set title
            if sub_metric_name:
                title = f'{metric_name} - {sub_metric_name}'
            else:
                title = metric_name
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            
            plt.show()
            
            return fig
            
        except Exception as e:
            print(f"Error in _plot_correlation_heatmap for {metric_name}:\n{str(e)}")
            return None