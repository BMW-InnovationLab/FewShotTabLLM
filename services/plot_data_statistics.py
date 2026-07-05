import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
import os

class DataVisualizer:
    def __init__(self, datasets: Dict[str, pd.DataFrame], output_dir: str = "dataset_statistics"):
        """
        Initialize DataVisualizer with multiple datasets.
        
        Args:
            datasets: Dictionary mapping dataset names to DataFrames
                     Example: {"Real": df1, "Synthetic_v1": df2, "Synthetic_v2": df3}
        """
        self.datasets = datasets
        self.dataset_names = list(datasets.keys())
        self.output_dir = output_dir
        self.colors = plt.cm.tab10(np.linspace(0, 1, len(datasets)))
    
    def visualize(self):
        """Visualize comparisons between all datasets."""
        figures = []
        try:
            fig = self._plot_distribution_comparison(self.datasets)
            if fig is not None:
                figures.append(fig)
        except Exception as e:
            print(f"Distribution comparison error:\n{e}")
            
        try:
            fig = self._plot_box_comparison(self.datasets)
            if fig is not None:
                figures.append(fig)
        except Exception as e:
            print(f"Box comparison error:\n{e}")
        
        try:
            stat_figs = self._plot_statistics_comparison(self.datasets)
            if stat_figs is not None:
                for fig in stat_figs:
                    if fig is not None:
                        figures.append(fig)
        except Exception as e:
            print(f"Statistics comparison error:\n{e}")

        try:            
            fig = self._plot_correlation_matrices(self.datasets)
            if fig is not None:
                figures.append(fig)
        except Exception as e:
            print(f"Correlation matrices error:\n{e}")
            
        return self._save_figures(figures, output_dir=self.output_dir)
    
    def _save_figures(self, figures, output_dir: str):
        """Save figures to the specified output directory with a given prefix."""
        try:
            os.makedirs(output_dir, exist_ok=True)
            saved_paths = []
            
            if not figures:
                print("No figures to save.")
                return saved_paths
                
            for idx, fig in enumerate(figures, start=1):
                # Skip None figures
                if fig is None:
                    print(f"Skipping figure {idx} (None)")
                    continue
                    
                filename = f"data_statistic_figure_{idx}.png"
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
                except Exception as e:
                    print(f"Error saving figure {idx}: {str(e)}")
                    print(f"Figure size: {fig.get_size_inches()[0]:.1f}x{fig.get_size_inches()[1]:.1f} inches")
                    # Try to save with minimal DPI and no tight bbox as last resort
                    try:
                        fig.savefig(filepath, dpi=50)
                        saved_paths.append(filepath)
                        print(f"Saved figure {idx} with reduced quality: {filepath}")
                    except:
                        print(f"Failed to save figure {idx}, skipping...")

            return saved_paths
        except Exception as e:
            raise Exception(f"Error in _save_figures:\n{str(e)}")

    def _plot_distribution_comparison(self, datasets: Dict[str, pd.DataFrame], columns=None, figsize=(15, 10)):
        """Plot distribution comparisons for multiple columns across all datasets."""
        try:
            # Get numerical columns from the first dataset
            first_dataset = next(iter(datasets.values()))
            if columns is None:
                columns = first_dataset.select_dtypes(include=[np.number]).columns

            n_cols = len(columns)
            n_rows = (n_cols + 2) // 3  # 3 columns per row

            fig, axes = plt.subplots(n_rows, 3, figsize=figsize)
            if n_rows == 1:
                axes = axes.reshape(1, -1)

            for i, col in enumerate(columns):
                row = i // 3
                col_idx = i % 3

                ax = axes[row, col_idx]

                # Plot histograms for all datasets
                for idx, (name, data) in enumerate(datasets.items()):
                    if col in data.columns:
                        ax.hist(data[col].dropna(), bins=30, alpha=0.5, label=name, density=True, color=self.colors[idx])

                ax.set_title(f'{col} Distribution')
                ax.set_xlabel(col)
                ax.set_ylabel('Density')
                ax.legend()
                ax.grid(True, alpha=0.3)

            # Hide empty subplots
            for i in range(n_cols, n_rows * 3):
                row = i // 3
                col_idx = i % 3
                axes[row, col_idx].set_visible(False)

            plt.tight_layout()
            plt.show()

            return fig
        except Exception as e:
            print(f"Error in _plot_distribution_comparison:\n{str(e)}")
            return None

    def _plot_box_comparison(self, datasets: Dict[str, pd.DataFrame], columns=None):
        """Create box plots for comparison across all datasets."""
        try:
            # Get numerical columns from the first dataset
            first_dataset = next(iter(datasets.values()))
            if columns is None:
                columns = first_dataset.select_dtypes(include=[np.number]).columns

            # Cap figure width to prevent excessive image sizes
            fig_width = min(4 * len(columns), 50)  # Cap at 50 inches wide
            fig, axes = plt.subplots(1, len(columns), figsize=(fig_width, 6))
            if len(columns) == 1:
                axes = [axes]

            for i, col in enumerate(columns):
                data_to_plot = []
                labels = []

                for name, data in datasets.items():
                    if col in data.columns:
                        data_to_plot.append(data[col].dropna())
                        labels.append(name)

                bp = axes[i].boxplot(data_to_plot, labels=labels, patch_artist=True)

                # Color the boxes
                for patch, color in zip(bp['boxes'], self.colors[:len(data_to_plot)]):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)

                axes[i].set_title(f'{col} Box Plot')
                axes[i].grid(True, alpha=0.3)
                axes[i].tick_params(axis='x', rotation=45)

            plt.tight_layout()
            plt.show()

            return fig

        except Exception as e:
            print(f"Error in _plot_box_comparison:\n{str(e)}")
            return None

    def _plot_correlation_matrices(self, datasets: Dict[str, pd.DataFrame], columns=None, figsize_per_plot=(8, 7)):
        """
        Plot correlation matrices for numerical columns of each dataset.
        
        Args:
            datasets: Dictionary mapping dataset names to DataFrames
            columns: List of column names to include (optional, defaults to all numerical columns)
            figsize_per_plot: Size of each correlation heatmap
        """
        try:
            # Get numerical columns from the first dataset
            first_dataset = next(iter(datasets.values()))
            if columns is None:
                columns = first_dataset.select_dtypes(include=[np.number]).columns.tolist()

            # Filter to only numerical columns that exist in all datasets
            numerical_cols = []
            for col in columns:
                if all(col in data.columns and pd.api.types.is_numeric_dtype(data[col])
                       for data in datasets.values()):
                    numerical_cols.append(col)

            if len(numerical_cols) < 2:
                print("Warning: Need at least 2 numerical columns to compute correlations")
                return None

            # Determine subplot layout
            n_datasets = len(datasets)
            n_cols_subplot = min(3, n_datasets)  # Max 3 columns
            n_rows_subplot = (n_datasets + n_cols_subplot - 1) // n_cols_subplot

            fig, axes = plt.subplots(n_rows_subplot, n_cols_subplot, figsize=(figsize_per_plot[0] * n_cols_subplot, figsize_per_plot[1] * n_rows_subplot))

            # Handle single subplot case
            if n_datasets == 1:
                axes = np.array([axes])
            axes = axes.flatten() if n_datasets > 1 else axes

            fig.suptitle('Correlation Matrices Comparison', fontsize=16, fontweight='bold', y=0.995)

            for idx, (name, data) in enumerate(datasets.items()):
                ax = axes[idx] if n_datasets > 1 else axes[0]

                # Compute correlation matrix
                corr_matrix = data[numerical_cols].corr()

                # Create heatmap
                im = ax.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)

                # Set ticks and labels
                ax.set_xticks(np.arange(len(numerical_cols)))
                ax.set_yticks(np.arange(len(numerical_cols)))
                ax.set_xticklabels(numerical_cols, rotation=45, ha='right', fontsize=9)
                ax.set_yticklabels(numerical_cols, fontsize=9)

                # Add colorbar for each subplot
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('Correlation', rotation=270, labelpad=15, fontsize=9)

                # Add correlation values as text
                for i in range(len(numerical_cols)):
                    for j in range(len(numerical_cols)):
                        value = corr_matrix.iloc[i, j]
                        # Determine text color based on background
                        text_color = 'white' if abs(value) > 0.5 else 'black'
                        text = ax.text(j, i, f'{value:.2f}', ha="center", va="center", color=text_color, fontsize=8)

                ax.set_title(f'{name}', fontsize=12, fontweight='bold', pad=10)
                ax.set_xlabel('Features', fontsize=10)
                ax.set_ylabel('Features', fontsize=10)

            # Hide empty subplots
            for idx in range(n_datasets, len(axes)):
                axes[idx].set_visible(False)

            plt.tight_layout()
            plt.show()

            return fig

        except Exception as e:
            print(f"Error in _plot_correlation_matrices:\n{str(e)}")
            return None

    def _plot_statistics_comparison(self, datasets: Dict[str, pd.DataFrame]):
        """
        Compare statistics for both numerical and categorical columns across all datasets.
        
        For numerical columns: mean, std, min, max, median
        For categorical columns: unique count, most frequent category, frequency of most common
        
        Args:
            datasets: Dictionary mapping dataset names to DataFrames
        
        Returns:
            list: List of figures (may contain None values)
        """
        try:
            # Get common columns across all datasets
            first_dataset = next(iter(datasets.values()))
            common_columns = set(first_dataset.columns)

            for data in datasets.values():
                common_columns = common_columns.intersection(data.columns)

            common_columns = list(common_columns)

            numerical_cols = []
            categorical_cols = []

            for col in common_columns:
                # Check if the column is numerical in all datasets
                if all(pd.api.types.is_numeric_dtype(data[col]) for data in datasets.values()):
                    numerical_cols.append(col)
                else:
                    categorical_cols.append(col)

            print(f"Numerical columns: {len(numerical_cols)}")
            print(f"Categorical columns: {len(categorical_cols)}")

            figures = []

            if numerical_cols:
                num_fig = self._compare_numerical_columns(datasets, numerical_cols)
                if num_fig is not None:
                    figures.append(num_fig)

            if categorical_cols:
                cat_fig = self._compare_categorical_columns(datasets, categorical_cols)
                if cat_fig is not None:
                    figures.append(cat_fig)

            summary = self._generate_statistics_summary(datasets, numerical_cols, categorical_cols)

            return figures if figures else None

        except Exception as e:
            print(f"Error in _plot_statistics_comparison:\n{str(e)}")
            return None

    def _compare_numerical_columns(self, datasets: Dict[str, pd.DataFrame], numerical_cols):
        """Compare numerical columns statistics across all datasets."""
        try:
            # Calculate statistics for all datasets
            all_stats = {}
            for name, data in datasets.items():
                all_stats[name] = pd.DataFrame({
                    'mean': data[numerical_cols].mean(),
                    'std': data[numerical_cols].std(),
                    'min': data[numerical_cols].min(),
                    'max': data[numerical_cols].max(),
                    'median': data[numerical_cols].median()
                })

            # Normalize statistics by dividing by maximum for each column-statistic pair across datasets
            normalized_stats = {}
            stats_to_plot = ['mean', 'std', 'min', 'max', 'median']

            # Initialize normalized_stats DataFrames
            for name in all_stats.keys():
                normalized_stats[name] = pd.DataFrame(index=all_stats[name].index)

            # For each statistic and each column, normalize by dividing by max across datasets
            for stat in stats_to_plot:
                for col in numerical_cols:
                    # Collect values for this specific column-statistic combination across all datasets
                    col_stat_values = []
                    for name, stats_df in all_stats.items():
                        col_stat_values.append(stats_df.loc[col, stat])

                    # Calculate maximum for this column-statistic across datasets
                    col_stat_max = np.max(col_stat_values)

                    # Normalize each dataset's value for this column-statistic
                    for name, stats_df in all_stats.items():
                        value = stats_df.loc[col, stat]
                        # Divide by maximum value
                        if col_stat_max != 0:
                            normalized_stats[name].loc[col, stat] = value / col_stat_max
                        else:
                            # If maximum is 0, set to 0
                            normalized_stats[name].loc[col, stat] = 0.0

            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Numerical Columns Statistics Comparison (Normalized by Max)', fontsize=16, fontweight='bold')

            positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]

            x_pos = np.arange(len(numerical_cols))
            bar_width = 0.8 / len(datasets)  # Adjust bar width based on number of datasets

            for i, stat in enumerate(stats_to_plot):
                row, col = positions[i]
                ax = axes[row, col]

                # Plot bars for each dataset using normalized values
                for idx, (name, stats_df) in enumerate(normalized_stats.items()):
                    offset = (idx - len(datasets)/2 + 0.5) * bar_width
                    ax.bar(x_pos + offset, stats_df[stat], bar_width, color=self.colors[idx], label=name, alpha=0.8)

                ax.set_title(f'{stat.capitalize()} (Normalized)', fontsize=12, fontweight='bold')
                ax.set_xlabel('Columns', fontsize=10)
                ax.set_ylabel('Value / Max', fontsize=10)
                ax.set_xticks(x_pos)
                ax.set_xticklabels(numerical_cols, rotation=45, ha='right')
                ax.set_ylim([0, 1.1])
                ax.legend()
                ax.grid(True, alpha=0.3)

            axes[1, 2].axis('off')

            plt.tight_layout()
            plt.show()
            return fig

        except Exception as e:
            print(f"Error in _compare_numerical_columns:\n{str(e)}")
            return None

    def _compare_categorical_columns(self, datasets: Dict[str, pd.DataFrame], categorical_cols: list):
        """Compare categorical columns statistics across all datasets."""
        # Calculate statistics for all datasets
        all_cat_stats = {}

        try:
            for name, data in datasets.items():
                cat_stats = []
                for col in categorical_cols:
                    unique_count = data[col].nunique()
                    mode_val = data[col].mode().iloc[0] if len(data[col].mode()) > 0 else 'N/A'
                    mode_freq = data[col].value_counts().iloc[0] if len(data[col].value_counts()) > 0 else 0
                    mode_pct = (mode_freq / len(data)) * 100 if len(data) > 0 else 0

                    cat_stats.append({
                        'column': col,
                        'unique_count': unique_count,
                        'most_frequent': mode_val,
                        'frequency': mode_freq,
                        'percentage': mode_pct
                    })

                all_cat_stats[name] = pd.DataFrame(cat_stats).set_index('column')

            # Normalize categorical statistics by dividing by maximum for each column-metric pair across datasets
            normalized_cat_stats = {}
            metrics_to_normalize = ['unique_count', 'frequency', 'percentage']

            # Initialize normalized_cat_stats DataFrames
            for name in all_cat_stats.keys():
                normalized_cat_stats[name] = pd.DataFrame(index=all_cat_stats[name].index)
                normalized_cat_stats[name]['most_frequent'] = all_cat_stats[name]['most_frequent']

            # For each metric and each column, normalize by dividing by max across datasets
            for metric in metrics_to_normalize:
                for col in categorical_cols:
                    # Collect values for this specific column-metric combination across all datasets
                    col_metric_values = []
                    for name, stats_df in all_cat_stats.items():
                        col_metric_values.append(stats_df.loc[col, metric])

                    # Calculate maximum for this column-metric across datasets
                    col_metric_max = np.max(col_metric_values)

                    # Normalize each dataset's value for this column-metric
                    for name, stats_df in all_cat_stats.items():
                        value = stats_df.loc[col, metric]
                        # Divide by maximum value
                        if col_metric_max != 0:
                            normalized_cat_stats[name].loc[col, metric] = value / col_metric_max
                        else:
                            # If maximum is 0, set to 0
                            normalized_cat_stats[name].loc[col, metric] = 0.0

            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle('Categorical Columns Statistics Comparison (Normalized by Max)', fontsize=16, fontweight='bold')

            x_pos = np.arange(len(categorical_cols))
            bar_width = 0.8 / len(datasets)  # Adjust bar width based on number of datasets

            # Plot 1: Unique Value Counts (Normalized)
            for idx, (name, stats_df) in enumerate(normalized_cat_stats.items()):
                offset = (idx - len(datasets)/2 + 0.5) * bar_width
                axes[0].bar(x_pos + offset, stats_df['unique_count'], bar_width, color=self.colors[idx], label=name, alpha=0.8)

            axes[0].set_title('Unique Value Counts (Normalized)', fontsize=12, fontweight='bold')
            axes[0].set_xlabel('Columns', fontsize=10)
            axes[0].set_ylabel('Value / Max', fontsize=10)
            axes[0].set_xticks(x_pos)
            axes[0].set_xticklabels(categorical_cols, rotation=45, ha='right')
            axes[0].set_ylim([0, 1.1])
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # Plot 2: Most Frequent Value Count (Normalized)
            for idx, (name, stats_df) in enumerate(normalized_cat_stats.items()):
                offset = (idx - len(datasets)/2 + 0.5) * bar_width
                axes[1].bar(x_pos + offset, stats_df['frequency'], bar_width, color=self.colors[idx], label=name, alpha=0.8)

            axes[1].set_title('Most Frequent Value Count (Normalized)', fontsize=12, fontweight='bold')
            axes[1].set_xlabel('Columns', fontsize=10)
            axes[1].set_ylabel('Value / Max', fontsize=10)
            axes[1].set_xticks(x_pos)
            axes[1].set_xticklabels(categorical_cols, rotation=45, ha='right')
            axes[1].set_ylim([0, 1.1])
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            # Plot 3: Most Frequent Value Percentage (Normalized)
            for idx, (name, stats_df) in enumerate(normalized_cat_stats.items()):
                offset = (idx - len(datasets)/2 + 0.5) * bar_width
                axes[2].bar(x_pos + offset, stats_df['percentage'], bar_width, color=self.colors[idx], label=name, alpha=0.8)

            axes[2].set_title('Most Frequent Value Percentage (Normalized)', fontsize=12, fontweight='bold')
            axes[2].set_xlabel('Columns', fontsize=10)
            axes[2].set_ylabel('Value / Max', fontsize=10)
            axes[2].set_xticks(x_pos)
            axes[2].set_xticklabels(categorical_cols, rotation=45, ha='right')
            axes[2].set_ylim([0, 1.1])
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()
            return fig

        except Exception as e:
            print(f"Error in _compare_categorical_columns:\n{str(e)}")
            return None

    def _generate_statistics_summary(self, datasets: Dict[str, pd.DataFrame], numerical_cols, categorical_cols):
        """Generate a summary table of statistics comparison across all datasets."""
        summary_data = []

        try:
            # Get the first dataset as reference (typically "Real" data)
            reference_name = list(datasets.keys())[0]
            reference_df = datasets[reference_name]

            # Numerical columns
            for col in numerical_cols:
                row = {
                    'Column': col,
                    'Type': 'Numerical',
                }

                # Add statistics for each dataset
                for name, data in datasets.items():
                    row[f'{name}_Mean'] = f"{data[col].mean():.4f}"
                    row[f'{name}_Std'] = f"{data[col].std():.4f}"

                # Add percentage differences relative to reference dataset
                for name, data in datasets.items():
                    if name != reference_name:
                        mean_diff = abs(data[col].mean() - reference_df[col].mean())
                        mean_pct = (mean_diff / reference_df[col].mean() * 100) if reference_df[col].mean() != 0 else 0
                        row[f'{name}_Mean_Diff%'] = f"{mean_pct:.2f}%"

                        std_diff = abs(data[col].std() - reference_df[col].std())
                        std_pct = (std_diff / reference_df[col].std() * 100) if reference_df[col].std() != 0 else 0
                        row[f'{name}_Std_Diff%'] = f"{std_pct:.2f}%"

                summary_data.append(row)

            # Categorical columns
            for col in categorical_cols:
                row = {
                    'Column': col,
                    'Type': 'Categorical',
                }

                # Add statistics for each dataset
                for name, data in datasets.items():
                    unique_count = data[col].nunique()
                    mode_val = data[col].mode().iloc[0] if len(data[col].mode()) > 0 else 'N/A'
                    row[f'{name}_Mean'] = f"Unique: {unique_count}"
                    row[f'{name}_Std'] = f"Mode: {mode_val}"

                # Add percentage differences relative to reference dataset
                ref_unique = reference_df[col].nunique()
                for name, data in datasets.items():
                    if name != reference_name:
                        unique_count = data[col].nunique()
                        unique_diff_pct = (abs(unique_count - ref_unique) / ref_unique * 100) if ref_unique != 0 else 0
                        row[f'{name}_Mean_Diff%'] = f"{unique_diff_pct:.2f}%"
                        row[f'{name}_Std_Diff%'] = "N/A"

                summary_data.append(row)

            summary_df = pd.DataFrame(summary_data)
            return summary_df

        except Exception as e:
            print(f"Error in _generate_statistics_summary:\n{str(e)}")
            return None