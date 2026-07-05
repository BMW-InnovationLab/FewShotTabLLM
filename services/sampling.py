import time

import pandas as pd
import numpy as np
import os
import argparse
from pathlib import Path
from typing import Optional, List
import matplotlib.pyplot as plt
import seaborn as sns


def select_representative_samples(
        input_file: str,
        output_file: Optional[str] = None,
        sample_size: Optional[int] = None,
        method: str = 'flexible',
        feature_threshold: float = 0.8,
        plot_features: int = 20,
        selected_features: Optional[List[str]] = None,
        lower_percentile: float = 0.25,
        higher_percentile: float = 0.75,
        enable_visualization: bool = True

) -> pd.DataFrame:
    """
    Select representative samples from dataset by filtering data between 25th and 75th percentiles
    for each feature to ensure the sample is suitable for calibration purposes.
    Optionally generates box plots before and after sampling for visualization.

    Parameters:
    -----------
    input_file : str
        Path to the input CSV file
    output_file : str, optional
        Path to save the representative sample. If None, will use input_file with '_representative' suffix
    sample_size : int, optional
        Number of samples to select from the filtered data. If None, uses all filtered data
    method : str
        'strict': All features must be within 25th-75th percentile
        'flexible': At least feature_threshold percentage of features must be within range
        'individual': Select samples where each feature is within range (more samples)
    feature_threshold : float
        For flexible method: percentage of features that must be within range (0.0 to 1.0)
    plot_features : int
        Number of numeric features to plot in the boxplot (to avoid overcrowding)
    selected_features : List[str], optional
        List of specific features to include in filtering. If None, uses all numeric features.
    enable_visualization : bool
        If True, generates boxplot visualizations. If False, skips visualization for faster execution.

    Returns:
    --------
    pandas.DataFrame
        Representative sample dataset
    """

    # Load the dataset
    print(f"Loading dataset from {input_file}...")
    df: pd.DataFrame = pd.read_csv(input_file, low_memory=False)

    print(f"Original dataset shape: {df.shape}")
    time.sleep(1)
    # Get numeric columns only (exclude any non-numeric columns)
    all_numeric_columns = df.select_dtypes(include=[np.number]).columns
    print(f"Total number of numeric features: {len(all_numeric_columns)}")
    time.sleep(1)

    # Use selected features if provided, otherwise use all numeric features
    if selected_features is not None:
        # Validate that selected features exist in the dataset
        available_features = set(all_numeric_columns)
        valid_features = [f for f in selected_features if f in available_features]
        invalid_features = [f for f in selected_features if f not in available_features]

        if invalid_features:
            print(f"Warning: The following features were not found in the dataset: {invalid_features}")

        if not valid_features:
            print("Error: No valid features selected. Using all numeric features instead.")
            numeric_columns = all_numeric_columns
        else:
            numeric_columns = valid_features
            print(f"Using {len(valid_features)} selected features for filtering.")
    else:
        numeric_columns = all_numeric_columns

    print(f"Number of features used for filtering: {len(numeric_columns)}")
    print("=" * 50)

    # Boxplot before sampling (only if visualization is enabled)
    if enable_visualization:
        time.sleep(1)
        # Select a subset of features to plot (to avoid overcrowding)
        plot_cols = list(numeric_columns[:plot_features])
        before_plot_path = str(Path(input_file).parent / 'boxplot_before.png')
        with plt.ioff():
            n_cols = int(np.ceil(np.sqrt(len(plot_cols))))
            n_rows = int(np.ceil(len(plot_cols) / n_cols))
            fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(max(8, 3 * n_cols), 3 * n_rows),
                                     squeeze=False)
            for i, col in enumerate(plot_cols):
                row, col_idx = divmod(i, n_cols)
                ax = axes[row, col_idx]
                sns.boxplot(data=df[[col]], orient='v', fliersize=1, ax=ax)
                ax.set_title(f'{col}')
                ax.set_xlabel('')
                ax.set_ylabel('')
            fig.suptitle('Boxplots of Numeric Features (Before Sampling)')
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            # plt.savefig(before_plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Boxplot before sampling saved to: {before_plot_path}")

    # Calculate 25th and 75th percentiles for each numeric feature
    print("Calculating percentiles for each feature...")
    percentile_lower: pd.Series = df[numeric_columns].quantile(lower_percentile)
    percentile_higher: pd.Series = df[numeric_columns].quantile(higher_percentile)

    if method == 'strict':
        print("Filtering data between 25th and 75th percentiles (strict method)...")
        mask: pd.Series = pd.Series([True] * len(df), index=df.index)
        for col in numeric_columns:
            col_mask = (df[col] >= percentile_lower[col]) & (df[col] <= percentile_higher[col])
            mask = mask & col_mask
    elif method == 'flexible':
        print(f"Filtering data with flexible method (threshold: {feature_threshold:.1%})...")
        feature_counts = pd.Series(0, index=df.index)
        for col in numeric_columns:
            col_mask = (df[col] >= percentile_lower[col]) & (df[col] <= percentile_higher[col])
            feature_counts += col_mask.astype(int)
        min_features_required = int(len(numeric_columns) * feature_threshold)
        mask = feature_counts >= min_features_required
        print(
            f"Features within range per sample - Min: {feature_counts.min()}, Max: {feature_counts.max()}, Required: {min_features_required}")
    elif method == 'individual':
        print("Filtering data with individual method (any feature within range)...")
        mask: pd.Series = pd.Series([False] * len(df), index=df.index)
        for col in numeric_columns:
            col_mask = (df[col] >= percentile_lower[col]) & (df[col] <= percentile_higher[col])
            mask = mask | col_mask
    else:
        raise ValueError("Method must be 'strict', 'flexible', or 'individual'")

    # Apply the filter
    representative_df: pd.DataFrame = df[mask].copy()
    print(f"Filtered dataset shape: {representative_df.shape}")
    print(f"Percentage of data retained: {(len(representative_df) / len(df)) * 100:.2f}%")
    print("=" * 50)

    # If no data remains, try with a more lenient approach
    if len(representative_df) == 0:
        print("No data found with current method. Retrying with a less strict approach...")
        # Retry with flexible method if strict failed
        if method == 'strict':
            return select_representative_samples(input_file, output_file, sample_size, 'flexible', 0.5, plot_features,
                                                 selected_features, lower_percentile, higher_percentile,
                                                 enable_visualization)
        elif method == 'flexible':
            return select_representative_samples(input_file, output_file, sample_size, 'individual', feature_threshold,
                                                 plot_features, selected_features, lower_percentile, higher_percentile,
                                                 enable_visualization)
        else:
            print("Warning: No representative samples found even with individual method.")

    # If sample_size is specified, randomly sample from the filtered data
    if sample_size is not None and sample_size < len(representative_df):
        print(f"Randomly sampling {sample_size} records from filtered data...")
        representative_df = representative_df.sample(n=sample_size)
        print(f"Final representative sample shape: {representative_df.shape}")

    # Save the representative sample
    if output_file is None:
        input_path = Path(input_file)
        output_file = str(input_path.parent / f"{input_path.stem}_representative{input_path.suffix}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    representative_df.to_csv(output_file, index=False)
    print(f"Representative sample saved to: {output_file}")

    # Boxplot after sampling (only if visualization is enabled)
    if enable_visualization:
        if len(representative_df) > 0:
            plot_cols = list(numeric_columns[:plot_features])
            after_plot_path = str(Path(input_file).parent / 'boxplot_after.png')
            with plt.ioff():
                n_cols = int(np.ceil(np.sqrt(len(plot_cols))))
                n_rows = int(np.ceil(len(plot_cols) / n_cols))
                fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(max(8, 3 * n_cols), 3 * n_rows),
                                         squeeze=False)
                for i, col in enumerate(plot_cols):
                    row, col_idx = divmod(i, n_cols)
                    ax = axes[row, col_idx]
                    sns.boxplot(data=representative_df[[col]], orient='v', fliersize=1, ax=ax)
                    ax.set_title(f'{col}')
                    ax.set_xlabel('')
                    ax.set_ylabel('')
                fig.suptitle('Boxplots of Numeric Features (After Sampling)')
                plt.tight_layout(rect=[0, 0, 1, 0.95])
                # plt.savefig(after_plot_path, dpi=800, bbox_inches='tight')
                plt.close(fig)
                print(f"Boxplot after sampling saved to: {after_plot_path}")
        else:
            print("No data to plot after sampling.")

    # Print summary statistics
    print("\nSummary Statistics:")
    print("=" * 50)
    
    if not numeric_columns.empty:
        print("Original Dataset:")
        print(df[numeric_columns].describe())
        print("\nRepresentative Sample:")
        print(representative_df[numeric_columns].describe())
    else:
        print("No numeric columns found for summary statistics.")

    return representative_df


def interactive_mode():
    """Interactive mode for selecting parameters"""
    print("=== Interactive Representative Sample Selection ===")

    # Get input file
    while True:
        input_file = input("Enter the path to your CSV file (or press Enter for 'data/train/1.csv'): ").strip()
        if not input_file:
            input_file = 'data/train/1.csv'

        if os.path.exists(input_file):
            break
        else:
            print(f"File not found: {input_file}")

    # Load dataset to show available features
    print(f"\nLoading {input_file} to show available features...")
    df = pd.read_csv(input_file, low_memory=False)
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    print(f"\nFound {len(numeric_columns)} numeric features:")

    # Feature selection with checklist
    print("\nFeature Selection:")
    print("1. Use all numeric features")
    print("2. Select specific features from checklist")
    print("3. Include features from a list file (one feature per line)")
    feature_choice = input("Choose option (1 or 2 or 3): ").strip()

    selected_features = None
    if feature_choice == "2":
        print("\n=== Feature Checklist ===")
        print("Enter the numbers of features you want to include (comma-separated)")
        print("Example: 1,3,5,7 or 1-5 for range or 'all' for all features")
        print("Press Enter when done\n")

        # Show all features with numbers
        for i, col in enumerate(numeric_columns):
            print(f"{i + 1:3d}. {col}")

        while True:
            selection_input = input("\nEnter feature numbers (comma-separated) or dashed example 1-5: ").strip()
            if not selection_input:
                print("No features selected, using all numeric features.")
                break

            if selection_input.lower() == 'all':
                selected_features = numeric_columns
                break

            try:
                # Parse the input
                selected_indices = []
                parts = [part.strip() for part in selection_input.split(',')]

                for part in parts:
                    if '-' in part:
                        # Handle range (e.g., "1-5")
                        start, end = map(int, part.split('-'))
                        selected_indices.extend(range(start, end + 1))
                    else:
                        # Handle single number
                        selected_indices.append(int(part))

                # Convert to 0-based indexing and validate
                valid_indices = []
                for idx in selected_indices:
                    if 1 <= idx <= len(numeric_columns):
                        valid_indices.append(idx - 1)
                    else:
                        print(f"Warning: Feature number {idx} is out of range (1-{len(numeric_columns)})")

                if valid_indices:
                    selected_features = [numeric_columns[i] for i in valid_indices]
                    print(f"\nSelected {len(selected_features)} features:")
                    for i, feature in enumerate(selected_features):
                        print(f"  {i + 1}. {feature}")
                    break
                else:
                    print("No valid features selected. Please try again.")

            except ValueError:
                print("Invalid input. Please enter numbers separated by commas (e.g., 1,3,5 or 1-5)")

        if not selected_features:
            print("No features selected, using all numeric features.")
            selected_features = None
    if feature_choice == "3":
        feature_list_file = input("Enter the path to the feature list file (one feature per line): ").strip()
        if os.path.exists(feature_list_file):
            with open(feature_list_file) as f:
                selected_features = [line.strip() for line in f if line.strip()]
            print(f"\nSelected {len(selected_features)} features:")
            for feature in selected_features:
                print(f" {feature}")
        else:
            print(f"File not found: {feature_list_file}. Using all numeric features.")
            selected_features = None

    # Method selection
    print("\nFiltering Method:")
    print("1. Strict - All features must be within 25th-75th percentile")
    print(
        "2. Flexible - At least X % of features must be within range; ex: if X=0.7, at least 70% of features must be within range of 25th-75th percentile")
    print("3. Individual - Any feature within range")

    method_choice = input("Choose method (1, 2, or 3): ").strip()
    method_map = {"1": "strict", "2": "flexible", "3": "individual"}
    method = method_map.get(method_choice, "flexible")

    print("\nLower Percentile Threshold:")
    lower_percentile = input("Enter lower percentile (default 0.25): ").strip()
    if not lower_percentile:
        lower_percentile = 0.25
    else:
        try:
            lower_percentile = float(lower_percentile)
            if not (0 <= lower_percentile <= 1):
                raise ValueError("Percentile must be between 0 and 1")
        except ValueError:
            print("Invalid input, using default 0.25")
            lower_percentile = 0.25
    print("\nHigher Percentile Threshold:")
    higher_percentile = input("Enter higher percentile (default 0.75): ").strip()

    if not higher_percentile:
        higher_percentile = 0.75
    else:
        try:
            higher_percentile = float(higher_percentile)
            if not (0 <= higher_percentile <= 1):
                raise ValueError("Percentile must be between 0 and 1")
        except ValueError:
            print("Invalid input, using default 0.25")
            higher_percentile = 0.25

    feature_threshold = 0.7
    if method == "flexible":
        while True:
            try:
                threshold_input = input("Enter feature threshold (0.0-1.0, default 0.7): ").strip()
                if not threshold_input:
                    break
                feature_threshold = float(threshold_input)
                if 0 <= feature_threshold <= 1:
                    break
                else:
                    print("Threshold must be between 0.0 and 1.0")
            except ValueError:
                print("Please enter a valid number")

    # Sample size
    sample_size = None
    sample_choice = input("\nDo you want to limit the sample size? (y/n, default n): ").strip().lower()
    if sample_choice in ['y', 'yes']:
        while True:
            try:
                size_input = input("Enter sample size: ").strip()
                sample_size = int(size_input)
                if sample_size > 0:
                    break
                else:
                    print("Sample size must be positive")
            except ValueError:
                print("Please enter a valid integer")

    # Plot features
    while True:
        try:
            plot_input = input(f"\nNumber of features to plot (1-{len(selected_features)}, default 20): ").strip()
            if not plot_input:
                plot_features = 20
                break
            plot_features = int(plot_input)
            if 1 <= plot_features <= len(selected_features):
                break
            else:
                print(f"Number must be between 1 and {len(selected_features)}")
        except ValueError:
            print("Please enter a valid integer")

    # Output file
    output_file = input("\nOutput file path (press Enter for default): ").strip()
    if not output_file:
        output_file = None

    # Execute
    print("\n" + "=" * 50)
    print("Executing with parameters:")
    print(f"Input file: {input_file}")
    print(f"Method: {method}")
    if method == "flexible":
        print(f"Feature threshold: {feature_threshold:.1%}")
    print(f"Sample size: {sample_size if sample_size else 'All filtered data'}")
    print(f"Plot features: {plot_features}")
    print(f"Selected features: {len(selected_features) if selected_features else 'All numeric'}")
    print("=" * 50)
    time.sleep(2)

    return select_representative_samples(
        input_file=input_file,
        output_file=output_file,
        sample_size=sample_size,
        method=method,
        feature_threshold=feature_threshold,
        plot_features=plot_features,
        selected_features=selected_features,
        lower_percentile=lower_percentile,
        higher_percentile=higher_percentile
    )


def process_all_training_files(input_dir: str = 'data/', output_dir: str = 'data/representative_samples') -> None:
    """
    Process all CSV files in the training directory to create representative samples.

    Parameters:
    -----------
    input_dir : str
        Directory containing training CSV files
    output_dir : str
        Directory to save representative samples
    """

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get all CSV files in the input directory
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]

    print(f"Found {len(csv_files)} CSV files to process...")

    for csv_file in csv_files:
        input_path = os.path.join(input_dir, csv_file)
        output_path = os.path.join(output_dir, f"{os.path.splitext(csv_file)[0]}_representative.csv")

        print(f"\nProcessing {csv_file}...")
        try:
            representative_df = select_representative_samples(input_path, output_path, method='flexible',
                                                              feature_threshold=0.7)
            print(f"Successfully processed {csv_file}")
        except Exception as e:
            print(f"Error processing {csv_file}: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description='Select representative samples from dataset for calibration purposes')
    parser.add_argument('input_file', nargs='?', help='Input CSV file path')
    parser.add_argument('--output', '-o', help='Output file path')
    parser.add_argument('--method', '-m', choices=['strict', 'flexible', 'individual'], default='flexible',
                        help='Filtering method')
    parser.add_argument('--threshold', '-t', type=float, default=0.7,
                        help='Feature threshold for flexible method (0.0-1.0)')
    parser.add_argument('--sample-size', '-s', type=int, help='Number of samples to select')
    parser.add_argument('--plot-features', '-p', type=int, default=20, help='Number of features to plot')
    parser.add_argument('--features', '-f', nargs='+', help='Specific features to include')
    parser.add_argument('--interactive', '-i', action='store_true', help='Run in interactive mode')
    parser.add_argument('--batch', '-b', action='store_true', help='Process all CSV files in data/ directory')
    parser.add_argument('--features_list', '-fl', action='store_true',
                        help='Provide a list file of the features to include; format <name>.txt with one feature per line')
    parser.add_argument('--lower_percentile', '-lp', help="Select the lower percentile for the sampling", type=float,
                        default=0.25)
    parser.add_argument('--higher_percentile', '-hp', help="Select the upper percentile for the sampling", type=float,
                        default=0.75)

    args = parser.parse_args()

    if args.features_list:
        with open(args.features_list) as f:
            args.features = [line.strip() for line in f if line.strip()]
            print(args.features)

    if args.interactive:
        interactive_mode()
    elif args.batch:
        process_all_training_files()
    elif args.input_file:
        select_representative_samples(
            input_file=args.input_file,
            output_file=args.output,
            sample_size=args.sample_size,
            method=args.method,
            feature_threshold=args.threshold,
            plot_features=args.plot_features,
            selected_features=args.features,
            lower_percentile=args.lower_percentile,
            higher_percentile=args.higher_percentile
        )
    else:
        print("No input file specified. Use --interactive for interactive mode or provide an input file.")
        parser.print_help()


if __name__ == "__main__":
    main()
