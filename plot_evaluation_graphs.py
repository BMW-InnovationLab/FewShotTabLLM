import warnings
import pandas as pd
from services.plot_data_statistics import DataVisualizer
from services.plot_evaluation_scores import EvaluationVisualizer
from services.parsers import plot_evaluation_parse_args
import os
warnings.filterwarnings('ignore')


def run(dataset: str, experiments: list[str], synthetic_datasets: list[str]):
    """Generate visualization plots for experiments.
    
    Args:
        dataset: Path to the real dataset CSV file
        experiments: List of experiment names to process
        synthetic_datasets: List of synthetic dataset names (without path or .csv extension)
                          e.g., ["ctgan", "tvae", "synthetic_llm_40_shots"]
    """
    dataset_name = os.path.splitext(os.path.basename(dataset))[0]

    dataset = pd.read_csv(dataset)
    for experiment in experiments:
        datasets = {}
        evaluation_reports = []
        
        length = len(dataset)
        dataset_sampled = dataset.sample(n=min(5000, length), random_state=42)
        dataset_sampled.to_csv(f"experiments/{dataset_name}/{experiment}/datasets/real/{dataset_name}_sampled.csv", index=False)
        datasets["real"] = dataset_sampled

        for synth_name in synthetic_datasets:
            # Try to load the synthetic dataset
            synth_path = f"experiments/{dataset_name}/{experiment}/datasets/synthetic/{synth_name}.csv"
            if os.path.exists(synth_path):
                datasets[synth_name] = pd.read_csv(synth_path)
                
                # Try to find corresponding evaluation report
                eval_report = f"experiments/{dataset_name}/{experiment}/evaluation_reports/{synth_name}.json"
                if os.path.exists(eval_report):
                    evaluation_reports.append(eval_report)
            else:
                print(f"Warning: Synthetic dataset not found: {synth_path}")
        try:
            DataVisualizer(
                datasets=datasets,
                output_dir=f"experiments/{dataset_name}/{experiment}/figures/data_statistics"
            ).visualize()
        except Exception as e:
            print(f"DataVisualizer error for experiment {experiment}:\n{e}")

        try:
            EvaluationVisualizer(
                evaluation_reports=evaluation_reports,
                output_dir=f"experiments/{dataset_name}/{experiment}/figures"
            ).visualize_evaluation_reports()
        except Exception as e:
            print(f"EvaluationVisualizer error for experiment {experiment}:\n{e}")




if __name__ == "__main__":
    args = plot_evaluation_parse_args()
    
    run(
        dataset=args.dataset,
        experiments=args.experiments,
        synthetic_datasets=args.synthetic_datasets
    )
