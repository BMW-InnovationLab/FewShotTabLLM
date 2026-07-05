"""
Prompt templates and few-shot retrieval for tabular data generation

Optimized for modern instruct models like Gemma 3 with proper formatting.
"""

from typing import Dict, Optional
import pandas as pd

from flash_tabgen_tensorrt.core.data_profiler import DatasetProfile
from flash_tabgen_tensorrt.core.encoding import BaseEncoder


class PromptBuilder:
    """
    Builds prompts with k-shot demonstrations

    Optimized for modern instruct models (Gemma 3, Llama 3.1) with
    proper formatting and structured instructions.
    """

    def __init__(
        self,
        profile: DatasetProfile,
        encoder: BaseEncoder,
        k_shots: int = 8,
        use_structured_format: bool = True,
        train_data: Optional[pd.DataFrame] = None,
    ):
        self.profile = profile
        self.encoder = encoder
        self.k_shots = k_shots
        self.use_structured_format = use_structured_format
        self.train_data = train_data

    def build_training_prompt(self, rows: pd.DataFrame, permute: bool = False) -> str:
        """Build training prompt with encoded rows"""
        encoded_rows = self.encoder.encode_batch(rows, permute=permute)
        return "\n".join([er.text for er in encoded_rows])

    def build_generation_prompt(
        self,
        demo_data: pd.DataFrame,
        n_samples: int = 1,
        conditional: Optional[Dict[str, str]] = None,
        mode: str = "flexible",
        permute: bool = False,
        use_correlation_matrix: bool = True,
    ) -> str:
        """
        Build generation prompt with k-shot demos and task instruction

        Optimized for Gemma 3 with clear structure and examples.
        """

        # Sample stratified demos
        demo_rows = self._stratified_sample(demo_data, self.k_shots)

        # Build structured prompt for modern models
        if self.use_structured_format:
            prompt = self._build_structured_prompt(
                demo_rows,
                n_samples,
                conditional,
                mode,
                permute=permute,
                use_correlation_matrix=use_correlation_matrix,
            )
        else:
            prompt = self._build_simple_prompt(demo_rows, n_samples, conditional, permute=permute)

        return prompt

    def _build_structured_prompt(
        self,
        demo_rows: pd.DataFrame,
        n_samples: int,
        conditional: Optional[Dict[str, str]],
        mode: str,
        permute: bool = False,
        use_correlation_matrix: bool = True,
    ) -> str:
        """Build structured prompt with statistical properties for Qwen3"""
        parts = []

        # Task description
        parts.append("## Task: Generate Synthetic Tabular Data")
        parts.append("")
        parts.append(
            "Generate realistic, diverse, and UNIQUE synthetic tabular data that matches the schema and statistical patterns below."
        )
        parts.append(
            "IMPORTANT: Generate rows that are different from the examples to avoid privacy leakage."
        )
        parts.append("")

        # Schema with statistics
        parts.append("## Schema and Statistics:")
        parts.append(self._build_schema_with_stats(use_correlation_matrix=use_correlation_matrix))
        parts.append("")

        # Examples section
        parts.append(f"## Examples ({len(demo_rows)} rows):")
        demo_encoded = self.encoder.encode_batch(demo_rows, permute=permute)
        for i, er in enumerate(demo_encoded, 1):
            parts.append(f"{i}. {er.text}")
        parts.append("")

        # Generation instructions with output format
        parts.append("## Instructions:")
        parts.append(
            f"Generate **only** {n_samples} NEW and UNIQUE synthetic row(s). **Do NOT generate any additional text, explanations, or repeats of the instructions after the {n_samples} rows.**"
        )
        parts.append("")

        # Detect encoder type from metadata
        is_json = hasattr(self.encoder, "build_json_schema")

        if is_json:
            parts.append("**Output Format (CRITICAL)**:")
            parts.append(f"Output {n_samples} rows, one per line, numbered 1 to {n_samples}.")
            parts.append("Each row must be a valid JSON object with ALL columns as keys.")
            # Build a compact example from the column names
            example_keys = ", ".join([f'"{col}": ...' for col in self.profile.column_order[:3]])
            parts.append(
                f'Example: 1. {{"{self.profile.column_order[0]}": 35, '
                f"{example_keys.split(', ', 1)[1] if ', ' in example_keys else ''}}}"
            )
            parts.append("")
        else:
            parts.append("**Output Format (CRITICAL)**:")
            parts.append(f"Output {n_samples} rows, one per line, numbered 1 to {n_samples}.")
            parts.append(
                "Format: NUMBER. column1 is value1, column2 is value2, ..., columnN is valueN"
            )
            parts.append("Example: 1. MedInc is 3.456, HouseAge is 25, ..., MedHouseVal is 2.100")
            parts.append("")

        if conditional:
            cond_str = ", ".join([f"{k}={v}" for k, v in conditional.items()])
            parts.append(f"**Constraint**: All rows must have {cond_str}")
            parts.append("")

        parts.append("**Requirements**:")
        parts.append("- Generate UNIQUE rows (different from examples)")
        if is_json:
            parts.append("- Each row MUST be a valid JSON object with ALL column keys")
            parts.append("- Use the EXACT column names as JSON keys")
            parts.append("- Use correct JSON types: integers as numbers, strings in double quotes")
        else:
            parts.append("- Follow the EXACT format: NUMBER. col1 is val1, col2 is val2, ...")
        parts.append("- Respect column types and statistical ranges")
        parts.append("- Maintain realistic correlations between features")
        parts.append("- Ensure diversity in generated samples")
        if not is_json:
            parts.append(
                "- Make sure NOT TO REPEAT the index twice in the same row, e.g., '1. 1. col1 is val1...' is incorrect"
            )
        parts.append(f"- Output exactly {n_samples} rows")
        parts.append(
            f"**- Once row {n_samples} is generated, STOP. No further text should be generated.**"
        )
        parts.append("")
        parts.append("## Generated Data:")

        return "\n".join(parts)

    def _build_schema_with_stats(self, use_correlation_matrix: bool = True) -> str:
        """Build schema with statistical properties (like GReaT/TabuLa)"""
        lines = []

        # Column statistics
        for col_name in self.profile.column_order:
            col_prof = self.profile.columns[col_name]

            # Column type and range
            if col_prof.dtype in ["integer", "float"]:
                range_str = f"min={col_prof.min_val:.2f}, max={col_prof.max_val:.2f}"
                mean_str = f"mean={col_prof.mean:.2f}"
                std_str = f"std={col_prof.std:.2f}"

                # Add quantiles if available
                quantiles_str = ""
                if col_prof.quantiles:
                    q25 = col_prof.quantiles.get("Q25", col_prof.quantiles.get("0.25"))
                    q50 = col_prof.quantiles.get("Q50", col_prof.quantiles.get("0.5"))
                    q75 = col_prof.quantiles.get("Q75", col_prof.quantiles.get("0.75"))

                    if all([q25, q50, q75]):
                        quantiles_str = f", q25={q25:.2f}, q50={q50:.2f}, q75={q75:.2f}"

                lines.append(
                    f"  {col_name}: {col_prof.dtype}, {range_str}, {mean_str}, {std_str}{quantiles_str}"
                )
            elif col_prof.dtype == "categorical":
                if col_prof.cardinality and col_prof.cardinality > 20:
                    # High-cardinality categorical: show cardinality + random sample
                    # to avoid collapsing the model onto a small set of listed values
                    sample_vals = (
                        [v for v, c in col_prof.top_k_values[:3]]
                        if col_prof.top_k_values
                        else []
                    )
                    sample_str = ", ".join(sample_vals)
                    lines.append(
                        f"  {col_name}: {col_prof.dtype}, {col_prof.cardinality} unique values"
                        f" (examples: {sample_str})"
                    )
                else:
                    # Low-cardinality categorical: safe to list values
                    top_vals = (
                        [v for v, c in col_prof.top_k_values[:5]]
                        if col_prof.top_k_values
                        else []
                    )
                    vals_str = ", ".join(top_vals)
                    lines.append(f"  {col_name}: {col_prof.dtype} (values: {vals_str})")
            elif col_prof.dtype == "id":
                # ID / unique identifier: show format hint, NOT enumerable values
                sample_vals = (
                    [v for v, _ in col_prof.top_k_values[:3]]
                    if col_prof.top_k_values
                    else []
                )
                fmt_str = ", ".join(sample_vals)
                lines.append(
                    f"  {col_name}: unique identifier"
                    f" (format examples: {fmt_str})"
                    f" — generate novel unique values, do NOT reuse examples"
                )
            else:
                lines.append(f"  {col_name}: {col_prof.dtype}")

        # Add correlation matrix for numerical columns
        if self.train_data is not None and use_correlation_matrix:
            corr_str = self._build_correlation_matrix()
            if corr_str:
                lines.append("")
                lines.append("Correlation Matrix (numerical features):")
                lines.append(corr_str)

        return "\n".join(lines)

    def _build_correlation_matrix(self) -> str:
        """Build a formatted correlation matrix for numerical columns"""
        if self.train_data is None:
            return ""

        # Get numerical columns
        numerical_cols = [
            col
            for col in self.profile.column_order
            if self.profile.columns[col].dtype in ["integer", "float"]
        ]

        if len(numerical_cols) < 2:
            return ""

        # Compute correlation matrix
        try:
            corr_matrix = self.train_data[numerical_cols].corr()

            # Format as a readable string with full column names
            lines = []

            # Determine column width (max of all column names, min 8, max 20)
            max_col_len = max(len(col) for col in numerical_cols)
            col_width = min(max(max_col_len, 8), 20)

            # Header row with full column names
            header = " " * (col_width + 2) + " ".join(
                [f"{col[:col_width]:>{col_width}}" for col in numerical_cols]
            )
            lines.append(header)

            # Data rows with full column names
            for i, row_col in enumerate(numerical_cols):
                row_values = []
                for j, col_col in enumerate(numerical_cols):
                    corr_val = corr_matrix.iloc[i, j]
                    row_values.append(f"{corr_val:>{col_width}.2f}")

                # Use full column name in row label
                row_name = row_col[:col_width].ljust(col_width)
                row_str = f"  {row_name} " + " ".join(row_values)
                lines.append(row_str)

            return "\n".join(lines)
        except Exception as e:
            # If correlation computation fails, return empty string
            return ""

    def _build_simple_prompt(
        self,
        demo_rows: pd.DataFrame,
        n_samples: int,
        conditional: Optional[Dict[str, str]],
        permute: bool = False,
    ) -> str:
        """Simple prompt for non-instruct models"""
        # Schema header
        schema_msg = self._build_schema_header() + "\n\n"

        # Demonstrations
        demo_encoded = self.encoder.encode_batch(demo_rows, permute=permute)
        demo_msg = "Examples:\n" + "\n".join([er.text for er in demo_encoded]) + "\n\n"

        # Task instruction
        task_msg = f"Generate {n_samples} new rows"
        if conditional:
            cond_str = ", ".join([f"{k}={v}" for k, v in conditional.items()])
            task_msg += f" where {cond_str}"
        task_msg += ":\n"

        return schema_msg + demo_msg + task_msg

    def _build_schema_header(self) -> str:
        """Build schema header with column names and types"""
        lines = ["Schema:"]
        for col_name in self.profile.column_order:
            col_prof = self.profile.columns[col_name]
            lines.append(f"  {col_name}: {col_prof.dtype}")
        return "\n".join(lines)

    def _stratified_sample(self, data: pd.DataFrame, n: int) -> pd.DataFrame:
        """Stratified sampling across categorical values and quantiles"""
        if len(data) <= n:
            return data

        # Simple random sampling for now
        # TODO: Implement proper stratified sampling
        return data.sample(n=n)
