"""
Encoding strategies for tabular data to text sequences

Implements three encoding modes:
1. GReaT-style: "X is Y" format with feature permutations for arbitrary conditioning
2. TabuLa-style: Compressed "X Y" format with single-token abbreviations and Middle Padding
3. Pred-LLM-style: Target-last ordering with feature-conditional sampling

References:
- GReaT: Fig. 2-3, §3.1-3.2 (permutations for conditioning)
- TabuLa: Fig. 1-2, §3.3-3.4 (compression and Middle Padding)
- Pred-LLM: Fig. 3-4, Algorithm 1 (target-last and feature-conditional)
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any
import json
import random
from dataclasses import dataclass, field

import pandas as pd
from flash_tabgen_tensorrt.core.data_profiler import DatasetProfile, ColumnProfile


_TRUTHY_STRINGS = frozenset({"true", "yes", "y", "1", "t"})
_FALSY_STRINGS = frozenset({"false", "no", "n", "0", "f"})


def _str_to_bool(value: Any) -> bool:
    """Convert a value to bool, correctly handling string representations.

    Unlike Python's built-in ``bool()``, this recognises common boolean-like
    strings such as "yes"/"no", "y"/"n", "t"/"f", "1"/"0" and maps them to
    ``True``/``False`` based on *meaning*, not truthiness.

    Args:
        value: The value to convert.  If already a native ``bool`` or numeric
            type it is passed through ``bool()`` directly.  Strings are matched
            case-insensitively against known truthy/falsy sets.

    Returns:
        The boolean interpretation of *value*.

    Raises:
        ValueError: If *value* is a string that does not match any known
            boolean representation.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lower = value.strip().lower()
        if lower in _TRUTHY_STRINGS:
            return True
        if lower in _FALSY_STRINGS:
            return False
        raise ValueError(f"Cannot convert {value!r} to bool")
    return bool(value)


@dataclass
class EncodedRow:
    """Encoded row representation"""

    text: str
    column_order: List[str]
    values: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseEncoder(ABC):
    """Base class for encoders"""

    def __init__(self, profile: DatasetProfile, float_precision: int = 3):
        self.profile = profile
        self.float_precision = float_precision

    @abstractmethod
    def encode_row(
        self, row: pd.Series, permute: bool = False, target_last: bool = False
    ) -> EncodedRow:
        """Encode a single row to text"""
        pass

    @abstractmethod
    def encode_batch(
        self, df: pd.DataFrame, permute: bool = False, target_last: bool = False
    ) -> List[EncodedRow]:
        """Encode a batch of rows"""
        pass

    def _format_value(self, value: Any, column_profile: ColumnProfile) -> str:
        """Format a value based on column type"""
        if pd.isna(value):
            return "Missing"

        dtype = column_profile.dtype

        if dtype == "categorical":
            return str(value)
        elif dtype == "integer":
            return str(int(value))
        elif dtype == "float":
            return f"{float(value):.{self.float_precision}f}"
        elif dtype == "boolean":
            return str(_str_to_bool(value))
        elif dtype == "datetime":
            return str(value)
        elif dtype == "text":
            val_str = str(value).replace('"', '""')
            if "," in val_str:
                return f'"{val_str}"'
            return val_str
        else:
            return str(value)


class GReaTEncoder(BaseEncoder):
    """GReaT-style encoder: 'X is Y' format with feature permutations"""

    def __init__(
        self,
        profile: DatasetProfile,
        float_precision: int = 3,
        separator: str = ", ",
        terminator: str = ".",
    ):
        super().__init__(profile, float_precision)
        self.separator = separator
        self.terminator = terminator

    def encode_row(
        self, row: pd.Series, permute: bool = False, target_last: bool = False
    ) -> EncodedRow:
        """Encode row in GReaT format"""
        columns = list(row.index)

        if target_last and self.profile.target_column:
            features = [c for c in columns if c != self.profile.target_column]
            target = self.profile.target_column
            if permute:
                random.shuffle(features)
            column_order = features + [target]
        elif permute:
            column_order = columns.copy()
            random.shuffle(column_order)
        else:
            column_order = columns

        clauses = []
        for col in column_order:
            col_profile = self.profile.columns[col]
            formatted_value = self._format_value(row[col], col_profile)
            clauses.append(f"{col} is {formatted_value}")

        text = self.separator.join(clauses) + self.terminator

        return EncodedRow(
            text=text,
            column_order=column_order,
            values=row.to_dict(),
            metadata={"encoder": "great", "permute": permute, "target_last": target_last},
        )

    def encode_batch(
        self, df: pd.DataFrame, permute: bool = False, target_last: bool = False
    ) -> List[EncodedRow]:
        """Encode batch of rows"""
        return [
            self.encode_row(row, permute=permute, target_last=target_last)
            for _, row in df.iterrows()
        ]


class TabulaEncoder(BaseEncoder):
    """
    TabuLa-style encoder: Compressed format with single-token abbreviations

    Reference: TabuLa Fig. 1-2, §3.3-3.4
    """

    def __init__(
        self,
        profile: DatasetProfile,
        float_precision: int = 3,
        use_abbreviations: bool = True,
        separator: str = ", ",
    ):
        super().__init__(profile, float_precision)
        self.use_abbreviations = use_abbreviations
        self.separator = separator

        # Generate column abbreviations
        self.column_abbrevs = self._generate_column_abbreviations()

    def _generate_column_abbreviations(self) -> Dict[str, str]:
        """Generate unique abbreviations for column names"""
        abbrevs = {}
        used = set()

        for col in self.profile.column_order:
            # Try first letter
            abbrev = col[0].upper()
            if abbrev in used:
                # Try first two letters
                abbrev = col[:2].upper()
                if abbrev in used:
                    # Add number
                    i = 0
                    while f"{abbrev}{i}" in used:
                        i += 1
                    abbrev = f"{abbrev}{i}"

            abbrevs[col] = abbrev
            used.add(abbrev)

        return abbrevs

    def encode_row(
        self, row: pd.Series, permute: bool = False, target_last: bool = False
    ) -> EncodedRow:
        """Encode row in TabuLa compressed format: 'A 39, E BSc, O Adm'"""
        # TabuLa uses fixed column order for speed
        column_order = self.profile.column_order

        # Build compressed sequence
        parts = []
        for col in column_order:
            col_profile = self.profile.columns[col]

            # Column abbreviation
            col_abbrev = self.column_abbrevs[col] if self.use_abbreviations else col

            # Value formatting with abbreviation
            value = row[col]
            if pd.isna(value):
                formatted_value = "M"  # "Missing" abbreviated
            elif (
                col_profile.dtype == "categorical"
                and self.use_abbreviations
                and col_profile.abbreviation_map
            ):
                # Use abbreviation map from profiler
                formatted_value = col_profile.abbreviation_map.get(str(value), str(value)[:3])
            else:
                formatted_value = self._format_value(value, col_profile)

            # Compressed format: "A 39" instead of "Age is 39"
            parts.append(f"{col_abbrev} {formatted_value}")

        text = self.separator.join(parts)

        return EncodedRow(
            text=text,
            column_order=column_order,
            values=row.to_dict(),
            metadata={
                "encoder": "tabula",
                "column_abbrevs": self.column_abbrevs,
            },
        )

    def encode_batch(
        self, df: pd.DataFrame, permute: bool = False, target_last: bool = False
    ) -> List[EncodedRow]:
        """Encode batch of rows"""
        return [self.encode_row(row) for _, row in df.iterrows()]


class PredLLMEncoder(BaseEncoder):
    """
    Pred-LLM-style encoder: Target-last ordering with feature-conditional sampling

    Reference: Pred-LLM Fig. 3-4, Algorithm 1, §III.B
    """

    def __init__(
        self,
        profile: DatasetProfile,
        float_precision: int = 3,
        separator: str = ", ",
        terminator: str = "",
    ):
        super().__init__(profile, float_precision)
        self.separator = separator
        self.terminator = terminator

        if not profile.target_column:
            raise ValueError("Pred-LLM encoder requires target_column in profile")

    def encode_row(
        self,
        row: pd.Series,
        permute: bool = False,
        target_last: bool = True,  # Always True for Pred-LLM
        include_target: bool = True,
    ) -> EncodedRow:
        """
        Encode row in Pred-LLM format: features first, target last

        Args:
            row: Input row
            permute: Whether to permute FEATURES only (not target)
            target_last: Always True (target is always last)
            include_target: Whether to include target (False for label query)
        """
        # Separate features and target
        features = [c for c in row.index if c != self.profile.target_column]
        target = self.profile.target_column

        # Permute features if requested (Pred-LLM §III.B step c)
        if permute:
            random.shuffle(features)

        # Build feature clauses
        clauses = []
        for col in features:
            col_profile = self.profile.columns[col]
            formatted_value = self._format_value(row[col], col_profile)
            clauses.append(f"{col} is {formatted_value}")

        # Add target if requested
        column_order = features.copy()
        if include_target:
            col_profile = self.profile.columns[target]
            formatted_value = self._format_value(row[target], col_profile)
            clauses.append(f"{target} is {formatted_value}")
            column_order.append(target)
        else:
            # For label query: "..., Income is" (incomplete)
            clauses.append(f"{target} is")
            column_order.append(target)

        text = self.separator.join(clauses) + self.terminator

        return EncodedRow(
            text=text,
            column_order=column_order,
            values=row.to_dict(),
            metadata={
                "encoder": "pred_llm",
                "permute": permute,
                "include_target": include_target,
            },
        )

    def encode_batch(
        self,
        df: pd.DataFrame,
        permute: bool = False,
        target_last: bool = True,
        include_target: bool = True,
    ) -> List[EncodedRow]:
        """Encode batch of rows"""
        return [
            self.encode_row(row, permute=permute, include_target=include_target)
            for _, row in df.iterrows()
        ]

    def encode_for_label_query(self, row: pd.Series, permute: bool = False) -> EncodedRow:
        """
        Encode features for label query (Pred-LLM §III.B.3)

        Returns: "Age is 39, Education is Bachelors, ..., Income is"
        """
        return self.encode_row(row, permute=permute, include_target=False)


class JsonEncoder(BaseEncoder):
    """
    JSON-based encoder: each row becomes a JSON object.

    Produces output like: ``{"age": 35, "workclass": "Private", "income": "<=50K"}``

    Benefits over GReaT format:
    - No ambiguity from commas or "is" in values
    - Parseable with ``json.loads`` — zero parsing failures
    - Compatible with vLLM guided JSON decoding for 100% valid output
    - LLMs are extensively trained on JSON

    The JSON keys match the exact column names from the profile so the decoder
    can map them back unambiguously.
    """

    def __init__(
        self,
        profile: DatasetProfile,
        float_precision: int = 3,
    ):
        super().__init__(profile, float_precision)

    def encode_row(
        self,
        row: pd.Series,
        permute: bool = False,
        target_last: bool = False,
    ) -> EncodedRow:
        """Encode row as a JSON object string."""
        columns = list(row.index)

        if target_last and self.profile.target_column:
            features = [c for c in columns if c != self.profile.target_column]
            target = self.profile.target_column
            if permute:
                random.shuffle(features)
            column_order = features + [target]
        elif permute:
            column_order = columns.copy()
            random.shuffle(column_order)
        else:
            column_order = columns

        obj: Dict[str, Any] = {}
        for col in column_order:
            col_profile = self.profile.columns[col]
            value = row[col]
            if pd.isna(value):
                obj[col] = None
            elif col_profile.dtype == "integer":
                obj[col] = int(value)
            elif col_profile.dtype == "float":
                obj[col] = round(float(value), self.float_precision)
            elif col_profile.dtype == "boolean":
                obj[col] = _str_to_bool(value)
            else:
                obj[col] = str(value)

        text = json.dumps(obj, ensure_ascii=False)

        return EncodedRow(
            text=text,
            column_order=column_order,
            values=row.to_dict(),
            metadata={"encoder": "json", "permute": permute, "target_last": target_last},
        )

    def encode_batch(
        self,
        df: pd.DataFrame,
        permute: bool = False,
        target_last: bool = False,
    ) -> List[EncodedRow]:
        """Encode batch of rows as JSON objects."""
        return [
            self.encode_row(row, permute=permute, target_last=target_last)
            for _, row in df.iterrows()
        ]

    def build_json_schema(self) -> Dict[str, Any]:
        """
        Build a JSON Schema describing the expected row format.

        Used for vLLM guided JSON decoding (``guided_json`` parameter).
        The schema constrains the LLM to produce only valid JSON matching
        the dataset's column types.

        Returns:
            JSON Schema dict compatible with vLLM's ``guided_json``.
        """
        properties: Dict[str, Any] = {}
        for col_name in self.profile.column_order:
            col_prof = self.profile.columns[col_name]
            if col_prof.dtype == "integer":
                properties[col_name] = {"type": "integer"}
            elif col_prof.dtype == "float":
                properties[col_name] = {"type": "number"}
            elif col_prof.dtype == "boolean":
                properties[col_name] = {"type": "boolean"}
            elif col_prof.dtype == "categorical":
                if col_prof.domain:
                    properties[col_name] = {
                        "type": "string",
                        "enum": list(col_prof.domain.keys()),
                    }
                else:
                    properties[col_name] = {"type": "string"}
            else:
                properties[col_name] = {"type": "string"}

        return {
            "type": "object",
            "properties": properties,
            "required": list(self.profile.column_order),
            "additionalProperties": False,
        }


def get_encoder(mode: str, profile: DatasetProfile, **kwargs) -> BaseEncoder:
    """Factory function to get encoder by mode"""
    if mode == "fast":
        return TabulaEncoder(profile, **kwargs)
    elif mode == "flexible":
        return GReaTEncoder(profile, **kwargs)
    elif mode == "predictive":
        return PredLLMEncoder(profile, **kwargs)
    elif mode == "json":
        return JsonEncoder(profile, **kwargs)
    else:
        raise ValueError(f"Unknown mode: {mode}. Choose from: fast, flexible, predictive, json")
