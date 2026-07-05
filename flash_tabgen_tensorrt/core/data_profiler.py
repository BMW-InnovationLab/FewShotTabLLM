"""
Data Profiler: Schema inference, type detection, and statistics computation

Implements robust type detection and statistical profiling for arbitrary tabular datasets.
Supports categorical, numeric (int/float), boolean, datetime, and text columns.
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import warnings

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_bool_dtype, is_datetime64_dtype


@dataclass
class ColumnProfile:
    """Profile for a single column"""
    name: str
    dtype: str  # 'categorical', 'integer', 'float', 'boolean', 'datetime', 'text', 'id'
    original_dtype: str
    
    # Statistics
    missing_rate: float = 0.0
    unique_count: int = 0
    
    # Categorical
    domain: Optional[Dict[str, int]] = None  # value -> count
    top_k_values: Optional[List[Tuple[str, int]]] = None
    cardinality: int = 0
    abbreviation_map: Optional[Dict[str, str]] = None  # For TabuLa compression
    
    # Numeric
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    quantiles: Optional[Dict[str, float]] = None  # Q5, Q25, Q50, Q75, Q95
    
    # Text
    avg_length: Optional[float] = None
    max_length: Optional[int] = None
    
    # Boolean — original labels so the decoder can restore the source representation
    boolean_true_label: Optional[str] = None   # e.g. "yes", "y", "true", "1"
    boolean_false_label: Optional[str] = None  # e.g. "no", "n", "false", "0"

    # Datetime
    date_format: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'name': self.name,
            'dtype': self.dtype,
            'original_dtype': self.original_dtype,
            'missing_rate': self.missing_rate,
            'unique_count': self.unique_count,
            'cardinality': self.cardinality,
            'domain': self.domain,
            'top_k_values': self.top_k_values,
            'abbreviation_map': self.abbreviation_map,
            'min_val': self.min_val,
            'max_val': self.max_val,
            'mean': self.mean,
            'std': self.std,
            'quantiles': self.quantiles,
            'avg_length': self.avg_length,
            'max_length': self.max_length,
            'boolean_true_label': self.boolean_true_label,
            'boolean_false_label': self.boolean_false_label,
            'date_format': self.date_format,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ColumnProfile":
        """Reconstruct a ColumnProfile from a dictionary.

        Args:
            data: Dictionary as produced by ``to_dict()``.

        Returns:
            Populated ColumnProfile instance.
        """
        # top_k_values is stored as list of [value, count] pairs
        top_k = data.get('top_k_values')
        if top_k is not None:
            top_k = [(str(v), int(c)) for v, c in top_k]

        return cls(
            name=data['name'],
            dtype=data['dtype'],
            original_dtype=data.get('original_dtype', ''),
            missing_rate=data.get('missing_rate', 0.0),
            unique_count=data.get('unique_count', 0),
            cardinality=data.get('cardinality', 0),
            domain=data.get('domain'),
            top_k_values=top_k,
            abbreviation_map=data.get('abbreviation_map'),
            min_val=data.get('min_val'),
            max_val=data.get('max_val'),
            mean=data.get('mean'),
            std=data.get('std'),
            quantiles=data.get('quantiles'),
            avg_length=data.get('avg_length'),
            max_length=data.get('max_length'),
            boolean_true_label=data.get('boolean_true_label'),
            boolean_false_label=data.get('boolean_false_label'),
            date_format=data.get('date_format'),
        )


@dataclass
class DatasetProfile:
    """Complete dataset profile"""
    n_rows: int
    n_cols: int
    columns: Dict[str, ColumnProfile] = field(default_factory=dict)
    column_order: List[str] = field(default_factory=list)
    target_column: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'n_rows': self.n_rows,
            'n_cols': self.n_cols,
            'columns': {name: col.to_dict() for name, col in self.columns.items()},
            'column_order': self.column_order,
            'target_column': self.target_column,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "DatasetProfile":
        """Reconstruct a DatasetProfile from a dictionary.

        Args:
            data: Dictionary as produced by ``to_dict()``.

        Returns:
            Populated DatasetProfile instance.
        """
        columns = {
            name: ColumnProfile.from_dict(col_data)
            for name, col_data in data.get('columns', {}).items()
        }
        return cls(
            n_rows=data['n_rows'],
            n_cols=data['n_cols'],
            columns=columns,
            column_order=data.get('column_order', list(columns.keys())),
            target_column=data.get('target_column'),
        )

    def save(self, path: str) -> None:
        """Save profile to a JSON file.

        Args:
            path: Filesystem path for the output JSON file.
        """
        import json
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, path: str) -> "DatasetProfile":
        """Load a profile from a JSON file.

        Args:
            path: Path to a JSON file produced by ``save()``.

        Returns:
            Populated DatasetProfile instance.
        """
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def apply_overrides(self, overrides: Dict[str, str]) -> None:
        """Apply type overrides to columns in-place.

        This lets an agent review the profile and correct misclassified
        columns before generation.  Statistics already computed for the
        original type are preserved; only ``dtype`` is changed.

        Args:
            overrides: Mapping of column name to corrected dtype string,
                e.g. ``{"billing_address": "text", "checkin_date": "datetime"}``.
        """
        valid_types = {
            'categorical', 'integer', 'float', 'boolean', 'datetime', 'text', 'id',
        }
        for col_name, new_type in overrides.items():
            if new_type not in valid_types:
                raise ValueError(
                    f"Invalid dtype '{new_type}' for column '{col_name}'. "
                    f"Valid types: {sorted(valid_types)}"
                )
            if col_name not in self.columns:
                raise KeyError(f"Column '{col_name}' not found in profile")
            self.columns[col_name].dtype = new_type


class DataProfiler:
    """
    Profiles tabular datasets for schema inference and statistics computation.
    
    Implements robust type detection with configurable thresholds and supports
    automatic abbreviation generation for TabuLa-style compression.
    """
    
    def __init__(
        self,
        categorical_threshold: int = 20,  # Max unique values for categorical
        integer_threshold: float = 0.95,  # Min % of integer values to be int type
        top_k_categories: int = 100,  # Top K categories to keep in domain
        abbreviation_length: int = 3,  # Max abbreviation length for compression
    ):
        self.categorical_threshold = categorical_threshold
        self.integer_threshold = integer_threshold
        self.top_k_categories = top_k_categories
        self.abbreviation_length = abbreviation_length
    
    def profile(
        self,
        data: pd.DataFrame,
        target_column: Optional[str] = None,
        type_overrides: Optional[Dict[str, str]] = None,
    ) -> DatasetProfile:
        """
        Profile a dataset and infer schema with statistics.
        
        Args:
            data: Input DataFrame
            target_column: Name of target column (for Predictive mode)
            type_overrides: Manual type specifications {column: type}
        
        Returns:
            DatasetProfile with complete schema and statistics
        """
        type_overrides = type_overrides or {}
        
        profile = DatasetProfile(
            n_rows=len(data),
            n_cols=len(data.columns),
            column_order=list(data.columns),
            target_column=target_column,
        )
        
        for col in data.columns:
            if col in type_overrides:
                dtype = type_overrides[col]
            else:
                dtype = self._infer_type(data[col])
            
            col_profile = self._profile_column(data[col], dtype)
            profile.columns[col] = col_profile
        
        return profile
    
    def _infer_type(self, series: pd.Series) -> str:
        """
        Infer column type using heuristics and sample-based checks.
        
        Returns one of: 'categorical', 'integer', 'float', 'boolean', 'datetime', 'text'
        """
        # Remove missing values for type inference
        non_null = series.dropna()
        if len(non_null) == 0:
            return 'categorical'  # Default for empty columns
        
        # Check pandas dtype first
        if is_bool_dtype(series):
            return 'boolean'
        
        if is_datetime64_dtype(series):
            return 'datetime'
        
        if is_numeric_dtype(series):
            # Check if integer or float
            if self._is_integer_column(non_null):
                return 'integer'
            return 'float'
        
        # String/object columns - need deeper inspection
        unique_count = series.nunique()
        total_count = len(non_null)
        
        # Check for boolean
        if self._is_boolean_like(non_null):
            return 'boolean'
        
        # Check for datetime
        if self._is_datetime_like(non_null):
            return 'datetime'
        
        # Check for numeric stored as string
        if self._is_numeric_string(non_null):
            if self._is_integer_column(pd.to_numeric(non_null, errors='coerce').dropna()):
                return 'integer'
            return 'float'
        
        # Categorical vs text based on cardinality
        if unique_count <= self.categorical_threshold:
            return 'categorical'

        # High cardinality string column — classify further
        avg_length = non_null.astype(str).str.len().mean()

        # Long strings are likely free text
        if avg_length > 50:
            return 'text'

        # ID detection: name-based (column name contains 'id' token)
        if self._is_id_column_name(series.name):
            return 'id'

        # ID detection: content-based — (near-)unique short strings
        uniqueness_ratio = unique_count / total_count
        if uniqueness_ratio > 0.95 and avg_length < 30:
            return 'id'

        # High-uniqueness medium-length strings (30-50 chars) are likely
        # free-form text (addresses, full names, descriptions) even though
        # they're below the 50-char text threshold.
        if uniqueness_ratio > 0.90 and avg_length >= 30:
            return 'text'

        # High-cardinality string column that is NOT id and NOT text.
        # Treat as categorical but the prompt builder will handle it
        # differently (sample values instead of exhaustive top-5).
        return 'categorical'
    
    def _is_integer_column(self, series: pd.Series) -> bool:
        """Check if numeric column is integer"""
        if not is_numeric_dtype(series):
            return False
        
        # Check if values are close to integers
        int_check = np.isclose(series, series.round(), rtol=1e-9, atol=1e-9)
        return int_check.mean() >= self.integer_threshold
    
    def _is_boolean_like(self, series: pd.Series) -> bool:
        """Check if column contains boolean-like values"""
        unique_values = set(series.astype(str).str.lower().unique())
        boolean_sets = [
            {'true', 'false'},
            {'yes', 'no'},
            {'y', 'n'},
            {'1', '0'},
            {'t', 'f'},
        ]
        return any(unique_values.issubset(s) for s in boolean_sets)
    
    def _is_datetime_like(self, series: pd.Series, sample_size: int = 100) -> bool:
        """Check if column contains datetime-like strings"""
        sample = series.sample(min(sample_size, len(series)))
        
        # Common date patterns
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'\d{2}-\d{2}-\d{4}',  # DD-MM-YYYY
        ]
        
        matches = 0
        for val in sample.astype(str):
            if any(re.search(pattern, val) for pattern in date_patterns):
                matches += 1
        
        return matches / len(sample) > 0.8
    
    def _is_numeric_string(self, series: pd.Series, sample_size: int = 100) -> bool:
        """Check if column contains numeric values stored as strings"""
        sample = series.sample(min(sample_size, len(series)))
        numeric_count = pd.to_numeric(sample, errors='coerce').notna().sum()
        return numeric_count / len(sample) > 0.9

    def _is_id_column_name(self, column_name: str) -> bool:
        """Return True if the column name contains 'id' as a distinct token.

        Handles underscores, hyphens, spaces, and camelCase.
        Examples that match: ``pet_id``, ``userId``, ``ID``, ``user-id``.
        Examples that don't: ``identity``, ``video``, ``grid``.
        """
        tokens = re.sub(r'[-_ ]+', ' ', str(column_name)).split()
        # Also split camelCase
        expanded = []
        for tok in tokens:
            parts = re.findall(r'[A-Z][^A-Z]*', tok) if tok != tok.upper() and tok != tok.lower() else [tok]
            expanded.extend(parts if parts else [tok])
        return 'id' in [t.lower() for t in expanded]
    
    def _profile_column(self, series: pd.Series, dtype: str) -> ColumnProfile:
        """Profile a single column based on its type"""
        profile = ColumnProfile(
            name=series.name,
            dtype=dtype,
            original_dtype=str(series.dtype),
            missing_rate=series.isna().mean(),
            unique_count=series.nunique(),
        )
        
        non_null = series.dropna()
        
        if dtype == 'categorical':
            self._profile_categorical(non_null, profile)
        elif dtype in ['integer', 'float']:
            self._profile_numeric(non_null, profile, dtype)
        elif dtype == 'boolean':
            self._profile_boolean(non_null, profile)
        elif dtype == 'datetime':
            self._profile_datetime(non_null, profile)
        elif dtype == 'text':
            self._profile_text(non_null, profile)
        elif dtype == 'id':
            self._profile_id(non_null, profile)
        
        return profile
    
    def _profile_categorical(self, series: pd.Series, profile: ColumnProfile):
        """Profile categorical column"""
        value_counts = series.value_counts()
        profile.cardinality = len(value_counts)
        
        # Top K values
        top_k = value_counts.head(self.top_k_categories)
        profile.top_k_values = [(str(val), int(count)) for val, count in top_k.items()]
        profile.domain = {str(val): int(count) for val, count in value_counts.items()}
        
        # Generate abbreviations for TabuLa compression
        profile.abbreviation_map = self._generate_abbreviations(
            list(value_counts.index)
        )
    
    def _profile_numeric(self, series: pd.Series, profile: ColumnProfile, dtype: str):
        """Profile numeric column"""
        # Convert to numeric if needed
        if not is_numeric_dtype(series):
            series = pd.to_numeric(series, errors='coerce').dropna()
        
        profile.min_val = float(series.min())
        profile.max_val = float(series.max())
        profile.mean = float(series.mean())
        profile.std = float(series.std())
        
        # Quantiles for bucketing
        profile.quantiles = {
            'Q5': float(series.quantile(0.05)),
            'Q25': float(series.quantile(0.25)),
            'Q50': float(series.quantile(0.50)),
            'Q75': float(series.quantile(0.75)),
            'Q95': float(series.quantile(0.95)),
        }
    
    def _profile_boolean(self, series: pd.Series, profile: ColumnProfile):
        """Profile boolean column and record original true/false labels."""
        value_counts = series.value_counts()
        profile.domain = {str(val): int(count) for val, count in value_counts.items()}
        profile.cardinality = len(value_counts)

        # Detect the original labels so the decoder can restore them.
        truthy = {"true", "yes", "y", "1", "t"}
        for val in profile.domain:
            low = val.strip().lower()
            if low in truthy:
                profile.boolean_true_label = val
            else:
                profile.boolean_false_label = val
    
    def _profile_datetime(self, series: pd.Series, profile: ColumnProfile):
        """Profile datetime column"""
        # Detect date format from sample
        profile.date_format = self._detect_date_format(series)
        
        # Try to convert to datetime
        try:
            if profile.date_format:
                dt_series = pd.to_datetime(series, format=profile.date_format, errors='coerce').dropna()
            else:
                dt_series = pd.to_datetime(series, errors='coerce').dropna()
            if len(dt_series) > 0:
                profile.min_val = dt_series.min().isoformat()
                profile.max_val = dt_series.max().isoformat()
        except:
            warnings.warn(f"Could not parse datetime column: {profile.name}")
    
    def _detect_date_format(self, series: pd.Series) -> Optional[str]:
        """Detect date format from sample values"""
        sample = series.dropna().head(10).astype(str)
        if len(sample) == 0:
            return None
        
        # Common date formats to try
        formats = [
            ('%Y%m%d', r'^\d{8}$'),  # 20210305
            ('%Y-%m-%d', r'^\d{4}-\d{2}-\d{2}$'),  # 2021-03-05
            ('%d/%m/%Y', r'^\d{2}/\d{2}/\d{4}$'),  # 05/03/2021
            ('%m/%d/%Y', r'^\d{2}/\d{2}/\d{4}$'),  # 03/05/2021
            ('%d-%m-%Y', r'^\d{2}-\d{2}-\d{4}$'),  # 05-03-2021
            ('%d %b %Y', r'^\d{2} \w{3} \d{4}$'),  # 05 Mar 2021
            ('%d %B %Y', r'^\d{2} \w+ \d{4}$'),  # 05 March 2021
            ('%b %d, %Y', r'^\w{3} \d{1,2}, \d{4}$'),  # Mar 05, 2021
            ('%Y-%m-%d %H:%M:%S', r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$'),  # 2021-03-05 10:30:00
        ]
        
        for fmt, pattern in formats:
            matches = 0
            for val in sample:
                try:
                    pd.to_datetime(val, format=fmt)
                    matches += 1
                except:
                    pass
            if matches >= len(sample) * 0.8:  # 80% threshold
                return fmt
        
        return None
    
    def _profile_text(self, series: pd.Series, profile: ColumnProfile):
        """Profile text column"""
        str_series = series.astype(str)
        lengths = str_series.str.len()
        profile.avg_length = float(lengths.mean())
        profile.max_length = int(lengths.max())

    def _profile_id(self, series: pd.Series, profile: ColumnProfile):
        """Profile ID / unique-identifier column.

        Captures length stats and a small sample for format illustration,
        but does NOT store a full domain or top-k values (to prevent the
        prompt from listing values that the model would then over-use).
        """
        str_series = series.astype(str)
        lengths = str_series.str.len()
        profile.avg_length = float(lengths.mean())
        profile.max_length = int(lengths.max())
        profile.cardinality = int(series.nunique())

        # Store a tiny sample (3 values) so the prompt can show the *format*
        sample_vals = series.sample(min(3, len(series))).astype(str).tolist()
        profile.top_k_values = [(v, 0) for v in sample_vals]
    
    def _generate_abbreviations(
        self,
        values: List[str],
        length: Optional[int] = None,
    ) -> Dict[str, str]:
        """
        Generate unique abbreviations for categorical values (TabuLa compression).
        
        Creates bijective mapping: value -> short abbreviation (1-3 chars)
        """
        length = length or self.abbreviation_length
        abbreviations = {}
        used_abbrevs = set()
        
        for value in values:
            value_str = str(value)
            
            # Try different abbreviation strategies
            candidates = []
            
            # Strategy 1: First N characters
            if len(value_str) >= length:
                candidates.append(value_str[:length].upper())
            
            # Strategy 2: Initials (for multi-word)
            words = re.findall(r'\b\w', value_str)
            if len(words) >= 2:
                candidates.append(''.join(words[:length]).upper())
            
            # Strategy 3: Consonants
            consonants = re.sub(r'[aeiouAEIOU\s]', '', value_str)
            if len(consonants) >= length:
                candidates.append(consonants[:length].upper())
            
            # Strategy 4: First char + vowel removal
            no_vowels = value_str[0] + re.sub(r'[aeiouAEIOU\s]', '', value_str[1:])
            if len(no_vowels) >= length:
                candidates.append(no_vowels[:length].upper())
            
            # Find first unique abbreviation
            abbrev = None
            for candidate in candidates:
                if candidate not in used_abbrevs:
                    abbrev = candidate
                    break
            
            # Fallback: add numeric suffix
            if abbrev is None:
                base = candidates[0] if candidates else value_str[:length].upper()
                suffix = 0
                while f"{base}{suffix}" in used_abbrevs:
                    suffix += 1
                abbrev = f"{base}{suffix}"
            
            abbreviations[value_str] = abbrev
            used_abbrevs.add(abbrev)
        
        return abbreviations
    
    def get_column_abbreviation(self, column_name: str) -> str:
        """Generate single-character abbreviation for column name"""
        # Use first character, capitalize
        return column_name[0].upper()
