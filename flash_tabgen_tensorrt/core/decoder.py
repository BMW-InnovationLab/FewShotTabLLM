"""
Decoder: Text to typed tabular data with validators and repair logic
"""

import json
import re
import logging
from typing import Dict, List, Any
import pandas as pd

from flash_tabgen_tensorrt.core.data_profiler import DatasetProfile, ColumnProfile

logger = logging.getLogger(__name__)


class DecoderError(Exception):
    """Decoder validation error"""

    pass


class Decoder:
    """Decodes generated text back to typed tabular data"""

    def __init__(self, profile: DatasetProfile, mode: str = "flexible"):
        self.profile = profile
        self.mode = mode
        self.repair_enabled = True

    def decode_text(self, text: str) -> Dict[str, Any]:
        """Decode single row from text"""
        # Parse text based on mode
        if self.mode == "flexible":
            return self._decode_great_format(text)
        elif self.mode == "json":
            return self._decode_json_format(text)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def decode_batch(self, texts: List[str]) -> pd.DataFrame:
        """
        Decode batch of texts to DataFrame.

        Handles:
        - Single row per text
        - Multiple rows in one text (Qwen3 batch output)
        - JSON mode: each text may contain multiple JSON objects (one per line)
        """
        rows = []
        logger.debug("decode_batch called with %d text(s), mode=%s", len(texts), self.mode)

        for text_idx, text in enumerate(texts):
            if self.mode == "json":
                rows.extend(self._extract_json_rows(text))
            else:
                # GReaT format: check if text contains multiple numbered rows
                lines = text.strip().split("\n")
                multi_row_lines = [l for l in lines if re.match(r"^\s*\d+\.", l.strip())]
                if len(multi_row_lines) > 1:
                    for line in multi_row_lines:
                        try:
                            row = self.decode_text(line)
                            rows.append(row)
                        except Exception:
                            continue
                else:
                    try:
                        row = self.decode_text(text)
                        rows.append(row)
                    except Exception:
                        continue

        df = pd.DataFrame(rows, columns=self.profile.column_order)

        # Type conversion and validation (preserves NaN — see _apply_types docs)
        df = self._apply_types(df)

        # ------------------------------------------------------------------
        # Drop rows with missing values — but ONLY for columns that are NOT
        # naturally nullable.  If the original dataset has missing values for
        # a column (``missing_rate > 0`` in the profile), NaN is expected
        # and the row should be kept.  This avoids discarding 20-50% of rows
        # for datasets like "child" where DuctFlow has 36% missingness.
        # ------------------------------------------------------------------
        # Determine which columns MUST be non-null (required columns)
        nullable_cols = set()
        for col in self.profile.column_order:
            if col in self.profile.columns:
                col_prof = self.profile.columns[col]
                if hasattr(col_prof, "missing_rate") and col_prof.missing_rate > 0:
                    nullable_cols.add(col)

        required_cols = [c for c in df.columns if c not in nullable_cols]

        before_count = len(df)
        null_counts = df.isnull().sum()
        total_missing = int(null_counts.sum())
        if total_missing > 0:
            logger.debug("[Decoder] Columns with missing values (before dropna):")
            for col, count in null_counts.items():
                if count > 0:
                    is_nullable = col in nullable_cols
                    logger.debug(
                        "  %s: %d/%d rows missing%s",
                        col,
                        count,
                        before_count,
                        " (nullable — kept)" if is_nullable else "",
                    )

        if nullable_cols:
            logger.debug("[Decoder] Nullable columns (NaN tolerated): %s", sorted(nullable_cols))

        # Only drop rows where REQUIRED (non-nullable) columns have NaN
        if required_cols:
            df = df.dropna(subset=required_cols)
        # else: all columns are nullable, keep everything

        dropped = before_count - len(df)
        if dropped > 0:
            logger.info(
                "[Decoder] Dropped %d/%d incomplete rows (%.0f%% loss)",
                dropped,
                before_count,
                dropped / before_count * 100 if before_count else 0,
            )
        logger.debug("Decoded DataFrame shape: %s", df.shape)

        # Convert integer columns to int dtype now that NaN rows are gone
        # (only for non-nullable integer columns; nullable ones keep NaN)
        for col in df.columns:
            if col in self.profile.columns and self.profile.columns[col].dtype == "integer":
                if col in nullable_cols:
                    # Use nullable integer type to preserve NaN
                    try:
                        df[col] = df[col].astype("Int64")
                    except (ValueError, TypeError):
                        pass
                else:
                    try:
                        df[col] = df[col].astype(int)
                    except (ValueError, TypeError):
                        pass

        return df

    def _decode_great_format(self, text: str) -> Dict[str, Any]:
        """
        Decode GReaT format: 'X is Y, Z is W'

        Handles both:
        - Direct format: "col1 is val1, col2 is val2"
        - Numbered format: "1. col1 is val1, col2 is val2" (Qwen3 output)
        - Repeated index format: "1. 1. col1 is val1, col2 is val2"
        """
        row = {}

        # Remove numbering if present (e.g., "1. " or "10. " or "1. 1. ")
        # Handle repeated indices by removing all leading number patterns
        text = text.strip()
        while re.match(r"^\d+\.\s*", text):
            text = re.sub(r"^\d+\.\s*", "", text, count=1)
        # Remove trailing punctuation
        text = text.rstrip(".")

        # Split on ", <column_name> is " boundaries instead of bare commas.
        # This avoids breaking values that contain commas (e.g. "Hong Kong, China").
        # Build a regex that matches ", " followed by any known column name + " is ".
        col_pattern = "|".join(
            re.escape(col) for col in sorted(self.profile.column_order, key=len, reverse=True)
        )
        # Split pattern: comma-space then column_name then " is " (lookahead keeps the match)
        split_re = re.compile(r",\s*(?=" + col_pattern + r"\s+is\s)", re.IGNORECASE)
        clauses = [c.strip() for c in split_re.split(text)]

        for clause in clauses:
            # Parse "column is value"
            match = re.match(r"(.+?)\s+is\s+(.+)", clause, re.IGNORECASE)
            if match:
                col_name = match.group(1).strip()
                value_str = match.group(2).strip()

                # Find matching column (case-insensitive)
                matched_col = None
                for col in self.profile.column_order:
                    if col.lower() == col_name.lower():
                        matched_col = col
                        break

                if matched_col:
                    value = self._parse_value(value_str, matched_col)
                    if value is not None:
                        row[matched_col] = value

        # Fill missing columns with None
        for col in self.profile.column_order:
            if col not in row:
                row[col] = None

        return row

    # ------------------------------------------------------------------
    # JSON format decoding
    # ------------------------------------------------------------------

    def _decode_json_format(self, text: str) -> Dict[str, Any]:
        """
        Decode a single JSON object string into a row dict.

        Handles:
        - Plain JSON: ``{"age": 35, "workclass": "Private"}``
        - Numbered prefix: ``1. {"age": 35, ...}``
        - Markdown fenced blocks: ```json ... ```

        Returns a dict with all profile columns (missing keys filled with None).
        """
        text = text.strip()

        # Strip leading numbering (e.g. "1. ", "10. ")
        text = re.sub(r"^\d+\.\s*", "", text)

        # Strip markdown code fences
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

        obj = json.loads(text)
        if not isinstance(obj, dict):
            raise DecoderError(f"Expected JSON object, got {type(obj).__name__}")

        row: Dict[str, Any] = {}
        for col in self.profile.column_order:
            if col in obj and obj[col] is not None:
                value = self._parse_value(str(obj[col]), col)
                row[col] = value
            else:
                row[col] = None

        return row

    def _extract_json_rows(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract all JSON objects from a multi-row LLM response.

        The LLM may produce:
        - One JSON object per line (JSONL): ``1. {...}\\n2. {...}``
        - A JSON array: ``[{...}, {...}]``
        - Mixed text with embedded JSON objects

        Returns a list of row dicts (invalid objects are silently skipped).
        """
        text = text.strip()
        rows: List[Dict[str, Any]] = []

        # Try JSON array first
        if text.startswith("["):
            try:
                arr = json.loads(text)
                if isinstance(arr, list):
                    for item in arr:
                        if isinstance(item, dict):
                            try:
                                rows.append(self._json_obj_to_row(item))
                            except Exception:
                                continue
                    return rows
            except json.JSONDecodeError:
                pass

        # Try line-by-line (JSONL or numbered JSONL)
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            # Strip leading number prefix
            line = re.sub(r"^\d+\.\s*", "", line)
            # Find the JSON object in the line
            start = line.find("{")
            if start == -1:
                continue
            # Find matching closing brace
            depth = 0
            end = start
            for i in range(start, len(line)):
                if line[i] == "{":
                    depth += 1
                elif line[i] == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            json_str = line[start:end]
            try:
                obj = json.loads(json_str)
                if isinstance(obj, dict):
                    rows.append(self._json_obj_to_row(obj))
            except (json.JSONDecodeError, DecoderError):
                continue

        return rows

    def _json_obj_to_row(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a parsed JSON object to a profile-aligned row dict.

        Applies type parsing and fills missing columns with None.
        """
        row: Dict[str, Any] = {}
        for col in self.profile.column_order:
            if col in obj and obj[col] is not None:
                value = self._parse_value(str(obj[col]), col)
                row[col] = value
            else:
                row[col] = None
        return row

    def _parse_value(self, value_str: str, column: str) -> Any:
        """Parse value string based on column type"""
        col_prof = self.profile.columns[column]

        if value_str.lower() in ["missing", "nan", "null", "none"]:
            return None

        dtype = col_prof.dtype

        try:
            if dtype == "categorical":
                # Validate against domain
                if col_prof.domain and value_str not in col_prof.domain:
                    # Try to find closest match
                    if self.repair_enabled:
                        return self._find_closest_category(value_str, col_prof)
                return value_str

            elif dtype == "integer":
                return int(float(value_str))

            elif dtype == "float":
                return float(value_str)

            elif dtype == "boolean":
                return value_str.lower() in ["true", "yes", "1", "t", "y"]

            elif dtype == "datetime":
                # Parse datetime with detected format
                value_str = str(value_str).strip()
                try:
                    # Get the original date format if detected
                    orig_format = getattr(col_prof, "date_format", None)

                    # Handle YYYYMMDD numeric format
                    if value_str.isdigit() and len(value_str) == 8:
                        dt = pd.to_datetime(value_str, format="%Y%m%d")
                        if orig_format == "%Y%m%d":
                            return int(dt.strftime("%Y%m%d"))
                        elif orig_format:
                            return dt.strftime(orig_format)
                        return dt

                    # Try parsing with original format first
                    if orig_format:
                        try:
                            dt = pd.to_datetime(value_str, format=orig_format)
                            return (
                                dt.strftime(orig_format)
                                if orig_format != "%Y%m%d"
                                else int(dt.strftime("%Y%m%d"))
                            )
                        except (ValueError, TypeError):
                            pass

                    # Fallback to pandas auto-detection
                    dt = pd.to_datetime(value_str)
                    if orig_format == "%Y%m%d":
                        return int(dt.strftime("%Y%m%d"))
                    elif orig_format:
                        return dt.strftime(orig_format)
                    return dt
                except Exception as e:
                    return None

            else:
                return value_str

        except Exception:
            return None

    def _find_closest_category(self, value: str, col_prof: ColumnProfile) -> str:
        """Find closest matching category using string distance"""
        if not col_prof.domain:
            return value

        # Simple substring matching
        value_lower = value.lower()
        for cat in col_prof.domain.keys():
            if value_lower in cat.lower() or cat.lower() in value_lower:
                return cat

        # Return most frequent category as fallback
        if col_prof.top_k_values:
            return col_prof.top_k_values[0][0]

        return value

    def _apply_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply correct dtypes to DataFrame.

        IMPORTANT: This must preserve NaN/None so that the subsequent ``dropna()``
        in ``decode_batch`` can remove incomplete rows.  Previous versions
        accidentally masked missing values by converting categorical None → the
        literal string ``"None"`` (via ``astype(str)``) and filling integer NaN
        with the column median.  Both prevented ``dropna()`` from working.
        """
        for col in df.columns:
            if col not in self.profile.columns:
                continue

            col_prof = self.profile.columns[col]
            dtype = col_prof.dtype

            try:
                if dtype == "integer":
                    # Convert to numeric; unparseable strings become NaN.
                    # Do NOT fill NaN here — let dropna() remove incomplete rows.
                    df[col] = pd.to_numeric(df[col], errors="coerce")

                elif dtype == "float":
                    df[col] = pd.to_numeric(df[col], errors="coerce")

                elif dtype == "boolean":
                    # Map LLM output (True/False) back to the original labels
                    # found in the real dataset (e.g. "yes"/"no", "Y"/"N").
                    true_label = getattr(col_prof, "boolean_true_label", None) or True
                    false_label = getattr(col_prof, "boolean_false_label", None) or False
                    df[col] = df[col].map({
                        "True": true_label, "False": false_label,
                        True: true_label, False: false_label,
                    })

                elif dtype == "datetime":
                    # Skip — already handled correctly in _parse_value
                    pass

                elif dtype == "categorical":
                    # Convert only non-null values to str.  A bare
                    # ``df[col].astype(str)`` would turn None/NaN into the
                    # literal string "None", which defeats dropna().
                    mask = df[col].notna()
                    if mask.any():
                        df.loc[mask, col] = df.loc[mask, col].astype(str)
            except Exception as e:
                # Keep original if conversion fails
                pass

        return df
