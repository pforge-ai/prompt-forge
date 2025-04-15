# prompt_forge/datasets/loaders.py

import json
import csv # <--- 导入 csv 模块 (Import csv module)
import logging
from typing import Any, Dict, List, Optional

# 导入基类
# Import base class
from ptforge.core.base import BaseDataset

logger = logging.getLogger(__name__)


class JsonlDataset(BaseDataset):
    """
    从 JSON Lines (.jsonl) 文件加载数据集。
    文件中的每一行都应该是一个有效的 JSON 对象。

    Loads a dataset from a JSON Lines (.jsonl) file.
    Each line in the file should be a valid JSON object.
    """

    def __init__(
        self,
        file_path: str,
        input_field: str = "input",
        reference_field: str = "reference",
        encoding: str = "utf-8",
    ):
        """
        初始化 JsonlDataset。

        Args:
            file_path: .jsonl 文件的路径。 (Path to the .jsonl file.)
            input_field: JSON 对象中包含输入数据的字段名称。
                            (Field name in the JSON object containing the input data.)
            reference_field: JSON 对象中包含参考输出的字段名称。
                                (Field name in the JSON object containing the reference output.)
            encoding: 读取文件时使用的编码。 (Encoding to use when reading the file.)

        Raises:
            FileNotFoundError: 如果文件路径不存在。 (If the file path does not exist.)
            ValueError: 如果文件解析或字段提取过程中发生错误。 (If errors occur during file parsing or field extraction.)
        """
        super().__init__()
        self.file_path = file_path
        self.input_field = input_field
        self.reference_field = reference_field
        self.encoding = encoding
        self._data: List[Dict[str, Any]] = [] # 存储加载的数据 (Store loaded data)

        self._load_data() # 初始化时加载数据 (Load data on initialization)

    def _load_data(self):
        """加载并解析 .jsonl 文件。"""
        logger.info(f"Loading data from JSON Lines file: {self.file_path}")
        line_num = 0
        loaded_count = 0
        error_count = 0
        try:
            with open(self.file_path, "r", encoding=self.encoding) as f:
                for line in f:
                    line_num += 1
                    line = line.strip()
                    if not line: # 跳过空行 (Skip empty lines)
                        continue
                    try:
                        record = json.loads(line)
                        # 验证必需的字段是否存在 (Validate required fields exist)
                        if self.input_field not in record:
                            logger.error(f"Missing input field '{self.input_field}' in line {line_num}: {line}")
                            error_count += 1
                            continue # 跳过此记录 (Skip this record)
                        # 参考字段是可选的，但如果指定了字段名，最好检查一下
                        # Reference field is optional, but check if specified field name exists
                        if self.reference_field not in record:
                            logger.warning(f"Missing reference field '{self.reference_field}' in line {line_num}. Setting reference to None.")
                            # Add the field with None value if missing? Or handle in getitem?
                            # Let's handle in getitem by using .get()

                        self._data.append(record)
                        loaded_count += 1
                    except json.JSONDecodeError as e:
                        logger.error(f"Error decoding JSON on line {line_num}: {e} - Line: '{line}'")
                        error_count += 1
                    except Exception as e: # Catch other potential errors per line
                            logger.error(f"Error processing line {line_num}: {e} - Line: '{line}'", exc_info=True)
                            error_count += 1

        except FileNotFoundError:
            logger.error(f"Dataset file not found: {self.file_path}")
            raise
        except Exception as e: # Catch errors during file opening/reading
            logger.error(f"Failed to load dataset file {self.file_path}: {e}", exc_info=True)
            raise ValueError(f"Failed to load dataset file {self.file_path}") from e

        logger.info(f"Successfully loaded {loaded_count} records from {self.file_path}.")
        if error_count > 0:
            logger.warning(f"Encountered {error_count} errors while loading data.")
        if not self._data:
                logger.warning(f"Loaded dataset is empty: {self.file_path}")


    def __len__(self) -> int:
        """返回加载的数据记录数。 (Return the number of loaded data records.)"""
        return len(self._data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        获取指定索引的数据样本，格式化为 {'input': ..., 'reference': ...}。
        Gets the data sample at the specified index, formatted as {'input': ..., 'reference': ...}.
        """
        if idx < 0 or idx >= len(self._data):
            raise IndexError(f"Index {idx} out of bounds for dataset with length {len(self._data)}")

        record = self._data[idx]

        input_data = record.get(self.input_field) # Use .get for safety, though checked in load
        # Reference data might be missing, use .get() which returns None if key is absent
        reference_data = record.get(self.reference_field)

        if input_data is None:
                # This shouldn't happen if loading logic is correct, but as a safeguard
                logger.error(f"Input data is None for index {idx} despite initial check. Record: {record}")
                # Decide how to handle this - raise error or return default? Let's raise.
                raise ValueError(f"Missing or null input data at index {idx} for field '{self.input_field}'")


        return {"input": input_data, "reference": reference_data}

    # get_batches 方法使用基类的默认实现即可
    # get_batches method uses the default implementation from the base class

    def __repr__(self) -> str:
        return f"JsonlDataset(file_path='{self.file_path}', count={len(self)})"


class CsvDataset(BaseDataset):
    """
    从 CSV (.csv) 文件加载数据集。
    假定文件第一行为表头。

    Loads a dataset from a CSV (.csv) file.
    Assumes the first row is the header.
    """

    def __init__(
        self,
        file_path: str,
        input_col: str, # 输入数据所在的列名 (Column name for input data)
        reference_col: str, # 参考输出所在的列名 (Column name for reference output)
        delimiter: str = ",",
        quotechar: str = '"',
        encoding: str = "utf-8",
        **csv_reader_kwargs: Any # 其他传递给 csv.DictReader 的参数 (Other args for csv.DictReader)
    ):
        """
        初始化 CsvDataset。

        Args:
            file_path: .csv 文件的路径。 (Path to the .csv file.)
            input_col: 包含输入数据的列名。 (Column name containing the input data.)
            reference_col: 包含参考输出的列名。 (Column name containing the reference output.)
            delimiter: CSV 文件使用的分隔符。 (Delimiter used in the CSV file.)
            quotechar: CSV 文件使用的引用字符。 (Quote character used in the CSV file.)
            encoding: 读取文件时使用的编码。 (Encoding to use when reading the file.)
            **csv_reader_kwargs: 其他传递给 csv.DictReader 的关键字参数。
                                    (Additional keyword arguments passed to csv.DictReader.)

        Raises:
            FileNotFoundError: 如果文件路径不存在。 (If the file path does not exist.)
            ValueError: 如果指定的列名在 CSV 表头中不存在，或在解析时发生错误。
                        (If specified columns are not found in the CSV header, or if parsing errors occur.)
        """
        super().__init__()
        self.file_path = file_path
        self.input_col = input_col
        self.reference_col = reference_col
        self.delimiter = delimiter
        self.quotechar = quotechar
        self.encoding = encoding
        self.csv_reader_kwargs = csv_reader_kwargs
        self._data: List[Dict[str, str]] = [] # CSV 读取器通常返回字符串 (CSV reader usually returns strings)

        self._load_data()

    def _load_data(self):
        """加载并解析 .csv 文件。"""
        logger.info(f"Loading data from CSV file: {self.file_path}")
        loaded_count = 0
        error_count = 0
        try:
            with open(self.file_path, "r", newline='', encoding=self.encoding) as csvfile:
                # 使用 DictReader 直接读取为字典列表
                # Use DictReader to read directly into a list of dictionaries
                reader = csv.DictReader(
                    csvfile,
                    delimiter=self.delimiter,
                    quotechar=self.quotechar,
                    **self.csv_reader_kwargs
                )

                # 检查表头是否包含指定的列
                # Check if header contains the specified columns
                headers = reader.fieldnames
                if headers is None:
                    raise ValueError("Could not read CSV header.")
                if self.input_col not in headers:
                    raise ValueError(f"Input column '{self.input_col}' not found in CSV header: {headers}")
                if self.reference_col not in headers:
                    # 参考列是可选的，但如果指定了列名却找不到，则应警告或报错
                    # Reference column is optional, but if specified and not found, warn or error
                    # Let's raise an error for clarity if specified but missing.
                    raise ValueError(f"Reference column '{self.reference_col}' not found in CSV header: {headers}")
                    # Alternatively, log warning and proceed, handling missing ref in getitem:
                    # logger.warning(f"Reference column '{self.reference_col}' not found in CSV header: {headers}. References will be None.")

                for row_num, row in enumerate(reader):
                        # row 是一个字典 {header: value}
                        # row is a dictionary {header: value}
                    try:
                        # 可以添加行级验证逻辑 (Row-level validation logic can be added here)
                        # 例如检查 input_col 是否为空 (e.g., check if input_col is empty)
                        if not row.get(self.input_col): # Check for empty input
                            logger.warning(f"Empty input value in column '{self.input_col}' at row {row_num + 2}. Skipping row.") # +2 for header and 0-index
                            error_count += 1
                            continue

                        self._data.append(row)
                        loaded_count += 1
                    except Exception as e: # Catch potential errors processing a specific row
                        logger.error(f"Error processing CSV row {row_num + 2}: {e} - Row: {row}", exc_info=True)
                        error_count += 1

        except FileNotFoundError:
            logger.error(f"Dataset file not found: {self.file_path}")
            raise
        except ValueError as e: # Catch header validation errors
                logger.error(f"Header validation failed for {self.file_path}: {e}")
                raise
        except csv.Error as e: # Catch general CSV parsing errors
                logger.error(f"Error parsing CSV file {self.file_path}: {e}", exc_info=True)
                raise ValueError(f"Failed to parse CSV file {self.file_path}") from e
        except Exception as e: # Catch other file reading errors
            logger.error(f"Failed to load dataset file {self.file_path}: {e}", exc_info=True)
            raise ValueError(f"Failed to load dataset file {self.file_path}") from e

        logger.info(f"Successfully loaded {loaded_count} records from {self.file_path}.")
        if error_count > 0:
            logger.warning(f"Encountered {error_count} errors while loading data.")
        if not self._data:
                logger.warning(f"Loaded dataset is empty: {self.file_path}")


    def __len__(self) -> int:
        """返回加载的数据记录数。 (Return the number of loaded data records.)"""
        return len(self._data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        获取指定索引的数据样本，格式化为 {'input': ..., 'reference': ...}。
        Gets the data sample at the specified index, formatted as {'input': ..., 'reference': ...}.
        """
        if idx < 0 or idx >= len(self._data):
            raise IndexError(f"Index {idx} out of bounds for dataset with length {len(self._data)}")

        record = self._data[idx]

        # DictReader ensures keys exist if header check passed, but use .get for safety
        input_data = record.get(self.input_col)
        reference_data = record.get(self.reference_col) # Returns None if column was missing (and warning was used instead of error)

        if input_data is None:
                # This might happen if the value was empty and we didn't skip it during load
                logger.error(f"Input data is None or empty at index {idx} for column '{self.input_col}'. Record: {record}")
                raise ValueError(f"Missing or null input data at index {idx} for column '{self.input_col}'")


        return {"input": input_data, "reference": reference_data}

    def __repr__(self) -> str:
        return f"CsvDataset(file_path='{self.file_path}', count={len(self)})"

