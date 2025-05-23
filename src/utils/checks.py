import os
import logging

logger = logging.getLogger(__name__)

def validate_file_exists(path: str):
    if not os.path.exists(path):
        logger.error(f"Datei nicht gefunden: {path}")
        raise FileNotFoundError(f"Datei existiert nicht: {path}")
    else:
        logger.debug(f"Datei existiert: {path}")


def validate_dataframe_columns(df, expected_columns: list, df_name="DataFrame"):
    missing = [col for col in expected_columns if col not in df.columns]
    if missing:
        logger.error(f"{df_name}: Fehlende Spalten: {missing}")
        raise ValueError(f"{df_name}: Fehlende Spalten: {missing}")
    else:
        logger.debug(f"{df_name}: Alle erwarteten Spalten vorhanden.")


def assert_no_column(df, column: str, df_name="DataFrame"):
    if column in df.columns:
        logger.error(f"{df_name} enthält verbotene Spalte '{column}' (Leakage-Gefahr).")
        raise ValueError(f"{df_name} darf Spalte '{column}' nicht enthalten.")
    else:
        logger.debug(f"{df_name} enthält korrekt NICHT die Spalte '{column}'.")
