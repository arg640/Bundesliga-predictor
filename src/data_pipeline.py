import pandas as pd
import soccerdata as sd
from pathlib import Path

def fetch_bundesliga_data(seasons):
    """
    Descarga datos históricos de la Bundesliga desde Understat.
    
    Args:
        seasons: Lista de temporadas en formato "YYYY/YYYY".
    
    Returns:
        DataFrame con una fila por partido y estadísticas avanzadas (incluye xG).
    """
    understat = sd.Understat(
        leagues="GER-Bundesliga",
        seasons=seasons
    )
    df = understat.read_team_match_stats()
    df = df.reset_index()
    return df

def save_raw_data(df, filename):
    """
    Guarda un DataFrame en data/raw/ como CSV.
    
    Args:
        df: DataFrame a guardar.
        filename: Nombre del archivo (ej: "bundesliga_matches.csv").
    
    Returns:
        Ruta completa del archivo guardado.
    """
    RAW_DATA_DIR = Path("data") / "raw"
    path = RAW_DATA_DIR / filename
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path

if __name__ == "__main__":
    seasons = ["2021/2022", "2022/2023", "2023/2024", "2024/2025", "2025/2026"]
    df = fetch_bundesliga_data(seasons)
    path = save_raw_data(df, "bundesliga_matches.csv")
    print(f"Listo: {path}")