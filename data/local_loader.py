"""
Local CSV Loader
================
outputs 폴더의 예측 결과 CSV를 로드하여 대시보드에 표시
"""

import pandas as pd
import json
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

# 상위 디렉토리 import
sys.path.insert(0, str(Path(__file__).parent.parent))
from components.chatbot import get_feature_description

# outputs 폴더 경로 (dashboard 기준 상위 폴더)
OUTPUTS_DIR = Path(__file__).parent.parent.parent / "outputs"


def get_available_dates(store: str) -> list:
    """
    해당 점포의 사용 가능한 date_t 목록 반환

    Parameters
    ----------
    store : str
        점포 코드 (210, 220, 480)

    Returns
    -------
    list
        사용 가능한 날짜 목록 (내림차순)
    """
    csv_path = OUTPUTS_DIR / f"{store}.csv"

    if not csv_path.exists():
        return []

    try:
        df = pd.read_csv(csv_path)
        dates = df["date_t"].unique().tolist()
        dates.sort(reverse=True)
        return dates
    except Exception:
        return []


def get_predictions(
    store: str,
    date_t: str,
    horizon: int
) -> Optional[pd.DataFrame]:
    """
    예측 데이터 로드

    Parameters
    ----------
    store : str
        점포 코드 (210, 220, 480)
    date_t : str
        기준일 (YYYY-MM-DD)
    horizon : int
        예측 horizon (1~4)

    Returns
    -------
    Optional[pd.DataFrame]
        예측 데이터, 없으면 None
    """
    csv_path = OUTPUTS_DIR / f"{store}.csv"

    if not csv_path.exists():
        return None

    try:
        df = pd.read_csv(csv_path)

        # 필터링
        df = df[(df["date_t"] == date_t) & (df["horizon"] == horizon)]

        if df.empty:
            return None

        return df
    except Exception:
        return None


def transform_to_display_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    CSV 데이터를 대시보드 표시용 DataFrame으로 변환

    Parameters
    ----------
    df : pd.DataFrame
        outputs CSV에서 로드한 데이터

    Returns
    -------
    pd.DataFrame
        대시보드 표시용 DataFrame
    """
    if df.empty:
        return pd.DataFrame()

    rows = []
    for idx, row in df.iterrows():
        # top_3_features 파싱 → 한글 설명으로 변환
        top_3_str = "-"
        if pd.notna(row.get("top_3_features")):
            try:
                top_3 = row["top_3_features"]
                if isinstance(top_3, str):
                    top_3 = json.loads(top_3)
                if isinstance(top_3, list):
                    # 한글 설명으로 변환
                    top_3_korean = [get_feature_description(f) for f in top_3[:3]]
                    top_3_str = ", ".join(top_3_korean)
            except Exception:
                top_3_str = str(row["top_3_features"])

        # row_data 구성 (챗봇용)
        row_data = {
            "sku_code": str(row["sku"]),
            "sku_name": row.get("sku_name", str(row["sku"])),
            "p10": int(row["p10"]) if pd.notna(row.get("p10")) else 0,
            "p50": int(row["p50"]) if pd.notna(row.get("p50")) else 0,
            "p90": int(row["p90"]) if pd.notna(row.get("p90")) else 0,
            "model_name": row.get("model_name", ""),
            "n_train_samples": row.get("n_train_samples", 0),
            "top_3_features": row.get("top_3_features", "[]"),
            "lag_1": row.get("lag_1"),
            "rolling_mean_6": row.get("rolling_mean_6"),
            "lag": row.get("lag", "{}"),
            "rolling": row.get("rolling", "{}"),
            "weather": row.get("weather", "{}"),
            "holiday": row.get("holiday", "{}"),
            "shap_values": row.get("shap_values", "{}"),
            "hyperparameters": row.get("hyperparameters", "{}"),
            "val_loss_global": row.get("val_loss_global"),
            "val_loss_sku": row.get("val_loss_sku"),
        }

        rows.append({
            "순번": len(rows) + 1,
            "단품코드": str(row["sku"]),
            "단품명": row.get("sku_name", str(row["sku"])),
            "단위": "EA",
            "의뢰수량": 0,
            "예측값(p50)": int(row["p50"]) if pd.notna(row.get("p50")) else 0,
            "하한값(p10)": int(row["p10"]) if pd.notna(row.get("p10")) else 0,
            "상한값(p90)": int(row["p90"]) if pd.notna(row.get("p90")) else 0,
            "주요 영향 변수": top_3_str,
            "비고": "",
            "_row_data": row_data,
        })

    return pd.DataFrame(rows)


def get_predictions_df(
    store: str,
    date_t: str,
    horizon: int
) -> pd.DataFrame:
    """
    대시보드용 예측 DataFrame 반환 (통합 함수)

    Parameters
    ----------
    store : str
        점포 코드
    date_t : str
        기준일 (YYYY-MM-DD)
    horizon : int
        예측 horizon (1~4)

    Returns
    -------
    pd.DataFrame
        대시보드 표시용 DataFrame
    """
    df = get_predictions(store, date_t, horizon)

    if df is None or df.empty:
        return pd.DataFrame()

    return transform_to_display_df(df)


def get_latest_date(store: str) -> Optional[str]:
    """
    해당 점포의 가장 최근 date_t 반환

    Parameters
    ----------
    store : str
        점포 코드

    Returns
    -------
    Optional[str]
        최신 날짜 (YYYY-MM-DD) 또는 None
    """
    dates = get_available_dates(store)
    return dates[0] if dates else None


def check_data_exists(store: str) -> bool:
    """
    해당 점포의 데이터 존재 여부 확인

    Parameters
    ----------
    store : str
        점포 코드

    Returns
    -------
    bool
        데이터 존재 여부
    """
    csv_path = OUTPUTS_DIR / f"{store}.csv"
    return csv_path.exists()