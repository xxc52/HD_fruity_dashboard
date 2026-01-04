"""
Admin Model Info Component
===========================
모델 정보 패널
- Best Model
- 튜닝 날짜
- 학습 날짜
- 학습 샘플 수
- 하이퍼파라미터 (토글) - best model만
- SKU별 val_loss (SKU 선택 시)
"""

import streamlit as st
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent


def get_model_info(store: str) -> Dict[str, Any]:
    """
    점포별 모델 정보 조회

    Returns:
        {
            'best_model': str,
            'last_tuning': str,
            'last_fitting': str,
            'n_train_samples': int,
            'tuning_params': dict,
            'val_loss': float
        }
    """
    model_dir = PROJECT_ROOT / "trained_models" / store / "pareto"

    info = {
        'store': store,
        'best_model': None,
        'last_tuning': None,
        'last_fitting': None,
        'n_train_samples': None,
        'tuning_params': None,
        'val_loss': None,
    }

    # Best Model 정보 (best_model_selection.json에서 모두 가져옴)
    best_model_file = model_dir / "best_model_selection.json"
    if best_model_file.exists():
        try:
            with open(best_model_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                info['best_model'] = data.get('model_name') or data.get('model_type')
                info['val_loss'] = data.get('pinball_loss') or data.get('val_loss')
                info['last_tuning'] = data.get('tuning_date')
                info['tuning_params'] = data.get('params')
        except Exception:
            pass

    # 학습 정보 (best_model과 일치하는 metadata 파일 사용)
    fitted_dir = model_dir / "global" / "fitted" / "latest"
    if fitted_dir.exists():
        try:
            # best_model 타입 추출 (xgboost_global → xgboost)
            best_model_type = None
            if info['best_model']:
                best_model_type = info['best_model'].split('_')[0].lower()

            # best_model에 해당하는 metadata 파일 찾기
            metadata_files = list(fitted_dir.glob("*_metadata.json"))
            target_metadata = None

            if best_model_type and metadata_files:
                # best_model과 일치하는 파일 우선
                for mf in metadata_files:
                    if best_model_type in mf.name.lower():
                        target_metadata = mf
                        break

            # 없으면 가장 최근 파일 사용 (saved_at 기준)
            if not target_metadata and metadata_files:
                latest_saved_at = None
                for mf in metadata_files:
                    try:
                        with open(mf, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            saved_at = data.get('saved_at', '')
                            if not latest_saved_at or saved_at > latest_saved_at:
                                latest_saved_at = saved_at
                                target_metadata = mf
                    except Exception:
                        continue

            if target_metadata:
                with open(target_metadata, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    info['last_fitting'] = metadata.get('fitting_date')
                    info['n_train_samples'] = metadata.get('n_train_samples')
        except Exception:
            pass

    return info


def get_sku_val_loss(store: str, sku: str) -> Optional[float]:
    """
    특정 SKU의 val_loss_sku 조회 (outputs CSV에서)

    Args:
        store: 점포 코드
        sku: SKU 코드

    Returns:
        val_loss_sku 값 또는 None
    """
    outputs_path = PROJECT_ROOT / "outputs" / f"{store}.csv"

    if not outputs_path.exists():
        return None

    try:
        df = pd.read_csv(outputs_path, usecols=['sku', 'val_loss_sku'])
        df['sku'] = df['sku'].astype(str)

        sku_data = df[df['sku'] == sku]
        if not sku_data.empty and 'val_loss_sku' in sku_data.columns:
            val_loss = sku_data['val_loss_sku'].dropna().iloc[0] if not sku_data['val_loss_sku'].isna().all() else None
            return val_loss
        return None
    except Exception:
        return None


def render_model_info(store: str):
    """모델 정보 패널 렌더링"""
    info = get_model_info(store)

    st.markdown("### 모델 정보")

    # Best Model
    if info['best_model']:
        model_display = info['best_model'].replace('_', ' ').title()
        st.success(f"**Best Model**: {model_display}")
    else:
        st.warning("모델 미설정")

    st.markdown("---")

    # 날짜 정보
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**튜닝 날짜**")
        st.markdown(f"📅 {info['last_tuning'] or 'N/A'}")

    with col2:
        st.markdown("**학습 날짜**")
        st.markdown(f"📅 {info['last_fitting'] or 'N/A'}")

    st.markdown("---")

    # 학습 정보
    if info['n_train_samples']:
        st.markdown(f"**학습 샘플 수**: {info['n_train_samples']:,}개")

    # Global Validation Loss
    if info['val_loss']:
        st.markdown(f"**Global Val Loss (Pinball)**: {info['val_loss']:.4f}")

    # SKU별 Validation Loss (SKU 선택 시)
    selected_sku = st.session_state.get('selected_sku_for_model_info', '전체 합계')

    if selected_sku != '전체 합계':
        # 형식: "SKU코드-SKU명(2주평균)"
        sku_code = selected_sku.split('-')[0]
        sku_val_loss = get_sku_val_loss(store, sku_code)

        if sku_val_loss is not None:
            st.markdown(f"**SKU Val Loss (Pinball)**: {sku_val_loss:.4f}")
            st.caption(f"선택된 SKU: {selected_sku}")

    # 하이퍼파라미터 (토글) - Best Model 파라미터만
    if info['tuning_params']:
        with st.expander("🔧 하이퍼파라미터"):
            # 주요 파라미터만 표시
            display_params = {}
            for key, value in info['tuning_params'].items():
                # 불필요한 파라미터 제외
                if key not in ['random_state', 'device', 'n_jobs']:
                    if isinstance(value, float):
                        display_params[key] = round(value, 6)
                    else:
                        display_params[key] = value

            st.json(display_params)