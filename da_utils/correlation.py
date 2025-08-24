# 'correlation.py'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.feature_selection import mutual_info_regression

import datetime as dt

# 상관관계 분석
from scipy.stats import pearsonr, spearmanr, kendalltau

import sys
import os
sys.path.append('/Users/jaehyuntak/Desktop/pjt-data-analysis')

# 한글 폰트 설정
from da_utils.font import setup_korean_font

# from scipy import stats # kendalltau, pearsonr, spearmanr 등은 scipy.stats에서 직접 임포트
# from sklearn.feature_selection import mutual_info_regression




# ----

# da_utils/correlation.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Note: scipy.stats.kendalltau는 analyze_kendall_tau 함수에서 사용될 예정입니다.
# sklearn.feature_selection.mutual_info_regression는 analyze_mutual_information 함수에서 사용될 예정입니다.
# 현재 analyze_basic_correlations 함수에서는 이들을 직접 사용하지 않습니다.

def analyze_basic_correlations(df: pd.DataFrame, numeric_cols: list, strong_corr_threshold: float = 0.5):
    """
    주어진 수치형 컬럼들에 대해 피어슨, 스피어만 상관관계 및 그 차이를 히트맵으로 시각화합니다.
    추가적으로 강한 상관관계(절대값 > strong_corr_threshold)만을 표시한 히트맵을 그립니다.

    Args:
        df (pd.DataFrame): 분석할 데이터프레임.
        numeric_cols (list): 상관관계 분석에 사용할 수치형 컬럼 이름 리스트.
        strong_corr_threshold (float, optional): 강한 상관관계를 판단하는 임계값. 기본값은 0.5.
    """
    print('=== 기본 상관관계 분석 시작 ===')

    # 필요한 컬럼만 선택하고, 없는 컬럼은 경고 후 제외
    available_numeric_cols = [col for col in numeric_cols if col in df.columns]
    missing_cols = [col for col in numeric_cols if col not in df.columns]
    if missing_cols:
        print(f"경고: 다음 컬럼들이 데이터프레임에 없습니다: {missing_cols}. 이 컬럼들은 분석에서 제외됩니다.")
    
    if not available_numeric_cols:
        print("오류: 상관관계 분석에 사용할 수 있는 수치형 컬럼이 없습니다.")
        return

    correlation_data = df[available_numeric_cols]

    # 피어슨 상관관계 계산
    pearson_corr = correlation_data.corr(method='pearson')
    # 스피어만 상관관계 계산
    spearman_corr = correlation_data.corr(method='spearman')

    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    a1, a2, a3, a4 = axes.ravel() # axes[0,0], axes[0,1], axes[1,0], axes[1,1] 대신 ravel() 사용

    # 피어슨 상관관계 히트맵
    sns.heatmap(pearson_corr, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, square=True, ax=a1, cbar_kws={'label': 'Pearson 상관계수'})
    a1.set_title('피어슨 상관관계 (선형 관계)', fontsize=14)

    # 스피어만 상관관계 히트맵
    sns.heatmap(spearman_corr, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, square=True, ax=a2, cbar_kws={'label': 'Spearman 상관계수'})
    a2.set_title('스피어만 상관관계 (순위 기반)', fontsize=14)

    # 피어슨 vs 스피어만 차이 히트맵 (비선형성 지표)
    corr_diff = abs(spearman_corr - pearson_corr)
    sns.heatmap(corr_diff, annot=True, fmt='.2f', cmap='Reds',
                square=True, ax=a3, cbar_kws={'label': '|차이|'})
    a3.set_title('피어슨 vs 스피어만 차이 (비선형성 지표)', fontsize=14)

    # 강한 상관관계 네트워크
    strong_corr = pearson_corr.copy()
    strong_corr[abs(strong_corr) < strong_corr_threshold] = 0
    np.fill_diagonal(strong_corr.values, 0) # 자기 자신과의 상관관계는 0으로 처리

    sns.heatmap(strong_corr, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, square=True, ax=a4, cbar_kws={'label': '강한 상관관계'})
    a4.set_title(f'강한 상관관계 (|r| > {strong_corr_threshold})', fontsize=14)

    plt.tight_layout()
    plt.show()
    print('=== 기본 상관관계 분석 완료 ===')
    
    # 기능 명세서에 따라 DataFrame들을 반환
    return pearson_corr, spearman_corr, corr_diff, strong_corr



# analyze_basic_correlations 함수 코드

def analyze_kendall_tau(df: pd.DataFrame, key_vars: list, p_value_threshold: float = 0.05, tau_threshold: float = 0.1) -> dict:
    """
    주어진 핵심 변수들에 대해 켄달 타우 상관계수를 분석합니다.
    유의미한 (p_value < p_value_threshold) 관계 중 tau 절대값이 tau_threshold 이상인 결과만 반환합니다.

    Args:
        df (pd.DataFrame): 분석할 데이터프레임.
        key_vars (list): 켄달 타우 분석에 사용할 변수 이름 리스트.
        p_value_threshold (float, optional): p-value 유의미성 임계값. 기본값은 0.05.
        tau_threshold (float, optional): tau 값의 절대값 임계값. 기본값은 0.1.

    Returns:
        dict: 유의미한 켄달 타우 상관관계 결과 (예: {'var1 vs var2': {'tau': 0.X, 'p_value': 0.Y}}).
    """
    print('\n=== 켄달 타우 상관계수 분석 시작 ===')
    
    # 필요한 컬럼만 선택하고, 없는 컬럼은 경고 후 제외
    available_key_vars = [col for col in key_vars if col in df.columns]
    missing_cols = [col for col in key_vars if col not in df.columns]
    if missing_cols:
        print(f"경고: 다음 핵심 변수 컬럼들이 데이터프레임에 없습니다: {missing_cols}. 이 컬럼들은 분석에서 제외됩니다.")
    
    if len(available_key_vars) < 2:
        print("오류: 켄달 타우 분석에 사용할 수 있는 핵심 변수가 2개 미만입니다.")
        return {}

    analysis_data = df[available_key_vars].copy()
    kendall_results = {}

    for idx, var1 in enumerate(available_key_vars):
        for var2 in available_key_vars[idx+1:]:
            tau, p_value = kendalltau(analysis_data[var1], analysis_data[var2])
            if p_value < p_value_threshold and abs(tau) > tau_threshold:
                kendall_results[f'{var1} vs {var2}'] = {'tau': tau, 'p_value': p_value}

    if not kendall_results:
        print(f"  유의미한 켄달 타우 상관관계를 찾지 못했습니다 (p<{p_value_threshold}, |tau|>{tau_threshold}).")
    else:
        print("  유의미한 켄달 타우 상관관계 결과:")
        for rel, stats in kendall_results.items():
            print(f'    {rel} = {stats["tau"]:.3f} (p={stats["p_value"]:.3f})')
    
    print('=== 켄달 타우 상관계수 분석 완료 ===')
    return kendall_results


# ... (analyze_basic_correlations, analyze_kendall_tau 함수 코드) ...

def analyze_mutual_information(df: pd.DataFrame, target_col: str, key_vars: list = None, random_state: int = 42) -> pd.DataFrame:
    """
    주어진 타겟 컬럼에 대한 다른 변수들의 상호 정보량(Mutual Information)을 계산합니다.
    상호 정보량은 선형/비선형 관계 구분 없이 변수 간 정보 공유 정도를 측정합니다.

    Args:
        df (pd.DataFrame): 분석할 데이터프레임.
        target_col (str): 상호 정보량 계산의 타겟 변수 이름.
        key_vars (list, optional): MI 분석에 사용할 특정 변수 이름 리스트. None이면 target_col을 제외한 모든 수치형 컬럼 사용.
        random_state (int, optional): 재현성을 위한 난수 시드. 기본값은 42.

    Returns:
        pd.DataFrame: 각 Feature와 Target 간의 MI_Score를 담은 DataFrame, MI_Score 기준 내림차순 정렬.
    """
    print('\n=== 상호 정보량 기반 연관성 분석 시작 ===')
    
    if target_col not in df.columns:
        print(f"오류: 타겟 컬럼 '{target_col}'이 데이터프레임에 존재하지 않습니다.")
        return pd.DataFrame(columns=['Feature', 'MI_Score'])

    analysis_data = df.copy()
    target = analysis_data[target_col]

    if key_vars:
        # key_vars가 제공되면 해당 변수들만 사용
        available_features = [col for col in key_vars if col in analysis_data.columns and col != target_col]
        missing_features = [col for col in key_vars if col not in analysis_data.columns or col == target_col]
        if missing_features:
            print(f"경고: 다음 Feature 컬럼들이 데이터프레임에 없거나 타겟 컬럼과 동일합니다: {missing_features}. 이 컬럼들은 분석에서 제외됩니다.")
        features = analysis_data[available_features]
    else:
        # key_vars가 없으면 target_col을 제외한 모든 수치형 컬럼 사용
        numeric_cols = analysis_data.select_dtypes(include=np.number).columns.tolist()
        features = analysis_data[numeric_cols].drop(columns=[target_col], errors='ignore')

    if features.empty:
        print(f"오류: 타겟 컬럼 '{target_col}'을 제외한 분석할 Feature가 없습니다.")
        return pd.DataFrame(columns=['Feature', 'MI_Score'])

    mi_scores = mutual_info_regression(features, target, random_state=random_state)
    mi_results = pd.DataFrame({
        'Feature': features.columns,
        'MI_Score': mi_scores
    }).sort_values('MI_Score', ascending=False).reset_index(drop=True)

    print(f"  (타겟: {target_col} 기준)")
    for _, row in mi_results.iterrows():
        print(f'  {row["Feature"]}: {row["MI_Score"]:.3f}')
    
    print('=== 상호 정보량 기반 연관성 분석 완료 ===')
    return mi_results


def detect_nonlinear_patterns(
    df: pd.DataFrame,
    u_shape_x_col: str,
    u_shape_y_col: str,
    saturation_x_col: str,
    saturation_y_col: str,
    exponential_x_col: str,
    exponential_y_col: str,
    scatter_x_col: str,
    scatter_y_col: str,
    high_value_segment_col: str,
    high_value_analysis_cols: list,
    q_segments: int = 10,
    quantile_threshold: float = 0.9
):
    """
    다양한 비선형 패턴(U자형, 포화점, 지수적 관계 등)을 탐지하고 시각화합니다.
    각 패턴 탐지에 필요한 컬럼명은 인자로 받아 범용성을 확보합니다.
    
    Args:
        df (pd.DataFrame): 분석할 데이터프레임.
        u_shape_x_col (str): U자형 관계 탐지 시 X축으로 사용할 컬럼.
        u_shape_y_col (str): U자형 관계 탐지 시 Y축으로 사용할 컬럼.
        saturation_x_col (str): 포화점 탐지 시 X축으로 사용할 컬럼.
        saturation_y_col (str): 포화점 탐지 시 Y축으로 사용할 컬럼.
        exponential_x_col (str): 지수적 관계 탐지 시 X축으로 사용할 컬럼.
        exponential_y_col (str): 지수적 관계 탐지 시 Y축으로 사용할 컬럼.
        scatter_x_col (str): 일반 산점도 X축 컬럼.
        scatter_y_col (str): 일반 산점도 Y축 컬럼.
        high_value_segment_col (str): 특정 세그먼트(예: 상위 10%)를 정의할 기준 컬럼.
        high_value_analysis_cols (list): 상위 세그먼트의 특성을 분석할 컬럼 리스트.
        q_segments (int, optional): U자형/포화점 탐지 시 데이터를 나눌 분위수. 기본값은 10.
        quantile_threshold (float, optional): 상위 세그먼트 기준 분위수. 기본값은 0.9 (상위 10%).
    """
    print('\n=== 비선형 패턴 심화 탐지 시작 ===')
    customer_stats_copy = df.copy() # 원본 데이터프레임 변경 방지를 위해 복사

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    a1, a2, a3, a4 = axes.ravel()

    # 모든 필수 컬럼이 존재하는지 사전 확인
    required_cols = [
        u_shape_x_col, u_shape_y_col,
        saturation_x_col, saturation_y_col,
        exponential_x_col, exponential_y_col,
        scatter_x_col, scatter_y_col,
        high_value_segment_col
    ] + high_value_analysis_cols
    
    missing_cols_overall = [col for col in required_cols if col not in customer_stats_copy.columns]
    if missing_cols_overall:
        print(f"오류: 다음 필수 컬럼들이 데이터프레임에 없습니다: {missing_cols_overall}. 비선형 패턴 분석을 진행할 수 없습니다.")
        plt.close(fig) # 빈 플롯 생성 방지
        return

    # 1. U자형 관계 탐지 (예: 건축년도 vs 단위면적당거래금액)
    if u_shape_x_col in customer_stats_copy.columns and u_shape_y_col in customer_stats_copy.columns:
        try:
            customer_stats_copy[f'{u_shape_x_col}_Segment'] = pd.qcut(
                customer_stats_copy[u_shape_x_col], q=q_segments, labels=False, duplicates='drop'
            )
            u_shape_analysis = customer_stats_copy.groupby(f'{u_shape_x_col}_Segment').agg({
                u_shape_y_col: 'mean',
                u_shape_x_col: 'mean'
            }).round(2)
            a1.plot(u_shape_analysis[u_shape_x_col], u_shape_analysis[u_shape_y_col], marker='o', linewidth=2, markersize=8)
            a1.set_xlabel(f'{u_shape_x_col} 평균')
            a1.set_ylabel(f'{u_shape_y_col} 평균')
            a1.set_title(f'{u_shape_x_col} vs {u_shape_y_col} (U자형 탐지)', fontsize=14)
            a1.grid(True, alpha=0.3)
        except Exception as e:
            a1.set_title(f'{u_shape_x_col} vs {u_shape_y_col} (분석 오류)', fontsize=14)
            a1.text(0.5, 0.5, f'오류: {e}', horizontalalignment='center', verticalalignment='center', transform=a1.transAxes)
    else:
        a1.set_title(f'{u_shape_x_col} vs {u_shape_y_col} (데이터 부족)', fontsize=14)
        a1.text(0.5, 0.5, '필요 컬럼 없음', horizontalalignment='center', verticalalignment='center', transform=a1.transAxes)


    # 2. 포화점 탐지 (예: 전용면적 vs 거래금액)
    if saturation_x_col in customer_stats_copy.columns and saturation_y_col in customer_stats_copy.columns:
        try:
            customer_stats_copy[f'{saturation_x_col}_Segment'] = pd.qcut(
                customer_stats_copy[saturation_x_col], q=q_segments, labels=False, duplicates='drop'
            )
            saturation_analysis = customer_stats_copy.groupby(f'{saturation_x_col}_Segment').agg({
                saturation_y_col: 'mean',
                saturation_x_col: 'mean'
            }).round(2)
            a2.plot(saturation_analysis[saturation_x_col], saturation_analysis[saturation_y_col], marker='s', linewidth=2, markersize=8)
            a2.set_xlabel(f'{saturation_x_col} 평균')
            a2.set_ylabel(f'{saturation_y_col} 평균')
            a2.set_title(f'{saturation_x_col} vs {saturation_y_col} (포화점 탐지)', fontsize=14)
            a2.grid(True, alpha=0.3)
        except Exception as e:
            a2.set_title(f'{saturation_x_col} vs {saturation_y_col} (분석 오류)', fontsize=14)
            a2.text(0.5, 0.5, f'오류: {e}', horizontalalignment='center', verticalalignment='center', transform=a2.transAxes)
    else:
        a2.set_title(f'{saturation_x_col} vs {saturation_y_col} (데이터 부족)', fontsize=14)
        a2.text(0.5, 0.5, '필요 컬럼 없음', horizontalalignment='center', verticalalignment='center', transform=a2.transAxes)


    # 3. 지수적 관계 확인 (예: 건축경과년수 vs 단위면적당거래금액)
    if exponential_x_col in customer_stats_copy.columns and exponential_y_col in customer_stats_copy.columns:
        try:
            # 유효한 데이터 필터링 (0 또는 음수 값 제외)
            valid_data = customer_stats_copy[
                (customer_stats_copy[exponential_x_col] > 0) & 
                (customer_stats_copy[exponential_y_col] > 0)
            ].copy()
            
            if not valid_data.empty:
                a3.scatter(valid_data[exponential_x_col], valid_data[exponential_y_col], alpha=0.6, s=30)
                a3.set_xlabel(exponential_x_col)
                a3.set_ylabel(exponential_y_col)

                # 로그 변환 선형화 및 상관계수 비교
                log_x = np.log1p(valid_data[exponential_x_col])
                log_y = np.log1p(valid_data[exponential_y_col])
                original_corr = valid_data[exponential_x_col].corr(valid_data[exponential_y_col])
                log_corr = log_x.corr(log_y)
                a3.set_title(f'{exponential_x_col} vs {exponential_y_col}\n원본r={original_corr:.2f}, 로그변환r={log_corr:.2f}', fontsize=14)
                a3.grid(True, alpha=0.3)
            else:
                a3.set_title(f'{exponential_x_col} vs {exponential_y_col} (유효 데이터 부족)', fontsize=14)
                a3.text(0.5, 0.5, '유효한 양수 데이터 없음', horizontalalignment='center', verticalalignment='center', transform=a3.transAxes)
        except Exception as e:
            a3.set_title(f'{exponential_x_col} vs {exponential_y_col} (분석 오류)', fontsize=14)
            a3.text(0.5, 0.5, f'오류: {e}', horizontalalignment='center', verticalalignment='center', transform=a3.transAxes)
    else:
        a3.set_title(f'{exponential_x_col} vs {exponential_y_col} (데이터 부족)', fontsize=14)
        a3.text(0.5, 0.5, '필요 컬럼 없음', horizontalalignment='center', verticalalignment='center', transform=a3.transAxes)


    # 4. 일반 산점도 및 특정 세그먼트 특성 분석 (예: 층 vs 거래금액)
    if scatter_x_col in customer_stats_copy.columns and scatter_y_col in customer_stats_copy.columns:
        try:
            a4.scatter(customer_stats_copy[scatter_x_col], customer_stats_copy[scatter_y_col], alpha=0.6, s=30, color='purple')
            a4.set_xlabel(scatter_x_col)
            a4.set_ylabel(scatter_y_col)
            a4.set_title(f'{scatter_x_col} vs {scatter_y_col}', fontsize=14)
            a4.grid(True, alpha=0.3)

            # 특정 세그먼트 (예: 상위 10%) 고객 특성 출력
            if high_value_segment_col in customer_stats_copy.columns and all(col in customer_stats_copy.columns for col in high_value_analysis_cols):
                high_segment = customer_stats_copy[high_value_segment_col] > customer_stats_copy[high_value_segment_col].quantile(quantile_threshold)
                print(f"\n  {high_value_segment_col} 상위 {(1-quantile_threshold)*100:.0f}% 고객 특성:")
                if not customer_stats_copy[high_segment].empty:
                    for col in high_value_analysis_cols:
                        if pd.api.types.is_numeric_dtype(customer_stats_copy[col]):
                            print(f"  - 평균 {col}: {customer_stats_copy[high_segment][col].mean():,.1f}")
                        else:
                            # 범주형 컬럼의 경우 최빈값 또는 상위 N개 출력
                            top_values = customer_stats_copy[high_segment][col].value_counts(normalize=True).head(3)
                            if not top_values.empty:
                                print(f"  - {col} (상위): {top_values.index.tolist()} (비율: {top_values.values.tolist()})")
                            else:
                                print(f"  - {col}: 데이터 없음")
                else:
                    print("  - 해당 조건의 고객이 없습니다.")
            else:
                print(f"경고: {high_value_segment_col} 또는 {high_value_analysis_cols} 중 일부 컬럼이 없어 특정 세그먼트 분석을 건너뜁니다.")
        except Exception as e:
            a4.set_title(f'{scatter_x_col} vs {scatter_y_col} (분석 오류)', fontsize=14)
            a4.text(0.5, 0.5, f'오류: {e}', horizontalalignment='center', verticalalignment='center', transform=a4.transAxes)
    else:
        a4.set_title(f'{scatter_x_col} vs {scatter_y_col} (데이터 부족)', fontsize=14)
        a4.text(0.5, 0.5, '필요 컬럼 없음', horizontalalignment='center', verticalalignment='center', transform=a4.transAxes)

    plt.tight_layout()
    plt.show()
    print('=== 비선형 패턴 심화 탐지 완료 ===')


    # 시계열 분석



def analyze_time_series_trend(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    resample_freq: str = 'M', # 'D', 'W', 'M', 'Q', 'Y' 등
    rolling_window: int = None
):
    """
    주어진 날짜 컬럼과 수치형 값 컬럼을 사용하여 시간 경과에 따른 데이터의 추세를 시각화합니다.
    데이터를 일별, 주별, 월별, 분기별, 연도별 등으로 집계하고, 이동 평균선을 추가할 수 있습니다.

    Args:
        df (pd.DataFrame): 분석할 데이터프레임.
        date_col (str): 날짜/시간 정보를 담고 있는 컬럼명. datetime 타입이어야 합니다.
        value_col (str): 시간 흐름에 따라 분석할 수치형 컬럼명.
        resample_freq (str, optional): 데이터를 집계할 시간 간격.
                                       'D' (일별), 'W' (주별), 'M' (월별), 'Q' (분기별), 'Y' (연도별) 등.
                                       기본값은 'M'.
        rolling_window (int, optional): 이동 평균을 계산할 윈도우 크기.
                                        None이면 이동 평균을 그리지 않습니다. 기본값은 None.
    """
    print(f'\n=== 시계열 추세 분석 시작 ({value_col} by {date_col}, {resample_freq} 단위) ===')

    # 필수 컬럼 존재 여부 확인
    if date_col not in df.columns:
        print(f"오류: 날짜 컬럼 '{date_col}'이 데이터프레임에 존재하지 않습니다.")
        return
    if value_col not in df.columns:
        print(f"오류: 값 컬럼 '{value_col}'이 데이터프레임에 존재하지 않습니다.")
        return

    # 날짜 컬럼이 datetime 타입인지 확인 및 변환
    # .copy()를 사용하여 원본 df 변경 방지
    temp_df = df.copy() 
    if not pd.api.types.is_datetime64_any_dtype(temp_df[date_col]):
        try:
            temp_df[date_col] = pd.to_datetime(temp_df[date_col])
            print(f"정보: '{date_col}' 컬럼을 datetime 타입으로 변환했습니다.")
        except Exception as e:
            print(f"오류: '{date_col}' 컬럼을 datetime 타입으로 변환할 수 없습니다. {e}")
            return
            
    # 값 컬럼이 수치형인지 확인
    if not pd.api.types.is_numeric_dtype(temp_df[value_col]):
        print(f"오류: 값 컬럼 '{value_col}'이 수치형이 아닙니다.")
        return

    # 날짜 컬럼에 NaN이 있는 행 제거 (시계열 분석의 정확성을 위해)
    analysis_df = temp_df[[date_col, value_col]].dropna(subset=[date_col]).set_index(date_col)
    
    if analysis_df.empty:
        print(f"오류: '{date_col}' 컬럼에 유효한 날짜 데이터가 없어 시계열 분석을 진행할 수 없습니다.")
        return

    # 시계열 데이터 집계
    resampled_data = analysis_df[value_col].resample(resample_freq).mean()

    # 이동 평균 계산
    if rolling_window is not None and rolling_window > 0:
        if len(resampled_data) < rolling_window:
            print(f"경고: 리샘플링된 데이터 포인트({len(resampled_data)}개)가 이동 평균 윈도우({rolling_window}개)보다 적습니다. 이동 평균을 계산할 수 없습니다.")
            rolling_mean = None
        else:
            rolling_mean = resampled_data.rolling(window=rolling_window).mean()
    else:
        rolling_mean = None

    # 시각화
    plt.figure(figsize=(14, 7))
    plt.plot(resampled_data.index, resampled_data.values, label=f'{value_col} ({resample_freq} 평균)', marker='o', markersize=4, linestyle='-')
    if rolling_mean is not None:
        plt.plot(rolling_mean.index, rolling_mean.values, label=f'{rolling_window} {resample_freq} 이동 평균', color='red', linestyle='--')

    plt.title(f'{value_col}의 시간 경과에 따른 추세', fontsize=16)
    plt.xlabel('날짜', fontsize=12)
    plt.ylabel(value_col, fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()
    
    print('=== 시계열 추세 분석 완료 ===')





# 월별/연별 성장률을 계산하고 시각화
    

# ----

# da_utils/correlation.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import kendalltau
from sklearn.feature_selection import mutual_info_regression

# analyze_basic_correlations 함수는 여기에 이미 정의되어 있다고 가정합니다.
# analyze_kendall_tau 함수는 여기에 이미 정의되어 있다고 가정합니다.
# analyze_mutual_information 함수는 여기에 이미 정의되어 있다고 가정합니다.
# detect_nonlinear_patterns 함수는 여기에 이미 정의되어 있다고 가정합니다.
# analyze_time_series_trend 함수는 여기에 이미 정의되어 있다고 가정합니다.
# ... (이전 함수 코드들) ...

def analyze_growth_rates(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    group_by_cols: list = None,
    summary_window: int = 12,
    num_top_groups: int = None,    # New: 상위 N개 그룹만 플로팅
    num_bottom_groups: int = None  # New: 하위 N개 그룹만 플로팅
):
    """
    주어진 날짜 컬럼과 수치형 값 컬럼을 사용하여 월별/연별 성장률(MoM, YoY)을 계산하고 시각화합니다.
    이를 통해 단기 모멘텀과 계절성을 배제한 장기 트렌드를 파악할 수 있습니다.
    그룹핑 컬럼이 지정된 경우, 상위/하위 N개 그룹만 선택하여 시각화할 수 있습니다.
    모든 시각화는 개별 그림(figure)으로 생성됩니다.
    
    Args:
        df (pd.DataFrame): 분석할 데이터프레임.
        date_col (str): 날짜/시간 정보를 담고 있는 컬럼명. datetime 타입이어야 합니다.
        value_col (str): 성장률을 계산할 수치형 컬럼명.
        group_by_cols (list, optional): value_col을 집계하기 전에 추가적으로 그룹핑할 컬럼 리스트.
                                        None이면 전체 데이터에 대해 집계합니다. 기본값은 None.
        summary_window (int, optional): 최근 성장률 요약에 사용할 기간(월). 기본값은 12.
        num_top_groups (int, optional): 그룹핑 시, 평균 value_col 기준으로 상위 N개 그룹만 플로팅.
                                        None이면 상위 그룹 필터링 없음.
        num_bottom_groups (int, optional): 그룹핑 시, 평균 value_col 기준으로 하위 N개 그룹만 플로팅.
                                           None이면 하위 그룹 필터링 없음.
    """
    print(f'\n=== 성장률 분석 시작 ({value_col} by {date_col}) ===')

    # 필수 컬럼 존재 여부 확인
    required_cols = [date_col, value_col]
    if group_by_cols:
        required_cols.extend(group_by_cols)
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"오류: 다음 필수 컬럼들이 데이터프레임에 없습니다: {missing_cols}. 성장률 분석을 진행할 수 없습니다.")
        return

    # 날짜 컬럼이 datetime 타입인지 확인 및 변환
    temp_df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(temp_df[date_col]):
        try:
            temp_df[date_col] = pd.to_datetime(temp_df[date_col])
            print(f"정보: '{date_col}' 컬럼을 datetime 타입으로 변환했습니다.")
        except Exception as e:
            print(f"오류: '{date_col}' 컬럼을 datetime 타입으로 변환할 수 없습니다. {e}")
            return
            
    # 값 컬럼이 수치형인지 확인
    if not pd.api.types.is_numeric_dtype(temp_df[value_col]):
        print(f"오류: 값 컬럼 '{value_col}'이 수치형이 아닙니다.")
        return

    # 월별 집계 데이터 생성
    grouping_keys = []
    if group_by_cols:
        grouping_keys.extend(group_by_cols)
    grouping_keys.append(temp_df[date_col].dt.to_period('M'))

    monthly_data = temp_df.groupby(grouping_keys)[value_col].sum().reset_index()
    monthly_data.rename(columns={temp_df[date_col].dt.to_period('M').name: 'month'}, inplace=True)
    
    if not group_by_cols: # 그룹핑이 없는 경우에만 컬럼명을 'value'로 변경
        monthly_data.columns = ['month', value_col] # value_col 이름을 그대로 사용
    
    monthly_data['month'] = monthly_data['month'].dt.to_timestamp() # Period를 Timestamp로 변환

    # 그룹별 성장률 계산
    if group_by_cols:
        growth_data_list = []
        for name, group in monthly_data.groupby(group_by_cols):
            group = group.sort_values('month').reset_index(drop=True)
            group[f'{value_col}_lag1'] = group[value_col].shift(1)
            group['mom_growth'] = (group[value_col] - group[f'{value_col}_lag1']) / group[f'{value_col}_lag1'] * 100
            
            group[f'{value_col}_lag12'] = group[value_col].shift(12)
            group['yoy_growth'] = (group[value_col] - group[f'{value_col}_lag12']) / group[f'{value_col}_lag12'] * 100
            growth_data_list.append(group)
        growth_data = pd.concat(growth_data_list).reset_index(drop=True)
    else:
        growth_data = monthly_data.sort_values('month').reset_index(drop=True)
        growth_data[f'{value_col}_lag1'] = growth_data[value_col].shift(1)
        growth_data['mom_growth'] = (growth_data[value_col] - growth_data[f'{value_col}_lag1']) / growth_data[f'{value_col}_lag1'] * 100
        
        growth_data[f'{value_col}_lag12'] = growth_data[value_col].shift(12)
        growth_data['yoy_growth'] = (growth_data[value_col] - growth_data[f'{value_col}_lag12']) / growth_data[f'{value_col}_lag12'] * 100

    # --- 그룹 필터링 로직 (상위 N개, 하위 N개) ---
    growth_data_to_plot = growth_data.copy()
    selected_group_names = []

    if group_by_cols and (num_top_groups is not None or num_bottom_groups is not None):
        # 그룹별 평균 value_col 계산하여 랭킹
        group_avg_values = growth_data.groupby(group_by_cols)[value_col].mean().sort_values(ascending=False)
        
        if num_top_groups is not None and num_top_groups > 0:
            top_groups = group_avg_values.head(num_top_groups).index.tolist()
            selected_group_names.extend(top_groups)
            print(f"\n  상위 {num_top_groups}개 그룹: {top_groups}")
        
        if num_bottom_groups is not None and num_bottom_groups > 0:
            bottom_groups = group_avg_values.tail(num_bottom_groups).index.tolist()
            selected_group_names.extend(bottom_groups)
            print(f"  하위 {num_bottom_groups}개 그룹: {bottom_groups}")
        
        # 중복 제거 (상위/하위 그룹이 겹칠 경우)
        selected_group_names = list(set(selected_group_names))

        if selected_group_names:
            # group_by_cols가 하나일 경우와 여러 개일 경우 처리
            if len(group_by_cols) == 1:
                growth_data_to_plot = growth_data[growth_data[group_by_cols[0]].isin(selected_group_names)]
            else:
                # 여러 컬럼으로 그룹핑된 경우 튜플 비교
                selected_group_names_tuples = [tuple(g) for g in selected_group_names]
                growth_data_to_plot = growth_data[
                    growth_data[group_by_cols].apply(tuple, axis=1).isin(selected_group_names_tuples)
                ]
            print(f"  총 {len(selected_group_names)}개 그룹만 시각화합니다.")
        else:
            print("  상위/하위 그룹 필터링 조건에 해당하는 그룹이 없습니다. 전체 그룹을 시각화합니다.")
            growth_data_to_plot = growth_data.copy()
    else:
        print("  그룹 필터링이 요청되지 않았습니다. 모든 그룹을 시각화합니다.")
        growth_data_to_plot = growth_data.copy()


    # 최근 성장률 요약 (플로팅되는 그룹에 대해서만)
    if not growth_data_to_plot.empty:
        if group_by_cols:
            print(f"\n  최근 {summary_window}개월 평균 성장률 (시각화된 그룹별):")
            for name, group in growth_data_to_plot.groupby(group_by_cols):
                avg_mom = group.tail(summary_window)['mom_growth'].mean()
                avg_yoy = group.tail(summary_window)['yoy_growth'].mean()
                group_name_str = ', '.join(map(str, name)) if isinstance(name, tuple) else str(name)
                print(f"  그룹 ({group_name_str}): MoM={avg_mom:.1f}%, YoY={avg_yoy:.1f}%")
        else:
            avg_mom = growth_data_to_plot.tail(summary_window)['mom_growth'].mean()
            avg_yoy = growth_data_to_plot.tail(summary_window)['yoy_growth'].mean()
            print(f'\n  최근 {summary_window}개월 평균 MoM 성장률: {avg_mom:.1f}%')
            print(f'  최근 {summary_window}개월 평균 YoY 성장률: {avg_yoy:.1f}%')
    else:
        print("  성장률을 계산할 데이터가 부족합니다.")

    # --- 시각화 (개별 그림으로 분리) ---
    if not growth_data_to_plot.empty:
        # 1. 월별 값 추이
        plt.figure(figsize=(12, 6))
        if group_by_cols:
            for name, group in growth_data_to_plot.groupby(group_by_cols):
                group_name_str = ', '.join(map(str, name)) if isinstance(name, tuple) else str(name)
                plt.plot(group['month'], group[value_col], marker='o', linewidth=2, label=f'그룹 {group_name_str}')
            plt.legend()
        else:
            plt.plot(growth_data_to_plot['month'], growth_data_to_plot[value_col], marker='o', linewidth=2)
        plt.title(f'월별 {value_col} 추이', fontsize=16)
        plt.ylabel(value_col, fontsize=12)
        plt.xlabel('날짜', fontsize=12)
        plt.tick_params(axis='x', rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # 2. MoM 성장률
        plt.figure(figsize=(12, 6))
        if group_by_cols:
            for name, group in growth_data_to_plot.groupby(group_by_cols):
                group_name_str = ', '.join(map(str, name)) if isinstance(name, tuple) else str(name)
                plt.plot(group['month'], group['mom_growth'], marker='o', linewidth=1, label=f'그룹 {group_name_str}')
            plt.legend()
        else:
            plt.bar(growth_data_to_plot['month'], growth_data_to_plot['mom_growth'], width=15)
        plt.axhline(y=0, color='red', linestyle='-', alpha=0.7)
        plt.title('월간 성장률 (MoM)', fontsize=16)
        plt.ylabel('성장률 (%)', fontsize=12)
        plt.xlabel('날짜', fontsize=12)
        plt.tick_params(axis='x', rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # 3. YoY 성장률
        plt.figure(figsize=(12, 6))
        if group_by_cols:
            for name, group in growth_data_to_plot.groupby(group_by_cols):
                group_name_str = ', '.join(map(str, name)) if isinstance(name, tuple) else str(name)
                plt.plot(group['month'], group['yoy_growth'], marker='o', linewidth=1, label=f'그룹 {group_name_str}')
            plt.legend()
        else:
            plt.bar(growth_data_to_plot['month'], growth_data_to_plot['yoy_growth'], color='lightgreen', width=15)
        plt.axhline(y=0, color='red', linestyle='-', alpha=0.7)
        plt.title('연간 성장률 (YoY)', fontsize=16)
        plt.ylabel('성장률 (%)', fontsize=12)
        plt.xlabel('날짜', fontsize=12)
        plt.tick_params(axis='x', rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # 4. MoM 성장률 분포 (그룹핑이 없는 경우만)
        if not group_by_cols:
            plt.figure(figsize=(12, 6))
            growth_rates_mom = growth_data_to_plot['mom_growth'].dropna()
            if not growth_rates_mom.empty:
                plt.hist(growth_rates_mom, bins=20, alpha=0.7, color='coral', edgecolor='black')
                plt.axvline(x=growth_rates_mom.mean(), color='red', linestyle='--', 
                                    label=f'평균: {growth_rates_mom.mean():.1f}%')
                plt.title('MoM 성장률 분포', fontsize=16)
                plt.xlabel('성장률 (%)', fontsize=12)
                plt.ylabel('빈도', fontsize=12)
                plt.legend()
                plt.grid(True, alpha=0.3)
            else:
                plt.title('MoM 성장률 분포 (데이터 부족)', fontsize=16)
                plt.text(0.5, 0.5, '성장률 데이터 없음', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
            plt.tight_layout()
            plt.show()
        else:
            print("\n  MoM 성장률 분포는 그룹 분석 시 개별 그림으로 생성되지 않습니다.")
    else:
        print("  시각화할 데이터가 부족합니다.")
    
    print('=== 성장률 분석 완료 ===')




#  다양한 기간의 이동 평균을 계산하고 시각화하며, 최근 트렌드 방향성을 분석


def analyze_rolling_trends(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    rolling_windows: list = None, # 일 단위 윈도우 크기 리스트
    trend_summary_windows: list = None # 최근 트렌드 요약 기간 리스트
):
    """
    주어진 날짜 컬럼과 수치형 값 컬럼을 사용하여 다양한 기간의 이동 평균을 계산하고 시각화합니다.
    이를 통해 단기 변동성을 제거하고 장기적인 트렌드 방향성을 파악합니다.

    Args:
        df (pd.DataFrame): 분석할 데이터프레임.
        date_col (str): 날짜/시간 정보를 담고 있는 컬럼명. datetime 타입이어야 합니다.
        value_col (str): 이동 평균을 계산할 수치형 컬럼명.
        rolling_windows (list, optional): 계산할 이동 평균 윈도우 크기 리스트 (일 단위).
                                         기본값은 [7, 30, 90] (주간, 월간, 분기).
        trend_summary_windows (list, optional): 최근 트렌드 방향성을 요약할 기간 리스트 (일 단위).
                                               기본값은 [7, 30].
    """
    print(f'\n=== 이동 평균 트렌드 분석 시작 ({value_col} by {date_col}) ===')

    if rolling_windows is None:
        rolling_windows = [7, 30, 90] # 기본값 설정
    if trend_summary_windows is None:
        trend_summary_windows = [7, 30] # 기본값 설정

    # 필수 컬럼 존재 여부 확인
    if date_col not in df.columns:
        print(f"오류: 날짜 컬럼 '{date_col}'이 데이터프레임에 존재하지 않습니다.")
        return
    if value_col not in df.columns:
        print(f"오류: 값 컬럼 '{value_col}'이 데이터프레임에 존재하지 않습니다.")
        return

    # 날짜 컬럼이 datetime 타입인지 확인 및 변환
    temp_df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(temp_df[date_col]):
        try:
            temp_df[date_col] = pd.to_datetime(temp_df[date_col])
            print(f"정보: '{date_col}' 컬럼을 datetime 타입으로 변환했습니다.")
        except Exception as e:
            print(f"오류: '{date_col}' 컬럼을 datetime 타입으로 변환할 수 없습니다. {e}")
            return
            
    # 값 컬럼이 수치형인지 확인
    if not pd.api.types.is_numeric_dtype(temp_df[value_col]):
        print(f"오류: 값 컬럼 '{value_col}'이 수치형이 아닙니다.")
        return

    # 일별 값 집계
    # 날짜 컬럼에 NaN이 있는 행 제거 후 일별 합계 (또는 평균) 계산
    daily_data = temp_df[[date_col, value_col]].dropna(subset=[date_col])
    daily_data['date_only'] = daily_data[date_col].dt.date # 날짜 부분만 추출
    daily_aggregated = daily_data.groupby('date_only')[value_col].sum().reset_index() # 일별 합계
    
    daily_aggregated.columns = ['date', value_col]
    daily_aggregated['date'] = pd.to_datetime(daily_aggregated['date'])
    daily_aggregated = daily_aggregated.sort_values('date').reset_index(drop=True)

    if daily_aggregated.empty:
        print(f"오류: '{date_col}' 컬럼에 유효한 날짜 데이터가 없어 일별 집계를 수행할 수 없습니다.")
        return

    # 다양한 기간의 이동 평균 계산
    for window in rolling_windows:
        if len(daily_aggregated) >= window:
            daily_aggregated[f'ma_{window}'] = daily_aggregated[value_col].rolling(window=window).mean()
        else:
            print(f"경고: 데이터 포인트({len(daily_aggregated)}개)가 이동 평균 윈도우({window}일)보다 적어 ma_{window}를 계산할 수 없습니다.")

    # 트렌드 방향성 분석
    print("\n📈 트렌드 방향성 분석 결과:")
    for window in trend_summary_windows:
        ma_col = f'ma_{window}'
        if ma_col in daily_aggregated.columns:
            # 이동 평균의 변화량 (diff) 계산
            daily_aggregated[f'trend_{window}'] = daily_aggregated[ma_col].diff()
            
            # 최근 트렌드 요약
            recent_trend_values = daily_aggregated[f'trend_{window}'].tail(window).dropna()
            if not recent_trend_values.empty:
                recent_trend_mean = recent_trend_values.mean()
                direction = '상승' if recent_trend_mean > 0 else '하락' if recent_trend_mean < 0 else '유지'
                print(f"  최근 {window}일 이동평균 트렌드: {direction} ({recent_trend_mean:.2f})")
            else:
                print(f"  최근 {window}일 트렌드 분석을 위한 데이터가 부족합니다.")
        else:
            print(f"  {window}일 이동평균이 계산되지 않아 트렌드 분석을 건너뜁니다.")

    # 트렌드 시각화
    plt.figure(figsize=(15, 8))

    plt.plot(daily_aggregated['date'], daily_aggregated[value_col], alpha=0.3, color='gray', label='일별 값')
    
    colors = ['blue', 'red', 'green', 'purple', 'orange'] # 이동평균선 색상
    for i, window in enumerate(rolling_windows):
        ma_col = f'ma_{window}'
        if ma_col in daily_aggregated.columns:
            plt.plot(daily_aggregated['date'], daily_aggregated[ma_col], 
                     color=colors[i % len(colors)], 
                     label=f'{window}일 이동평균', 
                     linewidth=2 if window == rolling_windows[0] else 1.5) # 첫 번째 MA만 두껍게

    plt.title(f'{value_col} 트렌드 분석 (이동평균)', fontsize=16)
    plt.xlabel('날짜', fontsize=12)
    plt.ylabel(value_col, fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    print('=== 이동 평균 트렌드 분석 완료 ===')



#  통합 실행 함수

def run_full_correlation_analysis(
    df: pd.DataFrame,
    basic_corr_params: dict,
    advanced_corr_params: dict,
    nonlinear_patterns_params: dict,
    time_series_trend_params: dict,
    growth_rates_params: dict,
    rolling_trends_params: dict
):
    """
    고객 데이터에 대한 포괄적인 상관관계 및 비선형 패턴, 시계열 추세, 성장률, 이동 평균 분석을 실행합니다.
    각 분석 모듈에 필요한 파라미터는 딕셔너리 형태로 전달받아 범용성을 확보합니다.

    Args:
        df (pd.DataFrame): 분석할 데이터프레임.
        basic_corr_params (dict): analyze_basic_correlations 함수에 전달할 파라미터 딕셔너리.
                                  필수 키: 'numeric_cols'.
        advanced_corr_params (dict): analyze_kendall_tau 및 analyze_mutual_information 함수에 전달할 파라미터 딕셔너리.
                                     필수 키: 'key_vars', 'mi_target_col'.
        nonlinear_patterns_params (dict): detect_nonlinear_patterns 함수에 전달할 파라미터 딕셔너리.
                                          필수 키: 'u_shape_x_col', 'u_shape_y_col', 'saturation_x_col', 'saturation_y_col',
                                          'exponential_x_col', 'exponential_y_col', 'scatter_x_col', 'scatter_y_col',
                                          'high_value_segment_col', 'high_value_analysis_cols'.
        time_series_trend_params (dict): analyze_time_series_trend 함수에 전달할 파라미터 딕셔너리.
                                         필수 키: 'date_col', 'value_col'.
        growth_rates_params (dict): analyze_growth_rates 함수에 전달할 파라미터 딕셔너리.
                                    필수 키: 'date_col', 'value_col'.
        rolling_trends_params (dict): analyze_rolling_trends 함수에 전달할 파라미터 딕셔너리.
                                      필수 키: 'date_col', 'value_col'.
    """
    print("=== 고객 행동 데이터 심층 분석 시작 ===")

    # 1. 기본 상관관계 분석 및 시각화
    print("\n--- [1] 기본 상관관계 분석 ---")
    analyze_basic_correlations(df=df, **basic_corr_params)

    # 2. 고급 상관관계 분석 (켄달 타우, 상호 정보량)
    print("\n--- [2] 고급 상관관계 분석 ---")
    # 켄달 타우
    kendall_results = analyze_kendall_tau(
        df=df,
        key_vars=advanced_corr_params['key_vars'],
        p_value_threshold=advanced_corr_params.get('kendall_p_value_threshold', 0.05),
        tau_threshold=advanced_corr_params.get('kendall_tau_threshold', 0.1)
    )
    # 상호 정보량
    mi_results = analyze_mutual_information(
        df=df,
        target_col=advanced_corr_params['mi_target_col'],
        key_vars=advanced_corr_params['key_vars'], # MI 분석에도 key_vars 사용
        random_state=advanced_corr_params.get('mi_random_state', 42)
    )

    # 3. 비선형 패턴 탐지 및 시각화
    print("\n--- [3] 비선형 패턴 탐지 ---")
    detect_nonlinear_patterns(df=df, **nonlinear_patterns_params)

    # 4. 시계열 추세 분석
    print("\n--- [4] 시계열 추세 분석 ---")
    analyze_time_series_trend(df=df, **time_series_trend_params)

    # 5. 성장률 분석
    print("\n--- [5] 성장률 분석 ---")
    analyze_growth_rates(df=df, **growth_rates_params)
    
    # 6. 이동 평균 트렌드 분석
    print("\n--- [6] 이동 평균 트렌드 분석 ---")
    analyze_rolling_trends(df=df, **rolling_trends_params)

    print("\n=== 고객 행동 데이터 심층 분석 완료 ===")