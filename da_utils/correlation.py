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