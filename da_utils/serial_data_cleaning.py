# 'serial_data_cleaning.py'

"""
serial_data_cleaning.py
데이터 정제를 위한 함수 모음

Author: jaehyuntak
Description: data_cleaning.ipynb를 함수화한 모듈
"""

# ============================================================================
# 1. 기본 라이브러리
# ============================================================================
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 3. 경로 설정 및 사용자 정의 모듈
# ============================================================================
import sys
import os
sys.path.append('/Users/jaehyuntak/Desktop/pjt-data-analysis')

# 한글 폰트 설정
from da_utils.font import setup_korean_font

# 데이터 품질 검증 모듈
from da_utils import data_profile, outliers, patterns

# ============================================================================
# 4. 모듈 초기화
# ============================================================================
# 한글 폰트 설정 (모듈 import 시 자동 실행)
setup_korean_font()

print("✅ serial_data_cleaning 모듈이 성공적으로 로드되었습니다.")
print("📋 사용 가능한 함수 목록:")
print("   - load_data()")
print("   - analyze_missing_values()")
print("   - replace_dash_with_nan()")
print("   - process_date_columns()")
print("   - classify_transaction_status()")
print("   - (추가 예정...)")


# ============================================================================
# 5. 데이터 로딩 함수
# ============================================================================

def load_data(file_path, encoding='utf-8', show_info=True):
    """
    CSV 파일을 로드하고 기본 정보를 출력합니다.
    
    Parameters:
    -----------
    file_path : str
        CSV 파일 경로
    encoding : str, default 'utf-8'
        파일 인코딩 방식
    show_info : bool, default True
        기본 정보 출력 여부
        
    Returns:
    --------
    pandas.DataFrame
        로드된 데이터프레임
        
    Example:
    --------
    >>> file_path = '/path/to/data.csv'
    >>> df = load_data(file_path)
    """
    try:
        # CSV 파일 로드
        df = pd.read_csv(file_path, encoding=encoding)
        
        if show_info:
            print("=" * 50)
            print("🔍 데이터 로드 완료")
            print("=" * 50)
            print(f"📁 파일 경로: {file_path}")
            print(f"📊 데이터 크기: {df.shape[0]:,}행 × {df.shape[1]}열")
            print(f"💾 메모리 사용량: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            print("\n" + "=" * 50)
            print("📋 데이터 타입 정보")
            print("=" * 50)
            print(df.dtypes)
            
            print("\n" + "=" * 50)
            print("🔢 각 컬럼별 고유값 개수")
            print("=" * 50)
            for col in df.columns:
                unique_count = df[col].nunique()
                print(f"   {col}: {unique_count:,}개")
            
            print("\n" + "=" * 50)
            print("📈 숫자형 컬럼 기본 통계")
            print("=" * 50)
            numeric_stats = df.describe()
            print(numeric_stats)
            
            print("\n" + "=" * 50)
            print("👀 데이터 미리보기 (상위 5행)")
            print("=" * 50)
            print(df.head())
            
        return df
        
    except FileNotFoundError:
        print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
        return None
    except Exception as e:
        print(f"❌ 데이터 로드 중 오류 발생: {str(e)}")
        return None
    


    

# ============================================================================
# 6. 결측치 분석 함수
# ============================================================================

def analyze_missing_values(df, show_details=True):
    """
    명시적 결측치(NaN)와 암묵적 결측치('-' 문자)를 분석합니다.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        분석할 데이터프레임
    show_details : bool, default True
        상세 정보 출력 여부
        
    Returns:
    --------
    dict
        결측치 분석 결과 딕셔너리
        - 'null_summary': 명시적 결측치 요약
        - 'dash_summary': 암묵적 결측치 요약
        
    Example:
    --------
    >>> missing_info = analyze_missing_values(df)
    """
    result = {}
    
    if show_details:
        print("=" * 50)
        print("🔍 결측치 분석")
        print("=" * 50)
    
    # 1. 명시적 결측치(NaN) 분석
    null_counts = df.isnull().sum()
    null_percentages = (df.isnull().sum() / len(df)) * 100
    null_summary = pd.DataFrame({
        '결측치 개수': null_counts,
        '결측치 비율(%)': null_percentages.round(2)
    })
    null_summary = null_summary[null_summary['결측치 개수'] > 0].sort_values('결측치 비율(%)', ascending=False)
    
    result['null_summary'] = null_summary
    
    if show_details:
        print("\n📊 명시적 결측치(NaN) 분석")
        print("-" * 30)
        if len(null_summary) > 0:
            print(null_summary)
        else:
            print("✅ 명시적 결측치가 없습니다.")
    
    # 2. 암묵적 결측치('-' 문자) 분석
    dash_counts = {}
    for col in df.columns:
        if df[col].dtype == 'object':  # 문자열 컬럼만 확인
            dash_count = (df[col] == '-').sum()
            if dash_count > 0:
                dash_counts[col] = {
                    '개수': dash_count,
                    '비율(%)': round((dash_count / len(df)) * 100, 2)
                }
    
    if dash_counts:
        dash_summary = pd.DataFrame(dash_counts).T
        dash_summary = dash_summary.sort_values('비율(%)', ascending=False)
    else:
        dash_summary = pd.DataFrame()  # 빈 데이터프레임
        
    result['dash_summary'] = dash_summary
    
    if show_details:
        print("\n📊 암묵적 결측치('-' 문자) 분석")
        print("-" * 30)
        if not dash_summary.empty:
            print(dash_summary)
        else:
            print("✅ '-' 문자로 표시된 결측치가 없습니다.")
    
    # 3. 전체 요약
    total_nulls = null_summary['결측치 개수'].sum() if not null_summary.empty else 0
    total_dashes = sum([info['개수'] for info in dash_counts.values()]) if dash_counts else 0
    
    if show_details:
        print(f"\n📋 결측치 요약")
        print("-" * 30)
        print(f"   전체 데이터: {len(df):,}건")
        print(f"   명시적 결측치(NaN): {total_nulls:,}개")
        print(f"   암묵적 결측치('-'): {total_dashes:,}개")
        print(f"   총 결측치: {total_nulls + total_dashes:,}개")
    
    result['total_nulls'] = total_nulls
    result['total_dashes'] = total_dashes
    result['total_missing'] = total_nulls + total_dashes
    
    return result



# ============================================================================
# 7. '-' 문자를 NaN으로 변환하는 함수
# ============================================================================

def replace_dash_with_nan(df, columns=None, show_details=True):
    """
    지정된 컬럼들의 '-' 문자를 NaN으로 변환합니다.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        처리할 데이터프레임
    columns : list or None, default None
        처리할 컬럼 리스트. None이면 기본 컬럼들 사용
    show_details : bool, default True
        변환 과정 출력 여부
        
    Returns:
    --------
    pandas.DataFrame
        '-' → NaN 변환된 데이터프레임 복사본
    """
    # 기본 컬럼 설정 (원본 ipynb 파일 기준)
    if columns is None:
        columns = ['해제사유발생일', '매수자', '매도자', '동', '등기일자', '중개사소재지']
    
    df_result = df.copy()
    
    if show_details:
        print("=" * 50)
        print("🔧 '-' 문자를 NaN으로 변환")
        print("=" * 50)
    
    for col in columns:
        if col in df_result.columns:
            before_count = (df_result[col] == '-').sum()
            df_result[col] = df_result[col].replace('-', np.nan)
            after_count = df_result[col].isna().sum()
            
            if show_details:
                print(f"   {col}: {before_count:,}개 '-' → {after_count:,}개 NaN")
        else:
            if show_details:
                print(f"   ⚠️ 컬럼 '{col}'이 데이터에 없습니다.")
    
    return df_result



# ============================================================================
# 8. 날짜 처리 함수들
# ============================================================================

def process_date_columns(df, show_details=True):
    """
    날짜 관련 컬럼들을 처리합니다.
    1. 중복 날짜 컬럼 제거 (계약일자 삭제)
    2. 계약날짜 생성 (계약년월 + 계약일)
    3. 등기일자 datetime 변환
    
    Parameters:
    -----------
    df : pandas.DataFrame
        처리할 데이터프레임
    show_details : bool, default True
        처리 과정 출력 여부
        
    Returns:
    --------
    pandas.DataFrame
        날짜 처리된 데이터프레임 복사본
    """
    df_result = df.copy()
    
    if show_details:
        print("=" * 50)
        print("📅 날짜 컬럼 처리")
        print("=" * 50)
    
    # 1. 중복 날짜 컬럼 제거 (계약일자가 있으면 삭제)
    if '계약일자' in df_result.columns:
        df_result = df_result.drop(columns=['계약일자'])
        if show_details:
            print("   ✅ 중복 컬럼 '계약일자' 삭제")
    
    # 2. 계약날짜 생성 (계약년월 + 계약일 → 계약날짜)
    if '계약년월' in df_result.columns and '계약일' in df_result.columns:
        df_result['계약날짜'] = pd.to_datetime(
            df_result['계약년월'].astype(str) + df_result['계약일'].astype(str).str.zfill(2),
            format='%Y%m%d'
        )
        if show_details:
            print("   ✅ 계약날짜 생성 (계약년월 + 계약일)")
            print(f"      기간: {df_result['계약날짜'].min()} ~ {df_result['계약날짜'].max()}")
    
    # 3. 등기일자 datetime 변환
    if '등기일자' in df_result.columns:
        before_count = df_result['등기일자'].notna().sum()
        df_result['등기일자'] = pd.to_datetime(
            df_result['등기일자'], 
            format='%y.%m.%d', 
            errors='coerce'
        )
        after_count = df_result['등기일자'].notna().sum()
        
        if show_details:
            print(f"   ✅ 등기일자 datetime 변환")
            print(f"      변환 성공: {after_count:,}건")
            print(f"      변환 실패 또는 원래 NaN: {len(df_result) - after_count:,}건")
    
    if show_details:
        print(f"\n   📋 처리 후 컬럼 수: {len(df_result.columns)}개")
        
    return df_result



# ============================================================================
# 9. 거래상태 분류 함수
# ============================================================================

def classify_transaction_status(df, show_details=True):
    """
    거래 상태를 분류합니다.
    - 정상완료: 해제사유 없음 & 등기일자 있음
    - 해제: 해제사유 있음
    - 진행중: 해제사유 없음 & 등기일자 없음
    
    Parameters:
    -----------
    df : pandas.DataFrame
        분류할 데이터프레임
    show_details : bool, default True
        분류 결과 출력 여부
        
    Returns:
    --------
    pandas.DataFrame
        거래상태 컬럼이 추가된 데이터프레임 복사본
    """
    df_result = df.copy()
    
    if show_details:
        print("=" * 50)
        print("🏷️ 거래상태 분류")
        print("=" * 50)
    
    # 거래상태 초기화
    df_result['거래상태'] = '기타'
    
    # 1. 정상완료: 해제사유 없음 & 등기일자 있음
    normal_mask = (df_result['해제사유발생일'].isna()) & (df_result['등기일자'].notna())
    df_result.loc[normal_mask, '거래상태'] = '정상완료'
    
    # 2. 해제: 해제사유 있음
    cancel_mask = df_result['해제사유발생일'].notna()
    df_result.loc[cancel_mask, '거래상태'] = '해제'
    
    # 3. 진행중: 해제사유 없음 & 등기일자 없음
    ongoing_mask = (df_result['해제사유발생일'].isna()) & (df_result['등기일자'].isna())
    df_result.loc[ongoing_mask, '거래상태'] = '진행중'
    
    if show_details:
        print("📊 거래상태별 분포:")
        status_counts = df_result['거래상태'].value_counts()
        for status, count in status_counts.items():
            percentage = count / len(df_result) * 100
            print(f"   - {status}: {count:,}건 ({percentage:.1f}%)")
        
        # 거래상태별 결측 패턴 분석
        print("\n📋 거래상태별 주요 필드 결측률:")
        status_missing = df_result.groupby('거래상태')[['동', '등기일자', '해제사유발생일', '매수자', '중개사소재지']].apply(
            lambda x: x.isna().sum() / len(x) * 100
        ).round(1)
        print(status_missing)
    
    return df_result



# ============================================================================
# 10. 정상거래 필터링 및 거래금액 변환 함수
# ============================================================================

def filter_normal_transactions(df, show_details=True):
    """
    정상완료 거래만 필터링하고 거래금액을 숫자형으로 변환합니다.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        필터링할 데이터프레임 (거래상태 컬럼 필요)
    show_details : bool, default True
        필터링 결과 출력 여부
        
    Returns:
    --------
    pandas.DataFrame
        정상완료 거래만 포함된 데이터프레임
    """
    if show_details:
        print("=" * 50)
        print("✅ 정상거래 필터링 및 데이터 변환")
        print("=" * 50)
    
    # 정상완료 거래만 필터링
    df_normal = df[df['거래상태'] == '정상완료'].copy()
    
    # 거래금액을 숫자로 변환
    df_normal['거래금액(만원)'] = df_normal['거래금액(만원)'].str.replace(',', '').astype(int)
    
    if show_details:
        print(f"📊 데이터 필터링 결과:")
        print(f"   원본 데이터: {len(df):,}건")
        print(f"   정상완료 거래: {len(df_normal):,}건 ({len(df_normal)/len(df)*100:.1f}%)")
        print(f"   제외된 거래: {len(df) - len(df_normal):,}건")
        
        print(f"\n📋 정상완료 거래의 특성:")
        if '계약날짜' in df_normal.columns:
            print(f"   기간: {df_normal['계약날짜'].min().strftime('%Y-%m-%d')} ~ {df_normal['계약날짜'].max().strftime('%Y-%m-%d')}")
        print(f"   평균 거래가격: {df_normal['거래금액(만원)'].mean():,.0f}만원")
        print(f"   평균 전용면적: {df_normal['전용면적(㎡)'].mean():.1f}㎡")
        
        # 지역별 분포 (시군구에서 지역 추출)
        if '시군구' in df_normal.columns:
            df_normal_temp = df_normal.copy()
            df_normal_temp['지역'] = df_normal_temp['시군구'].str.split().str[0]  # 첫 번째 단어만 추출
            print(f"\n📍 지역별 정상완료 거래 분포:")
            region_dist = df_normal_temp['지역'].value_counts()
            for region, count in region_dist.items():
                print(f"   - {region}: {count:,}건 ({count/len(df_normal)*100:.1f}%)")
    
    return df_normal


# ============================================================================
# 11. 데이터 저장 함수
# ============================================================================

def save_cleaned_data(df, output_path, filename=None, show_details=True):
    """
    정제된 데이터를 CSV 파일로 저장합니다.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        저장할 데이터프레임
    output_path : str
        저장할 폴더 경로
    filename : str or None, default None
        파일명. None이면 자동 생성
    show_details : bool, default True
        저장 결과 출력 여부
        
    Returns:
    --------
    str
        저장된 파일의 전체 경로
    """
    import os
    from datetime import datetime
    
    # 폴더가 없으면 생성
    os.makedirs(output_path, exist_ok=True)
    
    # 파일명 자동 생성
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d')
        filename = f'cleaned_normal_transactions_{timestamp}.csv'
    
    # 전체 경로
    full_path = os.path.join(output_path, filename)
    
    # CSV 저장
    df.to_csv(full_path, index=False, encoding='utf-8')
    
    if show_details:
        print("=" * 50)
        print("💾 데이터 저장 완료")
        print("=" * 50)
        print(f"   📁 저장 경로: {full_path}")
        print(f"   📊 저장된 데이터: {len(df):,}건")
        print(f"   📋 컬럼 수: {len(df.columns)}개")
        print(f"   💽 파일 크기: {os.path.getsize(full_path) / 1024**2:.2f} MB")
    
    return full_path


# ============================================================================
# 12. 전체 데이터 정제 파이프라인 함수
# ============================================================================

def clean_all_data(file_path, output_path=None, save_result=True, show_details=True):
    """
    전체 데이터 정제 파이프라인을 실행합니다.
    
    Parameters:
    -----------
    file_path : str
        원본 CSV 파일 경로
    output_path : str or None, default None
        저장할 폴더 경로. None이면 저장 안함
    save_result : bool, default True
        결과 저장 여부
    show_details : bool, default True
        전체 과정 출력 여부
        
    Returns:
    --------
    pandas.DataFrame
        정제 완료된 데이터프레임
    """
    if show_details:
        print("🚀 데이터 정제 파이프라인 시작")
        print("=" * 70)
    
    # 1. 데이터 로드
    df = load_data(file_path, show_info=show_details)
    if df is None:
        return None
    
    # 2. 결측치 분석
    missing_info = analyze_missing_values(df, show_details=show_details)
    
    # 3. '-' → NaN 변환
    df = replace_dash_with_nan(df, show_details=show_details)
    
    # 4. 날짜 처리
    df = process_date_columns(df, show_details=show_details)
    
    # 5. 거래상태 분류
    df = classify_transaction_status(df, show_details=show_details)
    
    # 6. 정상거래 필터링
    df_final = filter_normal_transactions(df, show_details=show_details)
    
    # 7. 데이터 저장
    if save_result and output_path:
        saved_path = save_cleaned_data(df_final, output_path, show_details=show_details)
    
    if show_details:
        print("\n" + "=" * 70)
        print("🎉 데이터 정제 파이프라인 완료!")
        print("=" * 70)
        print(f"📊 최종 결과: {len(df_final):,}건의 정상거래 데이터")
        
    return df_final