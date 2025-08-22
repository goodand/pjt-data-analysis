# %%
from da_utils.font import setup_korean_font

# 한글 폰트 설정 (한 번만 호출하면 됨)
setup_korean_font()

# %%
import pandas as pd

# CSV 파일 경로
file_path = "/Users/jaehyuntak/Desktop/pjt-data-analysis/data_row/220820-250819combined_capital_area_apt_sales.csv"

# CSV 불러오기
df = pd.read_csv(file_path, encoding='utf-8')  # 인코딩 utf-8, euc-kr 등 파일에 맞춰 조정

# 데이터 확인
print(df.head())       # 상위 5행 확인
print(df.info())       # 데이터 타입 및 결측치 확인
print(df.describe())   # 수치형 컬럼 기본 통계


# %%
# 이 내용을 GPT한테 공유해주니깐 진단을 해주네 
df.info()

# %%
df.info()

# %%
# 원본 df는 그대로, 전처리용 복사본 생성
df_clean = df.copy()

# 컬럼명 영문화 매핑
col_map = {
    "NO": "no",
    "시군구": "district",
    "번지": "lot_number",
    "본번": "main_number",
    "부번": "sub_number",
    "단지명": "complex_name",
    "전용면적(㎡)": "area_m2",
    "계약년월": "contract_ym",
    "계약일": "contract_day",
    "거래금액(만원)": "price_krw",
    "동": "building",
    "층": "floor",
    "매수자": "buyer",
    "매도자": "seller",
    "건축년도": "year_built",
    "도로명": "road_name",
    "해제사유발생일": "cancel_date",
    "거래유형": "trade_type",
    "중개사소재지": "broker_location",
    "등기일자": "register_date"
}
df_clean = df_clean.rename(columns=col_map)

# 거래금액 숫자화
df_clean["price_krw"] = df_clean["price_krw"].str.replace(",", "").astype(int)

# 계약일자 처리 → 연월+일 합쳐서 datetime
df_clean["contract_date"] = pd.to_datetime(
    df_clean["contract_ym"].astype(str) + df_clean["contract_day"].astype(str).str.zfill(2),
    format="%Y%m%d"
)

# 건축년도도 datetime 변환 (연만 있음 → 1월1일로 통일)
df_clean["built_date"] = pd.to_datetime(df_clean["year_built"].astype(str), format="%Y")

# 등기일자 datetime 변환 (결측값 있을 수 있음 → errors='coerce')
df_clean["register_date"] = pd.to_datetime(df_clean["register_date"], errors="coerce")

# 변환 후 확인
print(df_clean.info())
print(df_clean.head())


# %%
import os
import sys

# 현재 작업 디렉토리를 da_utils로 설정
sys.path.insert(0, os.path.abspath("/Users/jaehyuntak/Desktop/pjt-data-analysis/da_utils"))

from data_profile import get_data_profile


# df_clean 프로파일링
get_data_profile(df_clean)


# %%
from da_utils.outliers import outlier_detection

# 이상치 탐지 
outlier_detection(df_clean)


