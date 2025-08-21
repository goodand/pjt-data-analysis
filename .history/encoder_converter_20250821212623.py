# %% [markdown]
# # 'encoder_converter.py'
# - 여기 있는 코드를 실행해서 검증 후 모듈화 할 수 있어야 함
# - 처음에 파일 설정 등 잘못된 코드가 있었음

# %% [markdown]
# 

# %%
import pandas as pd
import glob
import os

# CSV 파일들이 있는 폴더 경로
source_folder = "IsADirectoryError                         Traceback (most recent call last)
Cell In[14], line 7
      4 file_path = "/Users/jaehyuntak/Desktop/Project_____현재 진행중인/pjt-data-Note/original_data"
      6 # CSV 불러오기
----> 7 df = pd.read_csv(file_path, encoding='utf-8')  # 인코딩 utf-8, euc-kr 등 파일에 맞춰 조정
      9 # 데이터 확인
     10 print(df.head())       # 상위 5행 확인

File ~/Desktop/pjt-data-analysis/venv/lib/python3.13/site-packages/pandas/io/parsers/readers.py:1026, in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1013 kwds_defaults = _refine_defaults_read(
   1014     dialect,
   1015     delimiter,
   (...)   1022     dtype_backend=dtype_backend,
   1023 )
   1024 kwds.update(kwds_defaults)
-> 1026 return _read(filepath_or_buffer, kwds)

File ~/Desktop/pjt-data-analysis/venv/lib/python3.13/site-packages/pandas/io/parsers/readers.py:620, in _read(filepath_or_buffer, kwds)
    617 _validate_names(kwds.get("names", None))
    619 # Create the parser.
--> 620 parser = TextFileReader(filepath_or_buffer, **kwds)
    622 if chunksize or iterator:
    623     return parser
...
--> 882         handle = open(handle, ioargs.mode)
    883     handles.append(handle)
    885 # Convert BytesIO or file objects passed with an encoding

IsADirectoryError: [Errno 21] Is a directory: '/Users/jaehyuntak/Desktop/Project_____현재 진행중인/pjt-data-Note/original_data'"


# 모든 CSV 파일 목록 가져오기
csv_files = glob.glob(os.path.join(source_folder, "*.csv"))

# %%
import chardet
import pandas as pd

# 인코딩 감지 함수
def detect_encoding(file_path):
    """파일의 인코딩을 감지하는 함수"""
    try:
        with open(file_path, 'rb') as f:
            # 파일의 일부만 읽어서 인코딩 감지 (속도 향상)
            raw_data = f.read(10000)  # 처음 10KB만 읽기
            result = chardet.detect(raw_data)
            return result
    except Exception as e:
        return {'encoding': 'Error', 'confidence': 0, 'error': str(e)}

# 각 CSV 파일의 인코딩 감지
encoding_results = []

print(f"총 {len(csv_files)}개의 CSV 파일 인코딩 감지 중...\n")

for i, file_path in enumerate(csv_files, 1):
    file_name = os.path.basename(file_path)
    print(f"[{i}/{len(csv_files)}] {file_name} 처리 중...")
    
    # 인코딩 감지
    result = detect_encoding(file_path)
    
    encoding_info = {
        'file_name': file_name,
        'file_path': file_path,
        'encoding': result.get('encoding', 'Unknown'),
        'confidence': result.get('confidence', 0),
        'file_size_mb': round(os.path.getsize(file_path) / (1024*1024), 2)
    }
    
    encoding_results.append(encoding_info)
    
    # 결과 출력
    if result.get('encoding'):
        print(f"   → 인코딩: {result['encoding']} (신뢰도: {result['confidence']:.2%})")
    else:
        print(f"   → 오류: {result.get('error', '인코딩 감지 실패')}")
    print()

# 결과를 DataFrame으로 정리
encoding_df = pd.DataFrame(encoding_results)

# 결과 요약
print("=" * 60)
print("인코딩 감지 결과 요약")
print("=" * 60)
print(encoding_df.to_string(index=False))

print(f"\n인코딩별 파일 개수:")
encoding_counts = encoding_df['encoding'].value_counts()
for encoding, count in encoding_counts.items():
    print(f"  {encoding}: {count}개")

# %%
def find_real_data_start(file_path, encoding='utf-8'):
    """
    실제 데이터 테이블 시작점을 찾는 함수
    연속된 줄들이 같은 열 개수를 가지고 충분한 열을 가진 지점을 찾기
    """
    try:
        with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
            lines = f.readlines()
        
        print(f"   📄 전체 줄 수: {len(lines)}")
        
        # 각 줄의 열 개수 분석
        line_info = []
        for i, line in enumerate(lines):
            clean_line = line.strip()
            if not clean_line:
                continue
            
            # 쉼표로 분리
            parts = clean_line.split(',')
            # 빈 부분 제외하고 실제 데이터 개수 계산
            meaningful_parts = [part.strip() for part in parts if part.strip()]
            col_count = len(meaningful_parts)
            
            line_info.append({
                'line_num': i,
                'col_count': col_count,
                'content': clean_line[:50] + '...' if len(clean_line) > 50 else clean_line
            })
        
        # 열 개수별 그룹화해서 분석
        from collections import Counter
        col_count_freq = Counter([info['col_count'] for info in line_info])
        print(f"   📊 열 개수 분포: {dict(col_count_freq)}")
        
        # 가장 많이 나타나는 열 개수 중 5개 이상인 것 찾기
        most_common_cols = [col for col, freq in col_count_freq.most_common() 
                           if col >= 5 and freq >= 2]  # 최소 5열, 최소 2번 나타남
        
        if not most_common_cols:
            print(f"   ❌ 충분한 열을 가진 데이터를 찾을 수 없음")
            return None
        
        target_col_count = most_common_cols[0]
        print(f"   🎯 목표 열 개수: {target_col_count}")
        
        # 연속으로 목표 열 개수를 가진 첫 번째 지점 찾기
        consecutive_count = 0
        data_start = None
        
        for info in line_info:
            if info['col_count'] == target_col_count:
                consecutive_count += 1
                if consecutive_count == 1:  # 첫 번째 일치하는 줄
                    data_start = info['line_num']
                if consecutive_count >= 2:  # 2줄 연속 일치
                    print(f"   ✅ 데이터 시작점: {data_start}번째 줄")
                    return data_start
            else:
                consecutive_count = 0
                data_start = None
        
        # 연속 2줄을 못 찾았다면 첫 번째 일치하는 줄 사용
        for info in line_info:
            if info['col_count'] == target_col_count:
                print(f"   ⚠️ 단일 일치점 사용: {info['line_num']}번째 줄")
                return info['line_num']
        
        return None
        
    except Exception as e:
        print(f"   ❌ 분석 오류: {e}")
        return None

# %%
def clean_csv_with_auto_header(file_path):
    """
    헤더 자동 감지 방식: 데이터 시작점 바로 위 줄을 헤더로 가정
    """
    file_name = os.path.basename(file_path)
    print(f"\n📁 {file_name}")
    
    encodings = ['utf-8', 'cp949', 'euc-kr', 'latin1']
    
    for encoding in encodings:
        try:
            # 데이터 시작점 찾기 (기존 함수 사용)
            data_start_line = find_real_data_start(file_path, encoding)
            
            if data_start_line is None:
                continue
            
            # 헤더는 데이터 시작점 바로 위 줄로 가정
            header_line = data_start_line - 1
            skiprows = header_line if header_line > 0 else 0
            
            print(f"   📋 헤더 줄: {header_line}번째")
            print(f"   📊 데이터 시작: {data_start_line}번째")
            print(f"   🔄 skiprows: {skiprows}")
            
            # CSV 읽기 (헤더 포함)
            df = pd.read_csv(file_path, 
                           encoding=encoding,
                           skiprows=skiprows,
                           on_bad_lines='skip')
            
            if df.empty:
                continue
            
            print(f"   ✅ 읽기 성공 (인코딩: {encoding})")
            print(f"   📊 Shape: {df.shape}")
            print(f"   📋 컬럼: {list(df.columns)}")
            
            # 기본 정리
            df = df.dropna(how='all')
            df = df.drop_duplicates()
            df.columns = df.columns.astype(str).str.strip()
            
            print(f"   🧹 정리 후 shape: {df.shape}")
            
            return df, encoding, file_name
            
        except Exception as e:
            print(f"   ❌ {encoding} 실패: {str(e)[:50]}...")
            continue
    
    print(f"   ❌ 모든 시도 실패")
    return None, None, None

# 테스트 실행
if csv_files:
    test_df, test_encoding, test_name = clean_csv_with_auto_header(csv_files[0])
    
    if test_df is not None:
        print(f"\n🎉 자동 헤더 감지 성공!")
        print(f"컬럼: {list(test_df.columns)}")
        print(f"\n샘플 데이터:")
        print(test_df.head(3))
    else:
        print(f"\n❌ 자동 헤더 감지 실패")

# %%
test_df.head(10)

# %%
def process_all_csv_files_final(csv_files):
    """
    모든 CSV 파일을 처리하여 통합 준비
    """
    print("=" * 60)
    print("전체 CSV 파일 처리 시작")
    print("=" * 60)
    
    processed_data = []
    
    for i, file_path in enumerate(csv_files, 1):
        print(f"\n[{i}/{len(csv_files)}] 처리 중...")
        
        df, encoding, file_name = clean_csv_with_auto_header(file_path)
        
        if df is not None and not df.empty:
            # NO 컬럼 삭제
            if 'NO' in df.columns:
                df = df.drop('NO', axis=1)
                print(f"   🗑️ NO 컬럼 삭제됨")
            
            # 파일명에서 날짜 추출 (앞의 8자리)
            import re
            date_match = re.search(r'^(\d{8})', file_name)
            file_date = date_match.group(1) if date_match else '99999999'
            
            processed_data.append({
                'dataframe': df,
                'file_name': file_name,
                'file_date': file_date,
                'rows': len(df)
            })
            
            print(f"   ✅ 완료: {len(df):,} 행, 파일날짜: {file_date}")
        else:
            print(f"   ❌ 처리 실패")
    
    print(f"\n🎉 총 {len(processed_data)}개 파일 처리 완료")
    return processed_data

# 실행
processed_data = process_all_csv_files_final(csv_files)

# %%
def combine_csv_files_by_date(processed_data):
    """
    파일날짜 기준으로 정렬하여 CSV 파일들 통합
    """
    print("\n" + "=" * 60)
    print("파일날짜 기준 정렬 및 통합")
    print("=" * 60)
    
    if not processed_data:
        print("❌ 처리된 데이터가 없습니다")
        return None
    
    # 파일날짜 기준으로 정렬
    processed_data.sort(key=lambda x: x['file_date'])
    
    print("📅 정렬된 파일 순서:")
    total_rows = 0
    for i, data in enumerate(processed_data, 1):
        date_formatted = f"{data['file_date'][:4]}-{data['file_date'][4:6]}-{data['file_date'][6:8]}"
        total_rows += data['rows']
        print(f"   {i}. {date_formatted} - {data['file_name'][:30]}... ({data['rows']:,} 행)")
    
    print(f"\n📊 총 예상 행 수: {total_rows:,}")
    
    # DataFrame들 통합
    print(f"\n🔄 DataFrame 통합 중...")
    dataframes = [data['dataframe'] for data in processed_data]
    combined_df = pd.concat(dataframes, ignore_index=True, sort=False)
    
    print(f"✅ 통합 완료: {len(combined_df):,} 행, {len(combined_df.columns)} 컬럼")
    
    # 새로운 NO 컬럼 생성 (1부터 시작)
    combined_df.insert(0, 'NO', range(1, len(combined_df) + 1))
    print(f"📋 새로운 NO 컬럼 추가됨 (1~{len(combined_df)})")
    
    return combined_df

# 실행
combined_df = combine_csv_files_by_date(processed_data)

# %%
from datetime import datetime

def save_combined_result(combined_df, output_folder):
    """
    통합된 결과를 파일로 저장
    """
    print("\n" + "=" * 60)
    print("결과 저장")
    print("=" * 60)
    
    if combined_df is None:
        print("❌ 저장할 데이터가 없습니다")
        return None
    
    # 출력 폴더 생성
    os.makedirs(output_folder, exist_ok=True)
    
    # 파일명 생성
    timestamp = datetime.now().strftime("%Y%m%d")
    output_filename = f"combined_아파트매매_{timestamp}.csv"
    output_path = os.path.join(output_folder, output_filename)
    
    # CSV 저장
    combined_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"✅ 파일 저장 완료: {output_filename}")
    print(f"📍 경로: {output_path}")
    print(f"📊 최종 크기: {len(combined_df):,} 행 × {len(combined_df.columns)} 컬럼")
    
    # 요약 정보
    print(f"\n📋 컬럼 정보:")
    for i, col in enumerate(combined_df.columns, 1):
        print(f"   {i:2d}. {col}")
    
    # 미리보기
    print(f"\n👀 결과 미리보기 (처음 3행):")
    print(combined_df.head(3))
    
    return output_path

# 실행
output_folder = "/Users/jaehyuntak/Desktop/Project_____현재 진행중인/pjt-data-Note/combine_origin/Combined_Results"
result_path = save_combined_result(combined_df, output_folder)

if result_path:
    print(f"\n🎉 통합 작업 완료!")
    print(f"결과 파일: {os.path.basename(result_path)}")


