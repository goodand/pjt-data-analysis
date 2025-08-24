# font.py - matplotlib 한글 폰트 설정 개선 모듈 (디버깅 강화)

"""
matplotlib 한글 폰트 설정 모듈 (개선 버전)

[사용법]
1. 가장 일반적인 사용법:
   import matplotlib.pyplot as plt
   from font import setup_korean_font
   
   setup_korean_font()  # 그래프를 그리기 전 한 번만 호출

   # 이후 matplotlib에서 한글이 정상적으로 표시됩니다.
   plt.title('한글 제목')
   plt.xlabel('X축 (한글)')
   plt.plot([1, 2, 3], [10, 20, 15])
   plt.show()

2. 폰트가 계속 깨질 경우 (폰트 설치 후):
   from font import rebuild_font_cache
   rebuild_font_cache() 
   # 위 함수 실행 후 파이썬(또는 주피터 노트북)을 완전히 재시작해야 합니다.

[주요 개선 사항]
- 시스템에 지정된 폰트가 없을 경우, 사용 가능한 다른 한글 폰트를 자동으로 찾아 설정합니다.
- 폰트 설정 결과에 대한 명확한 피드백을 제공합니다.
- matplotlib의 폰트 캐시를 재설정하는 유틸리티 함수를 추가했습니다.
- 문제 해결을 위한 상세한 디버깅 정보를 출력합니다.
"""

import platform
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import shutil
import warnings

def setup_korean_font(debug=False):
    """
    운영체제에 맞는 한글 폰트를 자동으로 설정하고, 없을 경우 대안을 찾는 함수.
    debug=True로 설정하면 더 상세한 정보를 출력합니다.
    """
    print("--- 한글 폰트 설정 시작 ---")
    plt.rcParams['axes.unicode_minus'] = False

    system = platform.system()
    print(f"운영체제: {system}")

    font_family = ""
    if system == "Darwin":
        font_family = "AppleGothic"
    elif system == "Windows":
        font_family = "Malgun Gothic"
    else:
        font_family = "NanumGothic"
    print(f"기본 폰트로 '{font_family}'를 사용합니다.")

    try:
        available_fonts = {f.name for f in fm.fontManager.ttflist}
        if debug:
            print("설치된 전체 폰트 목록 (일부):", sorted(list(available_fonts))[:10])

        if font_family in available_fonts:
            plt.rc('font', family=font_family)
            print(f"✅ 성공: '{font_family}' 폰트를 설정했습니다.")
        else:
            print(f"⚠️ 경고: '{font_family}' 폰트를 찾을 수 없습니다. 다른 한글 폰트를 검색합니다.")
            korean_fonts = get_available_korean_fonts()
            if korean_fonts:
                found_font = korean_fonts[0]
                plt.rc('font', family=found_font)
                print(f"✅ 성공: 대안 폰트인 '{found_font}'으로 설정했습니다.")
            else:
                print("❌ 오류: 시스템에 한글 폰트가 설치되어 있지 않습니다.")
                print("➡️ '나눔고딕(NanumGothic)'과 같은 한글 폰트를 설치한 후 다시 시도해 주세요.")
                # 기본 폰트로 되돌림
                plt.rc('font', family=fm.findfont(fm.FontProperties()))
    except Exception as e:
        print(f"❌ 오류: 폰트 설정 중 예기치 않은 문제가 발생했습니다: {e}")
    finally:
        print("--- 한글 폰트 설정 종료 ---")

def get_available_korean_fonts():
    """
    시스템에서 사용 가능한 한글 폰트 목록을 반환합니다.
    """
    korean_font_names = [
        'AppleGothic', 'Apple SD Gothic Neo', 'NanumGothic', 
        'Malgun Gothic', 'Batang', 'Gulim', 'Dotum', 'Gungsuh'
    ]
    korean_fonts_found = []
    
    for font_path in fm.findSystemFonts():
        try:
            font_prop = fm.FontProperties(fname=font_path)
            font_name = font_prop.get_name()
            if any(korean_name in font_name for korean_name in korean_font_names):
                if font_name not in korean_fonts_found:
                    korean_fonts_found.append(font_name)
        except Exception:
            continue
            
    return sorted(korean_fonts_found)

def rebuild_font_cache():
    """
    matplotlib의 폰트 캐시를 삭제하고 다시 빌드합니다.
    """
    try:
        cache_dir = fm.get_cachedir()
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            print(f"✅ Matplotlib 폰트 캐시('{cache_dir}')를 삭제했습니다.")
            print("🔴 중요: 변경사항을 적용하려면 Python(Jupyter, VSCode 등)을 완전히 재시작해야 합니다.")
        else:
            print("✅ Matplotlib 폰트 캐시가 존재하지 않습니다. 별도 조치가 필요 없습니다.")
    except Exception as e:
        print(f"❌ 오류: 폰트 캐시 삭제 중 문제가 발생했습니다: {e}")

def test_korean_display():
    """
    한글 표시가 잘 되는지 확인하기 위한 테스트 그래프를 생성합니다.
    """
    setup_korean_font(debug=True) # 디버그 모드로 폰트 설정 실행

    fig, ax = plt.subplots(figsize=(8, 4))
    
    test_texts = [
        '한글 폰트 테스트',
        '그래프 제목: 데이터 분석 결과',
        '축 레이블: 값, 빈도',
        '범례: 항목1, 항목2',
        '음수 부호: -1, -2, -3'
    ]
    
    for i, text in enumerate(test_texts):
        ax.text(0.1, 0.85 - i*0.18, text, fontsize=14, transform=ax.transAxes)
    
    ax.set_title('Korean Font Display Test / 한글 폰트 표시 테스트')
    ax.set_xlabel("X축 이름")
    ax.set_ylabel("Y축 이름")
    ax.plot([-1, 0, 1], [-1, 0, 1])
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("="*40)
    print("   Matplotlib 한글 폰트 설정 도우미 (v2)")
    print("="*40)
    
    print("\n1. 사용 가능한 한글 폰트 목록:")
    available_fonts = get_available_korean_fonts()
    if available_fonts:
        for font in available_fonts:
            print(f"- {font}")
    else:
        print("-> 시스템에서 한글 폰트를 찾을 수 없습니다.")
    
    print("\n2. 한글 폰트 표시 테스트 그래프를 생성합니다...")
    test_korean_display()