# 한글 폰트 설정 (선택사항)
# plt.rcParams['font.family'] = 'Malgun Gothic'
# plt.rcParams['axes.unicode_minus'] = False

"""
font.py - 한글 폰트 설정 모듈

사용법:
------
1. 기본 사용법 (가장 일반적):
   from font import setup_korean_font
   setup_korean_font()  # 한 번만 호출하면 됨
   
   # 이후 matplotlib 사용시 한글이 정상 표시됨
   import matplotlib.pyplot as plt
   plt.title('한글 제목')
   plt.xlabel('한글 x축')
   plt.ylabel('한글 y축')

2. 다른 폴더에서 사용 (예: da_utils 폴더에 저장한 경우):
   from da_utils.font import setup_korean_font
   setup_korean_font()

3. 선택적 기능:
   from font import get_available_korean_fonts, test_korean_display
   
   # 시스템의 한글 폰트 목록 확인
   fonts = get_available_korean_fonts()
   print(fonts)
   
   # 한글 표시 테스트
   fig = test_korean_display()
   plt.show()

주의사항:
--------
- setup_korean_font()는 프로그램당 한 번만 호출하면 됩니다
- macOS에서 AppleGothic 폰트가 자동으로 설정됩니다
- 음수 기호 깨짐도 자동으로 방지됩니다
"""


# 핵심 기능: 한글 폰트 설정
import matplotlib.pyplot as plt
import platform
import warnings

def setup_korean_font():
    """
    운영체제에 따라 한글 폰트를 자동으로 설정하는 핵심 함수
    """
    system = platform.system()
    
    try:
        if system == "Darwin":  # macOS
            # macOS에서 사용 가능한 한글 폰트들 (우선순위 순)
            mac_fonts = [
                'AppleGothic',      # 기본 한글 폰트
                'Apple SD Gothic Neo',  # 시스템 기본 폰트
                'Nanum Gothic',     # 나눔고딕 (설치된 경우)
                'Malgun Gothic'     # 맑은고딕 (설치된 경우)
            ]
            
            font_set = False
            for font in mac_fonts:
                try:
                    plt.rcParams['font.family'] = font
                    # 테스트용 한글 텍스트로 폰트 확인
                    test_fig = plt.figure(figsize=(1, 1))
                    plt.text(0.5, 0.5, '한글테스트', fontsize=10)
                    plt.close(test_fig)
                    print(f"✅ Korean font set successfully: {font}")
                    font_set = True
                    break
                except:
                    continue
            
            if not font_set:
                print("⚠️  No suitable Korean font found. Using default font.")
                
        elif system == "Windows":  # Windows
            plt.rcParams['font.family'] = 'Malgun Gothic'
            print("✅ Korean font set successfully: Malgun Gothic (Windows)")
            
        elif system == "Linux":  # Linux
            linux_fonts = [
                'Nanum Gothic',
                'DejaVu Sans'
            ]
            
            font_set = False
            for font in linux_fonts:
                try:
                    plt.rcParams['font.family'] = font
                    print(f"✅ Korean font set successfully: {font}")
                    font_set = True
                    break
                except:
                    continue
                    
            if not font_set:
                print("⚠️  Please install Nanum Gothic font for Korean text support")
        
        # 음수 기호 깨짐 방지 (모든 OS 공통)
        plt.rcParams['axes.unicode_minus'] = False
        
    except Exception as e:
        warnings.warn(f"Font setup failed: {e}. Using system default.")
        plt.rcParams['axes.unicode_minus'] = False


# 선택적 유틸리티 함수들 (setup_korean_font와 독립적으로 작동)
import matplotlib.font_manager as fm

def get_available_korean_fonts():
    """
    시스템에서 사용 가능한 한글 폰트 목록을 반환합니다.
    """
    korean_fonts = []
    font_list = fm.findSystemFonts()
    
    korean_font_names = [
        'AppleGothic', 'Apple SD Gothic Neo', 'Nanum Gothic', 
        'Malgun Gothic', 'Batang', 'Gulim', 'Dotum'
    ]
    
    for font_path in font_list:
        try:
            font_prop = fm.FontProperties(fname=font_path)
            font_name = font_prop.get_name()
            if any(korean in font_name for korean in korean_font_names):
                if font_name not in korean_fonts:
                    korean_fonts.append(font_name)
        except:
            continue
    
    return korean_fonts

def test_korean_display():
    """
    한글 표시 테스트 함수
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 테스트 텍스트
    test_texts = [
        '한글 폰트 테스트',
        '그래프 제목: 데이터 분석',
        '축 레이블: 값, 빈도',
        '범례: 항목1, 항목2'
    ]
    
    for i, text in enumerate(test_texts):
        ax.text(0.1, 0.8 - i*0.15, text, fontsize=14, transform=ax.transAxes)
    
    ax.set_title('Korean Font Display Test / 한글 폰트 표시 테스트')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    return fig

def get_available_korean_fonts():
    """
    시스템에서 사용 가능한 한글 폰트 목록을 반환합니다.
    """
    korean_fonts = []
    font_list = fm.findSystemFonts()
    
    korean_font_names = [
        'AppleGothic', 'Apple SD Gothic Neo', 'Nanum Gothic', 
        'Malgun Gothic', 'Batang', 'Gulim', 'Dotum'
    ]
    
    for font_path in font_list:
        try:
            font_prop = fm.FontProperties(fname=font_path)
            font_name = font_prop.get_name()
            if any(korean in font_name for korean in korean_font_names):
                if font_name not in korean_fonts:
                    korean_fonts.append(font_name)
        except:
            continue
    
    return korean_fonts

def test_korean_display():
    """
    한글 표시 테스트 함수
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 테스트 텍스트
    test_texts = [
        '한글 폰트 테스트',
        '그래프 제목: 데이터 분석',
        '축 레이블: 값, 빈도',
        '범례: 항목1, 항목2'
    ]
    
    for i, text in enumerate(test_texts):
        ax.text(0.1, 0.8 - i*0.15, text, fontsize=14, transform=ax.transAxes)
    
    ax.set_title('Korean Font Display Test / 한글 폰트 표시 테스트')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    return fig

# 모듈이 직접 실행될 때 폰트 설정 및 테스트
if __name__ == "__main__":
    print("=== Korean Font Setup ===")
    setup_korean_font()
    
    print("\n=== Available Korean Fonts ===")
    available_fonts = get_available_korean_fonts()
    if available_fonts:
        for font in available_fonts:
            print(f"- {font}")
    else:
        print("No Korean fonts detected")
    
    print("\n=== Running Display Test ===")
    fig = test_korean_display()
    plt.show()