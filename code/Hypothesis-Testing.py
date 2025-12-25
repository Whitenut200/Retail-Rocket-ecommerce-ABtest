import pandas as pd
import numpy as np
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

class MultiPeriodABTestAnalyzer:
    def __init__(self, csv_file):
        """CSV 파일 로드"""
        self.df = pd.read_csv(csv_file)
        print(f"CSV 파일 로드 완료: {len(self.df)}행")
        
        # 기간별 데이터 확인
        periods = sorted(self.df['기간'].unique())
        print(f"분석 기간: {periods}")
        
        self.results = {}
    
    def get_value(self, period, group, metric):
        """특정 기간, 그룹, 지표의 값 추출"""
        filtered = self.df[
            (self.df['기간'] == period) & 
            (self.df['ab_group'] == group) & 
            (self.df['구분'] == metric)
        ]
        
        if len(filtered) > 0:
            # 카테고리별 합계 반환
            return filtered['값'].sum()
        return 0
    
    def z_test_proportions(self, period, metric_name, num_suffix, den_suffix):
        """비율 Z-test 수행"""
        # A그룹 데이터
        a_num = self.get_value(period, 'A', num_suffix)
        a_den = self.get_value(period, 'A', den_suffix)
        
        # B그룹 데이터
        b_num = self.get_value(period, 'B', num_suffix)
        b_den = self.get_value(period, 'B', den_suffix)
        
        # 유효성 검사
        if a_den == 0 or b_den == 0:
            return None
            
        p1 = a_num / a_den
        p2 = b_num / b_den
        
        # 풀드 비율
        p_pooled = (a_num + b_num) / (a_den + b_den)
        
        if p_pooled == 0 or p_pooled == 1:
            return None
            
        # 표준오차
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/a_den + 1/b_den))
        
        if se == 0:
            return None
            
        # Z 통계량 및 p-value
        z_stat = (p2 - p1) / se
        p_value = 2 * (1 - norm.cdf(abs(z_stat)))
        effect_size = p2 - p1
        
        # 신뢰구간 계산 (95%)
        ci_lower = effect_size - 1.96 * se
        ci_upper = effect_size + 1.96 * se

        # 산출 지표 섡덩덩
        return {
            'period': period,
            'metric': metric_name,
            'a_rate': f"{p1:.4f} ({a_num:.0f}/{a_den:.0f})",
            'b_rate': f"{p2:.4f} ({b_num:.0f}/{b_den:.0f})",
            'z_stat': z_stat,
            'p_value': p_value,
            'effect_size': effect_size,
            'significant': p_value < 0.05,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }
    
    def run_all_tests(self):
        """모든 기간에 대해 5가지 검정 수행"""
        periods = sorted(self.df['기간'].unique())
        
        # 검정할 지표들
        tests = [
            ('CVR', 'cvr_numerator', 'cvr_denominator'),
            ('View→Cart', 'view_to_cart_numerator', 'view_to_cart_denominator'),
            ('Cart→Purchase', 'cart_to_purchase_numerator', 'cart_to_purchase_denominator'),
            ('Direct경로', 'direct_numerator', 'direct_denominator'),
            ('Cart경로', 'via_cart_numerator', 'via_cart_denominator')
        ]
      
        for period in periods:
            period_results = {}
            
            for test_name, num_suffix, den_suffix in tests:
                result = self.z_test_proportions(period, test_name, num_suffix, den_suffix)
                if result:
                    period_results[test_name] = result
            
            if period_results:
                self.results[period] = period_results
                print(f"{period}일: {len(period_results)}개 검정 완료")
    
    def print_results(self):
        """결과 출력"""
        print("\n" + "="*80)
        print("A/B 테스트 분석 결과")
        print("="*80)
        
        for period in sorted(self.results.keys()):
            print(f"\n{period}일 기간")
            print("-" * 50)
            
            for test_name, result in self.results[period].items():
                significant_mark = "" if result['significant'] else ""
                direction = "B가 더 높음" if result['effect_size'] > 0 else "A가 더 높음"
                
                print(f"{significant_mark} {test_name:12} | A: {result['a_rate']:15} B: {result['b_rate']:15} | "
                      f"p={result['p_value']:.4f} | {direction}")
    
    def to_dataframe(self):
        """결과를 DataFrame으로 변환"""
        rows = []
        
        for period, period_results in self.results.items():
            for test_name, result in period_results.items():
                rows.append({
                    '기간': period,
                    '지표': test_name,
                    'A그룹': result['a_rate'],
                    'B그룹': result['b_rate'],
                    'Z통계량': result['z_stat'],
                    'p_value': result['p_value'],
                    '효과크기': result['effect_size'],
                    '유의함': result['significant']
                })
        
        return pd.DataFrame(rows)
    
    def to_melted_dataframe(self):
        """결과를 MELT 형태로 변환 (태블로/시각화용)"""
        rows = []
        
        for period, period_results in self.results.items():
            for test_name, result in period_results.items():
                # 각 통계량을 별도 행으로 추가
                rows.extend([{
                        '기간': period,
                        '검정이름': test_name,
                        '구분': 'P-VALUE',
                        '값': result['p_value']
                    },
                    {
                        '기간': period,
                        '검정이름': test_name,
                        '구분': '효과크기',
                        '값': result['effect_size']
                    },
                    {
                        '기간': period,
                        '검정이름': test_name,
                        '구분': 'Z통계량',
                        '값': result['z_stat']
                    },                    
                    {
                        '기간': period,
                        '검정이름': test_name,
                        '구분': '신뢰구간_min',
                        '값': result['ci_lower']
                    },
                    {
                        '기간': period,
                        '검정이름': test_name,
                        '구분': '신뢰구간_max',
                        '값': result['ci_upper']
                    }])
            
        return pd.DataFrame(rows)
    
    def export_excel(self, filename='ab_test_results.xlsx'):
        """엑셀로 저장 (일반 형태 + MELT 형태)"""
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # 일반 형태
            results_df = self.to_dataframe()
            results_df.to_excel(writer, sheet_name='일반형태', index=False)
            
            # MELT 형태 (태블로용)
            melted_df = self.to_melted_dataframe()
            melted_df.to_excel(writer, sheet_name='MELT형태', index=False)
            
            print(f"\ 결과 저장: {filename}")
            print("  - 일반형태 시트: 기본 분석 결과")
            print("  - MELT형태 시트: 태블로/시각화용")
    
    def get_melted_results(self):
        """MELT 형태 결과 반환"""
        return self.to_melted_dataframe()

# 사용법
def analyze_csv(csv_file):
    """CSV 파일 분석 (원라이너)"""
    analyzer = MultiPeriodABTestAnalyzer(csv_file)
    analyzer.run_all_tests()
    analyzer.print_results()
    analyzer.export_excel()
    
    # MELT 형태 결과 미리보기
    melted_df = analyzer.get_melted_results()
    print(f"\n MELT 형태 데이터 샘플:")
    print(melted_df.head(10))
    
    return analyzer

# 예시
analyzer = analyze_csv("지표산출- case2+기간.csv")
analyzer
