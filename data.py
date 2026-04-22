import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. 데이터 불러오기 및 통합 복합 지수 계산
final_analysis_df = pd.read_csv('data/final_data.csv', index_col=0)
df_final_comp = final_analysis_df.copy()

# 정규화 함수 (0~100 범위)
def normalize(series):
    return (series - series.min()) / (series.max() - series.min()) * 100

# 각 지표 정규화
gdp_norm = normalize(df_final_comp['GDP_Contribution'])
ccsi_norm = normalize(df_final_comp['CCSI'])
unemp_norm = normalize(df_final_comp['Unemployment_Rate'])

# 최종 복합 지수 산출 (GDP 20%, CCSI 40%, 실업률 40%)
# 실업률은 낮을수록 좋으므로 100에서 차감
df_final_comp['Final_Economic_Index'] = (0.2 * gdp_norm) + (0.4 * ccsi_norm) + (0.4 * (100 - unemp_norm))

# 2. 시각화
fig, ax1 = plt.subplots(figsize=(16, 8))

# 막대: 술·담배 소비
ax1.bar(df_final_comp.index, df_final_comp['Alcohol_Tobacco'], color='skyblue', alpha=0.6, label='Alcohol & Tobacco Consumption')
ax1.set_ylabel('Consumption Amount', color='blue')
ax1.set_ylim(min(df_final_comp['Alcohol_Tobacco'])*0.9, max(df_final_comp['Alcohol_Tobacco'])*1.1)
ax1.tick_params(axis='x', rotation=45)

# 선: 통합 경제 지수 (GDP+CCSI+Unemployment)
ax2 = ax1.twinx()
ax2.plot(df_final_comp.index, df_final_comp['Final_Economic_Index'], color='darkgreen', marker='D', linewidth=3, label='Integrated Economic Index (GDP+CCSI+Unemp)')
ax2.set_ylabel('Economic Health (Higher = Better)', color='darkgreen')

plt.title('Final Analysis: Consumption vs Integrated Economic Index', fontsize=16)

# 범례 합치기
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper right')

plt.tight_layout()
plt.show()

# 3. 가설 검증 결과 출력 (경기가 어려우면 술/담배 소비가 늘어나는가?)
correlation = df_final_comp[['Alcohol_Tobacco', 'Final_Economic_Index']].corr().iloc[0,1]

print("\n" + "="*60)
print(f" [최종 분석 보고서: 가설 검증]")
print("="*60)
print(f"가설: '나라의 경기가 어려우면 술, 담배의 소비가 늘어난다.'")
print(f"연동 데이터: {len(df_final_comp)}개 분기 (2017~2025)")
print(f"상관계수(Correlation ratio): {correlation:.4f}")
print("-"*60)

print("[분석 결과]")
if correlation < 0:
    # 상관계수가 마이너스면 경제 지수가 낮을 때 소비가 높은 경향이 있음
    if correlation < -0.3:
        print(f"▶ 결론: 사용자님의 가설을 강력하게 지지합니다.")
        print(f"해설: 경제 건강도와 술/담배 소비 간에 뚜렷한 음(-)의 상관관계가 나타납니다.")
        print(f"      즉, 수치적으로 경기가 나빠질수록 소비가 늘어나는 패턴이 확실합니다.")
    else:
        print(f"▶ 결론: 가설을 부분적으로 지지합니다. (약한 상관성)")
        print(f"해설: 경기가 나쁠 때 소비가 늘어나는 경향이 미세하게 나타납니다(상관계수가 음수임).")
        print(f"      다만 상관성이 낮으므로 경기 외의 다른 요인(가격 인상 등)도 큼을 의미합니다.")
else:
    # 상관계수가 플러스면 경제가 좋을 때 소비가 늘거나 경기와 상관 없음
    print(f"▶ 결론: 가설이 성립하지 않습니다.")
    print(f"해설: 경제 지수가 낮아질 때 소비가 늘어나는 패턴이 데이터로 확인되지 않습니다.")
    print(f"      경기가 좋을 때 소비가 늘거나, 경기와 관계없이 일정하게 유지됩니다.")
print("="*60)