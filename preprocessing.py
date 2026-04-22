### [데이터 전처리 및 통합 실행 코드] ###
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

# 1. Environment Setup
os.makedirs('results', exist_ok=True)

def preprocess_wide_to_long(df, target_row_name):
    row = df[df.iloc[:, 0].str.contains(target_row_name, na=False)]
    if row.empty: return pd.Series()
    data = row.iloc[0, 1:].replace(',', '', regex=True).astype(float)
    return data

def run_full_analysis():
    os.makedirs('results', exist_ok=True)
    print("[RUN] Starting Full Analysis (Preprocessing & Visualization)...")

    # 2. Data Loading
    try:
        consumption_df = pd.read_csv('data/최종소비지출.csv', encoding='cp949')
        gdp_df = pd.read_csv('data/성장기여도.csv', encoding='cp949')
        unemp_df = pd.read_csv('data/실업률.csv', encoding='cp949')
        ccsi_file = glob.glob('data/*CCSI*.xlsx')[0]
        ccsi_raw = pd.read_excel(ccsi_file, header=None)
    except Exception as e:
        print(f"[ERROR] File Load Error: {e}")
        return

    # 3. Preprocessing (전처리 핵심 로직)
    alcohol = preprocess_wide_to_long(consumption_df, '주류 및 담배')
    gdp_cont = preprocess_wide_to_long(gdp_df, '가계')

    unemp_row = unemp_df[(unemp_df['성별'] == '계') & (unemp_df['연령계층별'] == '계')]
    unemp_data = unemp_row.iloc[0, 2:].replace('-', np.nan).astype(float)
    unemp_data.index = pd.to_datetime(unemp_data.index, format='%Y.%m')
    unemp_q = unemp_data.resample('Q').mean()
    unemp_q.index = unemp_q.index.to_period('Q').astype(str).str.replace('Q', '.') + '/4'

    ccsi_data = ccsi_raw.iloc[7:, :2]
    ccsi_data.columns = ['Date', 'CCSI']
    ccsi_data['Date'] = pd.to_datetime(ccsi_data['Date'])
    ccsi_data['CCSI'] = pd.to_numeric(ccsi_data['CCSI'])
    ccsi_q = ccsi_data.set_index('Date')['CCSI'].resample('Q').mean()
    ccsi_q.index = ccsi_q.index.to_period('Q').astype(str).str.replace('Q', '.') + '/4'

    df = pd.DataFrame({
        'Alcohol_Tobacco': alcohol,
        'GDP_Contribution': gdp_cont,
        'Unemployment_Rate': unemp_q,
        'CCSI': ccsi_q
    }).dropna()

    # 4. Visualization (1열 배치 및 x축 가독성 개선)
    fig, axes = plt.subplots(4, 1, figsize=(14, 24))
    
    # x축 라벨을 띄엄띄엄 표시하기 위한 설정 (예: 1년 단위)
    xticks_pos = range(0, len(df.index), 4)
    xticks_labels = [df.index[i] for i in xticks_pos]

    # 그래프 1: 트렌드 (소비 vs 실업률)
    ax1 = axes[0]
    ax1.plot(df.index, df['Alcohol_Tobacco'], color='orange', marker='o', label='Alcohol & Tobacco')
    ax1_twin = ax1.twinx()
    ax1_twin.plot(df.index, df['Unemployment_Rate'], color='red', marker='x', linestyle='--', label='Unemployment Rate')
    ax1.set_title('Trend: Consumption vs Unemployment')
    ax1.set_xticks(xticks_pos)
    ax1.set_xticklabels(xticks_labels, rotation=45)
    
    # 그래프 2: 상관관계 (실업률)
    sns.regplot(x='Unemployment_Rate', y='Alcohol_Tobacco', data=df, ax=axes[1], color='red')
    axes[1].set_title('Correlation: Unemployment Rate vs Alcohol/Tobacco')
    
    # 그래프 3: 트렌드 (소비 vs 소비자심리)
    ax3 = axes[2]
    ax3.plot(df.index, df['Alcohol_Tobacco'], color='orange', marker='o')
    ax3_twin = ax3.twinx()
    ax3_twin.plot(df.index, df['CCSI'], color='blue', marker='d', linestyle=':')
    ax3.set_title('Trend: Consumption vs Consumer Sentiment')
    ax3.set_xticks(xticks_pos)
    ax3.set_xticklabels(xticks_labels, rotation=45)

    # 그래프 4: 상관관계 (소비자심리)
    sns.regplot(x='CCSI', y='Alcohol_Tobacco', data=df, ax=axes[3], color='blue')
    axes[3].set_title('Correlation: Consumer Sentiment vs Alcohol/Tobacco')

    plt.tight_layout()
    fig.savefig('results/final_report_english.png', dpi=300)
    df.to_csv('results/final_data_processed.csv', encoding='utf-8-sig')

    print("[SUCCESS] Analysis Completed! Preprocessed data saved to 'results/final_data_processed.csv'")

if __name__ == '__main__':
    run_full_analysis()