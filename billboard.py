import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')


plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

path = r'C:\Users\justi\Downloads\Billboard_Hot_100_Data.csv'
df = pd.read_csv(path)


cols = ['Date', 'Weeks at Number One', 'Label', 'Parent Label',
        'Discogs Genre', 'Discogs Style', 'Artist Structure', 'Multiple Lead Vocalists',
        'Front Person Age', 'Artist Male', 'Artist White', 'Artist Black',
        'Songwriter Male', 'Songwriter White',
        'Artist is a Songwriter', 'Artist is Only Songwriter',
        'Producer Male', 'Producer White', 'Artist is a Producer', 'Artist is Only Producer',
        'Length (Sec)']
df = df[cols]
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)


df['Year'] = df['Date'].dt.year
df['Length (min)'] = df['Length (Sec)'] / 60


def assign_era(date):
    year = date.year
    if year < 2007:
        return 'Pre-Digital'
    elif year < 2020:
        return 'Streaming'
    else:
        return 'Post-Short-Form'

df['Era'] = df['Date'].apply(assign_era)

# binary indicators
df['Solo_Artist'] = (df['Artist Structure'] == 1).astype(int)
df['Duo'] = (df['Artist Structure'] == 2).astype(int)
df['Group'] = (df['Artist Structure'] == 0).astype(int)

df['All_Male_Artist'] = (df['Artist Male'] == 1).astype(int)
df['All_Female_Artist'] = (df['Artist Male'] == 0).astype(int)
df['Mixed_Gender_Artist'] = (df['Artist Male'] == 2).astype(int)

df['All_Male_Songwriter'] = (df['Songwriter Male'] == 1).astype(int)
df['All_Female_Songwriter'] = (df['Songwriter Male'] == 0).astype(int)
df['Mixed_Gender_Songwriter'] = (df['Songwriter Male'] == 2).astype(int)

df['All_Male_Producer'] = (df['Producer Male'] == 1).astype(int)
df['All_Female_Producer'] = (df['Producer Male'] == 0).astype(int)
df['Mixed_Gender_Producer'] = (df['Producer Male'] == 2).astype(int)

df['All_White_Artist'] = (df['Artist White'] == 1).astype(int)
df['All_Black_Artist'] = (df['Artist Black'] == 1).astype(int)
df['All_White_Songwriter'] = (df['Songwriter White'] == 1).astype(int)
df['All_White_Producer'] = (df['Producer White'] == 1).astype(int)

pre_digital = df[df['Era'] == 'Pre-Digital']
streaming = df[df['Era'] == 'Streaming']
post_short = df[df['Era'] == 'Post-Short-Form']
era_order = ['Pre-Digital', 'Streaming', 'Post-Short-Form']

era_colors = {
    'Pre-Digital': '#FF6B6B',
    'Streaming': '#4ECDC4',
    'Post-Short-Form': '#95E1D3'
}

#creating visuals
print(f"Dataset: {len(df)} songs from {df['Date'].min().date()} to {df['Date'].max().date()}\n")

# 1. song length
print("song length analysis")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('RQ1: Song Length Across Consumption Eras', fontsize=14, fontweight='bold')

# Box plot
bp = axes[0].boxplot([pre_digital['Length (min)'].dropna(),
                       streaming['Length (min)'].dropna(),
                       post_short['Length (min)'].dropna()],
                      labels=['Pre-Digital\n(1958-2006)', 'Streaming\n(2007-2019)', 'Post-Short-Form\n(2020-2025)'],
                      patch_artist=True, showmeans=True, meanline=True)
for patch, color in zip(bp['boxes'], era_colors.values()):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[0].set_ylabel('Song Length (minutes)', fontsize=11)
axes[0].set_title('Distribution of Song Length by Era', fontsize=12)
axes[0].grid(True, alpha=0.3, axis='y')

# Add statistics text
contingency = pd.crosstab(df['Era'], pd.cut(df['Length (min)'], bins=3, labels=['Short', 'Medium', 'Long']))
chi2, p_val, dof, expected = chi2_contingency(contingency)
axes[0].text(0.02, 0.98, f'χ² test: p={p_val:.4f}',
            transform=axes[0].transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Time series
yearly_length = df.groupby('Year')['Length (min)'].mean()
axes[1].plot(yearly_length.index, yearly_length.values, linewidth=2, color='#2C3E50', marker='o', markersize=3)
axes[1].axvline(2007, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Era boundaries')
axes[1].axvline(2020, color='red', linestyle='--', linewidth=2, alpha=0.7)
axes[1].set_xlabel('Year', fontsize=11)
axes[1].set_ylabel('Average Song Length (minutes)', fontsize=11)
axes[1].set_title('Song Length Over Time', fontsize=12)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('song_length.png', dpi=300, bbox_inches='tight')
print("saved song length plot")
plt.close()

# 2. "artist structure"
print("artist structure analysis")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('RQ2: Artist Structure Across Eras', fontsize=14, fontweight='bold')

# Stacked bar chart
structure_by_era = df.groupby('Era')[['Solo_Artist', 'Duo', 'Group']].mean() * 100
structure_by_era = structure_by_era.reindex(era_order)
structure_data = structure_by_era.T

x = np.arange(len(era_order))
width = 0.6
axes[0].bar(x, structure_data.loc['Solo_Artist'], width, label='Solo',
           color='#E74C3C', edgecolor='black', alpha=0.8)
axes[0].bar(x, structure_data.loc['Duo'], width, bottom=structure_data.loc['Solo_Artist'],
           label='Duo', color='#F39C12', edgecolor='black', alpha=0.8)
axes[0].bar(x, structure_data.loc['Group'], width,
           bottom=structure_data.loc['Solo_Artist'] + structure_data.loc['Duo'],
           label='Group (3+)', color='#27AE60', edgecolor='black', alpha=0.8)
axes[0].set_ylabel('Percentage', fontsize=11)
axes[0].set_title('Artist Structure Distribution by Era', fontsize=12)
axes[0].set_xticks(x)
axes[0].set_xticklabels(['Pre-Digital\n(1958-2006)', 'Streaming\n(2007-2019)', 'Post-Short-Form\n(2020-2025)'], fontsize=9)
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')

# Chi-square test
contingency_solo = pd.crosstab(df['Era'], df['Solo_Artist'])
chi2, p_val, dof, expected = chi2_contingency(contingency_solo)
axes[0].text(0.02, 0.98, f'χ² test: p={p_val:.4f}',
            transform=axes[0].transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Time series
yearly_solo = df.groupby('Year')['Solo_Artist'].mean() * 100
yearly_group = df.groupby('Year')['Group'].mean() * 100
axes[1].plot(yearly_solo.index, yearly_solo.values, linewidth=2.5, color='#E74C3C', marker='o', markersize=3, label='Solo')
axes[1].plot(yearly_group.index, yearly_group.values, linewidth=2.5, color='#27AE60', marker='o', markersize=3, label='Group')
axes[1].axvline(2007, color='red', linestyle='--', linewidth=2, alpha=0.7)
axes[1].axvline(2020, color='red', linestyle='--', linewidth=2, alpha=0.7)
axes[1].set_xlabel('Year', fontsize=11)
axes[1].set_ylabel('Percentage', fontsize=11)
axes[1].set_title('Solo vs Group Artists Over Time', fontsize=12)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('artist_structure.png', dpi=300, bbox_inches='tight')
print("saved artist structure plot")
plt.close()

# 3. demographic evolution

print("demographics analysis")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('RQ3: Racial & Gender Demographics Over Time', fontsize=14, fontweight='bold')

gender_by_era = df.groupby('Era')[['All_Male_Artist', 'All_Female_Artist', 'Mixed_Gender_Artist']].mean() * 100
gender_by_era = gender_by_era.reindex(era_order)
racial_by_era = df.groupby('Era')[['All_White_Artist', 'All_Black_Artist']].mean() * 100
racial_by_era = racial_by_era.reindex(era_order)
songwriter_gender_by_era = df.groupby('Era')[['All_Male_Songwriter', 'All_Female_Songwriter', 'Mixed_Gender_Songwriter']].mean() * 100
songwriter_gender_by_era = songwriter_gender_by_era.reindex(era_order)
producer_gender_by_era = df.groupby('Era')[['All_Male_Producer', 'All_Female_Producer', 'Mixed_Gender_Producer']].mean() * 100
producer_gender_by_era = producer_gender_by_era.reindex(era_order)

x = np.arange(len(era_order))
width = 0.25

# Artist Gender
axes[0, 0].bar(x - width, gender_by_era['All_Male_Artist'], width, label='All Male',
              color='#3498DB', edgecolor='black', alpha=0.8)
axes[0, 0].bar(x, gender_by_era['All_Female_Artist'], width, label='All Female',
              color='#E91E63', edgecolor='black', alpha=0.8)
axes[0, 0].bar(x + width, gender_by_era['Mixed_Gender_Artist'], width, label='Mixed',
              color='#9C27B0', edgecolor='black', alpha=0.8)
axes[0, 0].set_ylabel('Percentage', fontsize=10)
axes[0, 0].set_title('Artist Gender by Era', fontsize=11, fontweight='bold')
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(['Pre-Digital', 'Streaming', 'Post-SF'], fontsize=9)
axes[0, 0].legend(fontsize=8)
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Artist Race
axes[0, 1].bar(x - width/2, racial_by_era['All_White_Artist'], width, label='All White',
              color='#3498DB', edgecolor='black', alpha=0.8)
axes[0, 1].bar(x + width/2, racial_by_era['All_Black_Artist'], width, label='All Black',
              color='#2ECC71', edgecolor='black', alpha=0.8)
axes[0, 1].set_ylabel('Percentage', fontsize=10)
axes[0, 1].set_title('Artist Race by Era', fontsize=11, fontweight='bold')
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(['Pre-Digital', 'Streaming', 'Post-SF'], fontsize=9)
axes[0, 1].legend(fontsize=8)
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Songwriter Gender
axes[0, 2].bar(x - width, songwriter_gender_by_era['All_Male_Songwriter'], width, label='All Male',
              color='#3498DB', edgecolor='black', alpha=0.8)
axes[0, 2].bar(x, songwriter_gender_by_era['All_Female_Songwriter'], width, label='All Female',
              color='#E91E63', edgecolor='black', alpha=0.8)
axes[0, 2].bar(x + width, songwriter_gender_by_era['Mixed_Gender_Songwriter'], width, label='Mixed',
              color='#9C27B0', edgecolor='black', alpha=0.8)
axes[0, 2].set_ylabel('Percentage', fontsize=10)
axes[0, 2].set_title('Songwriter Gender by Era', fontsize=11, fontweight='bold')
axes[0, 2].set_xticks(x)
axes[0, 2].set_xticklabels(['Pre-Digital', 'Streaming', 'Post-SF'], fontsize=9)
axes[0, 2].legend(fontsize=8)
axes[0, 2].grid(True, alpha=0.3, axis='y')

# Producer Gender
axes[1, 0].bar(x - width, producer_gender_by_era['All_Male_Producer'], width, label='All Male',
              color='#3498DB', edgecolor='black', alpha=0.8)
axes[1, 0].bar(x, producer_gender_by_era['All_Female_Producer'], width, label='All Female',
              color='#E91E63', edgecolor='black', alpha=0.8)
axes[1, 0].bar(x + width, producer_gender_by_era['Mixed_Gender_Producer'], width, label='Mixed',
              color='#9C27B0', edgecolor='black', alpha=0.8)
axes[1, 0].set_ylabel('Percentage', fontsize=10)
axes[1, 0].set_title('Producer Gender by Era', fontsize=11, fontweight='bold')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(['Pre-Digital', 'Streaming', 'Post-SF'], fontsize=9)
axes[1, 0].legend(fontsize=8)
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Time series: Female representation
yearly_female_artist = df.groupby('Year')['All_Female_Artist'].mean() * 100
yearly_female_songwriter = df.groupby('Year')['All_Female_Songwriter'].mean() * 100
yearly_female_producer = df.groupby('Year')['All_Female_Producer'].mean() * 100

axes[1, 1].plot(yearly_female_artist.index, yearly_female_artist.values, linewidth=2,
               color='#E91E63', marker='o', markersize=2, label='Artist')
axes[1, 1].plot(yearly_female_songwriter.index, yearly_female_songwriter.values, linewidth=2,
               color='#9C27B0', marker='o', markersize=2, label='Songwriter')
axes[1, 1].plot(yearly_female_producer.index, yearly_female_producer.values, linewidth=2,
               color='#673AB7', marker='o', markersize=2, label='Producer')
axes[1, 1].axvline(2007, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
axes[1, 1].axvline(2020, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
axes[1, 1].set_xlabel('Year', fontsize=10)
axes[1, 1].set_ylabel('Percentage', fontsize=10)
axes[1, 1].set_title('All-Female Representation Over Time', fontsize=11, fontweight='bold')
axes[1, 1].legend(fontsize=8)
axes[1, 1].grid(True, alpha=0.3)

# Time series: Black artist representation
yearly_black_artist = df.groupby('Year')['All_Black_Artist'].mean() * 100
axes[1, 2].plot(yearly_black_artist.index, yearly_black_artist.values, linewidth=2.5,
               color='#2ECC71', marker='o', markersize=3)
axes[1, 2].axvline(2007, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
axes[1, 2].axvline(2020, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
axes[1, 2].set_xlabel('Year', fontsize=10)
axes[1, 2].set_ylabel('Percentage', fontsize=10)
axes[1, 2].set_title('All-Black Artist Representation Over Time', fontsize=11, fontweight='bold')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('demographics.png', dpi=300, bbox_inches='tight')
print("saved demographics plot")
plt.close()

# 4. creative control
print("creative control analysis")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('RQ4: Artist Creative Control Across Eras', fontsize=14, fontweight='bold')

# Artist as Songwriter
songwriter_data = df.groupby('Era')[['Artist is a Songwriter', 'Artist is Only Songwriter']].mean() * 100
songwriter_data = songwriter_data.reindex(era_order)
x = np.arange(len(era_order))
width = 0.35
axes[0].bar(x - width/2, songwriter_data['Artist is a Songwriter'], width,
           label='Co-Writes', color='#9B59B6', edgecolor='black', alpha=0.8)
axes[0].bar(x + width/2, songwriter_data['Artist is Only Songwriter'], width,
           label='Only Writer', color='#3498DB', edgecolor='black', alpha=0.8)
axes[0].set_ylabel('Percentage', fontsize=11)
axes[0].set_title('Artist Songwriting by Era', fontsize=12)
axes[0].set_xticks(x)
axes[0].set_xticklabels(['Pre-Digital\n(1958-2006)', 'Streaming\n(2007-2019)', 'Post-Short-Form\n(2020-2025)'], fontsize=9)
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')

# Chi-square test
contingency_songwriter = pd.crosstab(df['Era'], df['Artist is a Songwriter'])
chi2, p_val, dof, expected = chi2_contingency(contingency_songwriter)
axes[0].text(0.02, 0.98, f'χ² test: p={p_val:.4f}',
            transform=axes[0].transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Artist as Producer
producer_data = df.groupby('Era')[['Artist is a Producer', 'Artist is Only Producer']].mean() * 100
producer_data = producer_data.reindex(era_order)
axes[1].bar(x - width/2, producer_data['Artist is a Producer'], width,
           label='Co-Produces', color='#E67E22', edgecolor='black', alpha=0.8)
axes[1].bar(x + width/2, producer_data['Artist is Only Producer'], width,
           label='Only Producer', color='#E74C3C', edgecolor='black', alpha=0.8)
axes[1].set_ylabel('Percentage', fontsize=11)
axes[1].set_title('Artist Production by Era', fontsize=12)
axes[1].set_xticks(x)
axes[1].set_xticklabels(['Pre-Digital\n(1958-2006)', 'Streaming\n(2007-2019)', 'Post-Short-Form\n(2020-2025)'], fontsize=9)
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

# Chi-square test
contingency_producer = pd.crosstab(df['Era'], df['Artist is a Producer'])
chi2, p_val, dof, expected = chi2_contingency(contingency_producer)
axes[1].text(0.02, 0.98, f'χ² test: p={p_val:.4f}',
            transform=axes[1].transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('creative_control.png', dpi=300, bbox_inches='tight')
print("saved creative control plot")
plt.close()


print("success")