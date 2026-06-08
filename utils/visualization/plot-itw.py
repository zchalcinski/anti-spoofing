# ==============================================================================
# --- SECTION 1: SETUP AND DEPENDENCIES ---
# ==============================================================================
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# ==============================================================================
# --- SECTION 2: DATA LOADING AND PREPARATION ---
# ==============================================================================
# 1. Load data using a relative path
csv_file = "cross_corpus_tts_results.csv"

# Validate file existence to prevent runtime errors
if not os.path.exists(csv_file):
    raise FileNotFoundError(f"File not found: {csv_file}. Ensure the script is executed from the correct directory.")

df = pd.read_csv(csv_file)

# 2. Data Preparation
# Sort by EER ascending so that when plotted on the Y-axis, the highest error is at the top
df = df.sort_values(by="EER (%)", ascending=True)

engines = df['Silnik TTS/AI']
eer_values = df['EER (%)']
sdr_values = df['SDR (%)']

# Y-axis positions for individual engines
y = np.arange(len(engines))
height = 0.35 # Thickness of a single bar

# ==============================================================================
# --- SECTION 3: PLOT CONFIGURATION AND STYLING ---
# ==============================================================================
# 3. Aesthetic global settings (Academic standard)
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'sans-serif'

fig, ax = plt.subplots(figsize=(14, 9))

# 4. Draw horizontal bars
# EER bars (Red - symbolizing error/vulnerability)
bars_eer = ax.barh(y + height/2, eer_values, height, 
                   label='EER - Równa Stopa Błędów (mniej = lepiej)', 
                   color='#e74c3c', edgecolor='black', linewidth=0.7)

# SDR bars (Green - symbolizing correct detection/success)
bars_sdr = ax.barh(y - height/2, sdr_values, height, 
                   label='SDR - Wskaźnik Wykrywalności (więcej = lepiej)', 
                   color='#2ecc71', edgecolor='black', linewidth=0.7)

# 5. Function to append numerical values at the end of the bars
def add_labels(bars):
    for bar in bars:
        width = bar.get_width()
        if width > 0.01: # Do not display 0.0% if the bar is practically non-existent (e.g., OpenAI SDR)
            ax.annotate(f'{width:.1f}%',
                        xy=(width, bar.get_y() + bar.get_height() / 2),
                        xytext=(4, 0), # Shift to the right
                        textcoords="offset points",
                        ha='left', va='center', fontsize=11, fontweight='bold', color='#333333')
        elif width == 0:
            # Highlight total failures in red
            ax.annotate('0.0%',
                        xy=(width, bar.get_y() + bar.get_height() / 2),
                        xytext=(4, 0),
                        textcoords="offset points",
                        ha='left', va='center', fontsize=11, fontweight='bold', color='#e74c3c')

add_labels(bars_eer)
add_labels(bars_sdr)

# 6. Axis and grid styling
ax.set_yticks(y)
ax.set_yticklabels(engines, fontsize=12, fontweight='bold')
ax.set_xlabel('Wartość procentowa [%]', fontsize=14, labelpad=12, fontweight='bold')
ax.set_title('Zdolność modelu WavLM do generalizacji: EER vs SDR (Test In-The-Wild)', 
             fontsize=16, fontweight='bold', pad=20)

# Set X-axis limit to ~105% to accommodate text annotations above long bars
ax.set_xlim(0, 105)

# 7. Legend and background styling
ax.xaxis.grid(True, linestyle='--', alpha=0.7)
ax.yaxis.grid(False)

# Legend centered at the very top (above the plotting area), split into two columns (ncol=2)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2, 
          framealpha=1, edgecolor='black', fontsize=12)

# Remove top and right borders (despine) for a cleaner look
sns.despine(left=True, bottom=False)


# ==============================================================================
# --- SECTION 4: EXPORT ---
# ==============================================================================
# 8. High-quality export
plt.tight_layout()
plot_filename_png = "in_the_wild_eer_sdr_plot.png"
plot_filename_pdf = "in_the_wild_eer_sdr_plot.pdf"

plt.savefig(plot_filename_png, dpi=300, bbox_inches='tight')
plt.savefig(plot_filename_pdf, bbox_inches='tight')

print(f"Execution complete. Plots generated: {plot_filename_png} and {plot_filename_pdf}")
plt.show()