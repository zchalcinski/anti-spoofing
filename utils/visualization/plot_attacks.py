# ==============================================================================
# --- SECTION 1: SETUP AND DEPENDENCIES ---
# ==============================================================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import os

# ==============================================================================
# --- SECTION 2: DATA LOADING AND PREPARATION ---
# ==============================================================================
# 1. Load data using a relative path
csv_file = "scientific_exports/wavlm-base-plus/per_attack_eer.csv"

# Validate file existence to prevent runtime errors
if not os.path.exists(csv_file):
    raise FileNotFoundError(f"File not found: {csv_file}. Ensure the script is executed from the correct root directory.")

df = pd.read_csv(csv_file)

# Sort values: target is to display the hardest attacks (highest EER) at the top of the plot
df = df.sort_values(by="EER_%", ascending=False)

# 2. Define specific adversarial attacks based on the ASVspoof protocol
adversarial_attacks = ['A18', 'A20', 'A23', 'A27', 'A30', 'A31', 'A32']

# Modify Y-axis labels to explicitly tag adversarial attacks for clarity
df['Label'] = df['Attack_Type'].apply(
    lambda x: f"{x} (Adv)" if x in adversarial_attacks else x
)

# ==============================================================================
# --- SECTION 3: PLOT CONFIGURATION AND STYLING ---
# ==============================================================================
# 3. Aesthetic global settings
sns.set_theme(style="white", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'sans-serif'

fig, ax = plt.subplots(figsize=(14, 8))

# 4. Construct colormap (Viridis gradient)
# Highest values map to yellow-green, lowest to dark blue
cmap = plt.get_cmap('viridis')
norm = mcolors.Normalize(vmin=df['EER_%'].min(), vmax=df['EER_%'].max())
colors = [cmap(norm(val)) for val in df['EER_%']]

# 5. Draw horizontal bar chart
bars = ax.barh(df['Label'], df['EER_%'], color=colors, height=0.6)

# Invert Y-axis to position the largest bar at the very top
ax.invert_yaxis()

# 6. Grid and axis styling
# Apply vertical dashed grid lines exclusively for readability
ax.xaxis.grid(True, linestyle='--', alpha=0.7)
ax.yaxis.grid(False)

# Enforce a solid black border (spines) around the plot area
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color('black')
    spine.set_linewidth(0.5)

# 7. Apply labels
ax.set_xlabel('Equal Error Rate (EER) [%]', fontsize=13, labelpad=10)
ax.set_ylabel('Typ Ataku', fontsize=13, labelpad=10)
ax.set_title('Wskaźnik EER w zależności od typu ataku na zbiorze ewaluacyjnym', 
             fontsize=15, pad=15)

plt.figtext(0.1, 0.01, "*(Adv) oznacza ataki adwersarialne (Adversarial attacks)", 
            ha="left", fontsize=11, color="gray", style="italic")

# ==============================================================================
# --- SECTION 4: EXPORT ---
# ==============================================================================
# 8. Adjust layout to accommodate the bottom annotation
plt.tight_layout(rect=[0, 0.03, 1, 1]) 

# Save vector (.pdf) and raster (.png) formats locally
plt.savefig("eer_gradient_plot.png", dpi=300, bbox_inches='tight')
plt.savefig("eer_gradient_plot.pdf", bbox_inches='tight')

print("Execution complete. Gradient plot saved to the current working directory.")
plt.show()