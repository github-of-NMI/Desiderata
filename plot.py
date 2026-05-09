import json
import re
import os
import sys
from datetime import date
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

result_path = "results/"
graphics_path = "graphics/"
source_link = "https://github.com/github-of-NMI/Desiderata"

stats = list()
for root, _dir, files in os.walk(result_path):
    for file in files:
        file_path = os.path.join(root, file)
        with open(file_path, "r") as f:
            data = json.load(f)
            
            model = data["model"]
            repeats = data["repeats"]
            qna_count = 0
            nones = 0
            corrs = 0
            wrong = 0
            uniqs = 0
            consistency = list()

            for key in [key for key in data.keys() if re.fullmatch(r"^[a-fA-F0-9]{64}$", key)]:
                qna_count += 1

                nones += data[key]["None_answer"]
                corrs += data[key]["correct"]
                wrong += data[key]["incorrect"]
                uniqs += data[key]["unique_answers"]
                consistency.append(100/data[key]["unique_answers"])

            consistency = sum(consistency)/len(consistency)
            
            totals = repeats * qna_count
            stats.append(
                {
                    "plot_date": f"{date.today()}",
                    "eval_date": data["date"],
                    "model": model,
                    "repeats": repeats,
                    "correct": corrs / (corrs+wrong) * 100,
                    "consistency": consistency,
                }
            )

plot_date = stats[0]["plot_date"]
# --- Data Setup (recreating your df logic) ---
df = pd.DataFrame(stats)
df = df[['model', 'correct', 'consistency', 'repeats', 'eval_date']]
df = df.sort_values(by='correct', ascending=False)

df_display = df.copy()
df_display['correct'] = df['correct'].map('{:.2f}%'.format)
df_display['consistency'] = df['consistency'].map('{:.2f}%'.format)

fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('off')

# Background shading for rows (Zebra Striping)
for i in range(len(df)):
    color = '#f9f9f9' if i % 2 == 0 else 'white'
    ax.axhspan(i - 0.5, i + 0.5, color=color, zorder=0)

# Add the Header Line
ax.axhline(y=-0.5, color='black', linewidth=1.5, xmin=0.05, xmax=0.95)

# Plot the Text
cols = df_display.columns
for i, row in df_display.iterrows():
    y_pos = df_display.index.get_loc(i)
    
    # Model Name (Left Aligned)
    ax.text(0, y_pos, row['model'], va='center', ha='left', weight='bold' if y_pos == 0 else 'normal')
    
    # Correct % (Centered)
    ax.text(4, y_pos, row['correct'], va='center', ha='center', weight='bold')
    
    # Small visual bar for "Correct" performance
    bar_width = (df.loc[i, 'correct'] / 100) * 1.5
    ax.barh(y_pos, bar_width, left=4.5, height=0.4, color='#3498db', alpha=0.6)
    
    # Other stats
    ax.text(7, y_pos, row['consistency'], va='center', ha='center', color='#555555')
    ax.text(9, y_pos, row['eval_date'], va='center', ha='center', fontsize=9, color='#888888')

# 4. Column Headers
header_y = -1
ax.text(0, header_y, "MODEL", va='center', ha='left', weight='bold', color='#333333')
ax.text(4, header_y, "CORRECT", va='center', ha='center', weight='bold', color='#333333')
ax.text(7, header_y, "CONSISTENCY", va='center', ha='center', weight='bold', color='#333333')
ax.text(9, header_y, "DATE", va='center', ha='center', weight='bold', color='#333333')

# Final Touches
ax.set_ylim(len(df) - 0.5, -1.5)
ax.set_xlim(-0.5, 10)
plt.title(f"Model Evaluations • {plot_date}", fontsize=16, fontweight='bold', pad=10, loc='left')

plt.text(0.05, 0.02, f"Source: {source_link}",
         transform=fig.transFigure,
         fontsize=8,
         color='#7f8c8d',
         ha='left')

# Increase the bottom margin slightly so the footer isn't cut off
plt.subplots_adjust(bottom=0.15)

# Save as PNG
filename = os.path.join(graphics_path, f"{plot_date}.png")
plt.savefig(filename, bbox_inches='tight', dpi=300)
print(f"Successfully saved to {filename}")
