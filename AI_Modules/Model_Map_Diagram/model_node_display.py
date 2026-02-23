import sys
from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import torch

# Add the xLSTM_TS_Model directory to sys.path
xLSTM_path = Path(__file__).resolve().parent.parent / "xLSTM_TS_Model"
sys.path.insert(0, str(xLSTM_path))

from xLSTM_TS import xLSTM_TS_Model

project_root = Path(__file__).resolve().parent.parent.parent
csv_path = project_root / "Dataset_Modules" / "dataset_output" / "2015-2025_dataset_denoised.csv"
charts_path = project_root / "AI_Modules" / "Model_Map_Diagram" / "Model_Diagram_output"
df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

sequence_length = 10
n_features = df.shape[1]
input_shape = (sequence_length, n_features)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = xLSTM_TS_Model(input_shape, embedding_dim=64, output_size=1).to(device)
model.eval()

dummy = torch.randn(1, sequence_length, n_features, device=device, requires_grad=True)

executed = []
def save_output(name):
    def hook(module, inp, out):
        try:
            shape = tuple(out.shape) if isinstance(out, torch.Tensor) else str(type(out))
        except Exception:
            shape = str(type(out))
        executed.append((name, module.__class__.__name__, shape))
    return hook

hooks = []
for name, module in model.named_modules():
    if name == "" or name.count('.') != 0:
        continue
    hooks.append(module.register_forward_hook(save_output(name)))

out = model(dummy)
for h in hooks:
    h.remove()

# --- Color map keyed by class name fragments ---
COLOR_MAP = {
    "InputLayer":         "#F0A500",   # orange/gold
    "Dense":              "#A8C8E8",   # light blue
    "LayerNorm":          "#90C97A",   # green
    "Conv":               "#1A3A5C",   # dark navy
    "Lambda":             "#7B3FA0",   # purple
    "LSTM":               "#E07070",   # salmon/red
    "MultiheadAttention": "#F0C030",   # yellow/gold
    "Attention":          "#F0C030",
    "Add":                "#90C97A",   # green (same as LayerNorm)
    "Softmax":            "#00A090",   # teal-green
    "Multiply":           "#007080",   # dark teal
    "default":            "#A8C8E8",   # fallback light blue
}

# Human-readable legend labels matching the image style
LEGEND_LABELS = {
    "InputLayer":         "InputLayer",
    "Dense":              "Dense",
    "LayerNorm":          "LayerNormalization",
    "Conv":               "Conv1D",
    "Lambda":             "Lambda",
    "LSTM":               "LSTM",
    "MultiheadAttention": "MultiHeadAttention",
    "Softmax":            "Softmax",
    "Multiply":           "Multiply",
    "Add":                "Add",
}

def get_color_key(cls_name):
    for key in COLOR_MAP:
        if key.lower() in cls_name.lower():
            return key
    return "default"

def get_color(cls_name):
    return COLOR_MAP[get_color_key(cls_name)]

# --- Build main Graphviz diagram (NO text in nodes) ---
from graphviz import Digraph

g = Digraph("xLSTM_TS_modules", format="png")
g.attr(
    rankdir='LR',
    splines='ortho',
    nodesep='0.15',
    ranksep='0.2',
    size='30,4',        # force wide, short layout (inches)
    dpi='150',
)
g.attr('node', margin='0', penwidth='0.5')

# Input node
g.node("input", label="", shape="box", style="filled",
       fillcolor=COLOR_MAP["InputLayer"],
       width="0.3", height="0.8", fixedsize="true")

prev = "input"
for name, cls, shape in executed:
    # Vary height by output channels to give the 3D-block feel
    if isinstance(shape, tuple) and len(shape) >= 1:
        try:
            channels = int(shape[-1])
        except Exception:
            channels = 32
    else:
        channels = 32

    height = max(0.4, min(2.5, channels / 64.0 * 1.2 + 0.3))
    color = get_color(cls)

    # No label — just colored box like the TF diagram
    g.node(name, label="", shape="box", style="filled",
           fillcolor=color,
           width="0.25", height=str(round(height, 2)), fixedsize="true")
    g.edge(prev, name, arrowsize="0.4")
    prev = name

# Output node
g.node("output", label="", shape="box", style="filled",
       fillcolor=COLOR_MAP["Softmax"],
       width="0.2", height="0.4", fixedsize="true")
g.edge(prev, "output", arrowsize="0.4")

chart_base = charts_path / "xLSTM-TS_chart_modules_colored"
g.render(str(chart_base), cleanup=True)

# --- Build legend using Pillow (compact, two rows like the image) ---
main_img = Image.open(str(chart_base.with_suffix(".png"))).convert("RGBA")

# Legend settings
LEGEND_ITEMS = [
    (k, LEGEND_LABELS[k]) for k in LEGEND_LABELS.keys()
]

SQUARE_SIZE = 36        # colored square size in pixels
TEXT_PADDING = 8        # space between square and text
ITEM_H_GAP = 40         # horizontal gap between legend items
ITEM_V_GAP = 12         # vertical gap between rows
LEGEND_PADDING = 20     # outer padding of the legend strip
LEGEND_BG = (255, 255, 255, 255)

# Try to load a small font; fall back to default
try:
    font = ImageFont.truetype("arial.ttf", 22)
except Exception:
    font = ImageFont.load_default()

# Measure text widths
def text_width(draw, text, font):
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0]

# Decide items per row (up to 5 per row to match the image)
ITEMS_PER_ROW = 5
rows = [LEGEND_ITEMS[i:i+ITEMS_PER_ROW] for i in range(0, len(LEGEND_ITEMS), ITEMS_PER_ROW)]

# Calculate legend strip dimensions
# First pass: measure all items
temp_img = Image.new("RGBA", (1, 1))
temp_draw = ImageDraw.Draw(temp_img)

item_widths = []
for key, label in LEGEND_ITEMS:
    tw = text_width(temp_draw, label, font)
    item_widths.append(SQUARE_SIZE + TEXT_PADDING + tw)

# Max item width per column position
col_widths = []
for col in range(ITEMS_PER_ROW):
    col_items = [item_widths[row * ITEMS_PER_ROW + col]
                 for row in range(len(rows))
                 if row * ITEMS_PER_ROW + col < len(item_widths)]
    col_widths.append(max(col_items) if col_items else 0)

legend_content_width = sum(col_widths) + ITEM_H_GAP * (ITEMS_PER_ROW - 1)
legend_content_height = len(rows) * (SQUARE_SIZE + ITEM_V_GAP) - ITEM_V_GAP
legend_strip_w = max(main_img.width, legend_content_width + 2 * LEGEND_PADDING)
legend_strip_h = legend_content_height + 2 * LEGEND_PADDING

legend_strip = Image.new("RGBA", (legend_strip_w, legend_strip_h), LEGEND_BG)
draw = ImageDraw.Draw(legend_strip)

idx = 0
for row_idx, row in enumerate(rows):
    x = LEGEND_PADDING
    y = LEGEND_PADDING + row_idx * (SQUARE_SIZE + ITEM_V_GAP)
    for col_idx, (key, label) in enumerate(row):
        color_hex = COLOR_MAP.get(key, COLOR_MAP["default"])
        # Convert hex to RGB
        r = int(color_hex[1:3], 16)
        g_c = int(color_hex[3:5], 16)
        b = int(color_hex[5:7], 16)

        # Draw colored square with thin border
        draw.rectangle([x, y, x + SQUARE_SIZE, y + SQUARE_SIZE],
                       fill=(r, g_c, b, 255), outline=(80, 80, 80, 255), width=1)
        # Draw label text
        draw.text((x + SQUARE_SIZE + TEXT_PADDING, y + SQUARE_SIZE // 2),
                  label, fill=(0, 0, 0, 255), font=font, anchor="lm")

        # Advance x by this column's width + gap
        col_advance = col_widths[col_idx] + ITEM_H_GAP
        x += col_advance
        idx += 1

# --- Combine main diagram + legend strip ---
total_width = max(main_img.width, legend_strip_w)
total_height = main_img.height + legend_strip_h

combined = Image.new("RGBA", (total_width, total_height), (255, 255, 255, 255))
combined.paste(main_img, (0, 0))
combined.paste(legend_strip, (0, main_img.height))

combined_path = charts_path / "xLSTM-TS_chart_modules_with_legend.png"
combined.save(str(combined_path))
combined.show()
print(f"Saved to {combined_path}")