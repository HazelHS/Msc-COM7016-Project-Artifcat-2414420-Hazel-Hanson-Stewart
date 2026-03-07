# AI declaration:
# Github copilot was used for portions of the planning, research, feedback and editing of the software artefact. Mostly utilised for syntax, logic and error checking with ChatGPT and Claude Sonnet 4.6 used as the models.

"""
This script model_node_display.py generates a visual architecture diagram 
for any model script selected in the "AI Model Designs" pipeline stage. 
Designed to work with the xLSTM-TS and MEMD-TCN models but should be robust to a 
wide range of PyTorch nn.Module architectures.
"""

import sys
import importlib.util
import inspect
from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
from graphviz import Digraph

# Resolve the model script path from argv
if len(sys.argv) < 2:
    print("[ERROR] Usage: python model_node_display.py <model_script.py>",
          file=sys.stderr)
    sys.exit(1)

model_script_path = Path(sys.argv[1]).resolve()
if not model_script_path.exists():
    print(f"[ERROR] Model script not found: {model_script_path}", file=sys.stderr)
    sys.exit(1)

model_name = model_script_path.stem
print(f"Generating diagram for: {model_name}")
print(f"  Source: {model_script_path}")

# Fixed project paths
project_root = Path(__file__).resolve().parent.parent.parent
csv_path     = project_root / "Dataset_Modules" / "dataset_output" / "2015-2025_dataset_denoised.csv"
charts_path  = project_root / "AI_Modules" / "Model_Map_Diagram" / "Model_Diagram_output"
charts_path.mkdir(parents=True, exist_ok=True)

# Dynamically import the model module
model_dir = str(model_script_path.parent)
if model_dir not in sys.path:
    sys.path.insert(0, model_dir)

spec   = importlib.util.spec_from_file_location(model_name, model_script_path)
module = importlib.util.module_from_spec(spec)
module.__name__ = model_name

try:
    spec.loader.exec_module(module)
except Exception as exc:
    print(f"[ERROR] Failed to import {model_script_path.name}: {exc}", file=sys.stderr)
    sys.exit(1)

# Find the primary model class
def find_model_class(mod) -> type | None: # (Anthropic, 2026)
    """Find the primary nn.Module subclass defined in mod.

    Prefers classes whose name contains 'Model'. If none match, returns
    the last nn.Module subclass found whose __module__ matches the file.

    Args:
        mod: Dynamically imported Python module object.

    Returns:
        The primary model class, or None if no nn.Module subclass is found.
    """
    preferred, fallback = None, None
    for obj_name, obj in mod.__dict__.items():
        if not (isinstance(obj, type) and issubclass(obj, nn.Module)):
            continue
        if getattr(obj, "__module__", None) != model_name:
            continue
        if "Model" in obj_name:
            preferred = obj
        fallback = obj
    return preferred if preferred is not None else fallback

model_class = find_model_class(module)
if model_class is None:
    print(f"[ERROR] No nn.Module subclass found in {model_script_path.name}.",
          file=sys.stderr)
    sys.exit(1)

print(f"  Found model class: {model_class.__name__}")

# Build dummy input dimensions
if csv_path.exists():
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    n_features = df.shape[1]
    print(f"  Dataset loaded — n_features={n_features}")
else:
    print(f"[WARN] Dataset CSV not found. Defaulting to n_features=10.")
    n_features = 10

sequence_length = 10
input_shape     = (sequence_length, n_features)
device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Inspect constructor
def _get_init_params(cls) -> list[str]: # (Anthropic, 2026)
    """Return constructor parameter names for cls, excluding self.

    Args:
        cls: Class whose __init__ signature will be inspected.

    Returns:
        A list of parameter name strings, or an empty list if inspection fails.
    """
    try:
        sig = inspect.signature(cls.__init__)
        return [k for k in sig.parameters if k != "self"]
    except Exception:
        return []

def _try_instantiate(cls, candidates: list[dict], device) -> nn.Module | None: # (Anthropic, 2026)
    """Try each kwarg dict in candidates in order and return the first successful instantiation.

    Args:
        cls: The nn.Module subclass to instantiate.
        candidates: Ordered list of keyword-argument dicts to attempt.
        device: Torch device to move the model to on success.

    Returns:
        An instantiated model moved to device, or None if all attempts fail.
    """
    accepted = set(_get_init_params(cls))
    for kwargs in candidates:
        for kw in [kwargs, {k: v for k, v in kwargs.items() if k in accepted}]:
            try:
                m = cls(**kw).to(device)
                print(f"  Instantiated with kwargs: {list(kw.keys()) or '(none)'}")
                return m
            except Exception:
                continue
    return None

instantiation_candidates = [
    # xLSTM-style
    {"input_shape": input_shape, "embedding_dim": 64, "output_size": 1},
    {"input_shape": input_shape, "embedding_dim": 64},
    {"input_shape": input_shape},
    # MEMD-TCN-style — reduced K and max_imfs so diagram generation is fast
    {"in_channels": 5, "kernel_size": 2, "dilations": [1, 2, 4],
     "dropout": 0.2, "K": 64, "max_imfs": 3},
    # Bare default
    {},
]

model = _try_instantiate(model_class, instantiation_candidates, device)
if model is None:
    print(f"[ERROR] Could not instantiate {model_class.__name__} with any known signature.",
          file=sys.stderr)
    sys.exit(1)

model.eval()
print(f"  Instantiated {model_class.__name__} on {device}")

# Colour map
# Keys are matched as substrings (case-insensitive) against module class names.
# Order matters — first match wins.
COLOR_MAP: dict[str, str] = {
    "InputLayer":         "#F0A500",   # orange/gold  — input node
    "CausalConv":         "#1A3A5C",   # dark navy    — causal conv
    "Conv":               "#2E6DA4",   # medium navy  — generic conv
    "ResidualBlock":      "#C8A8E8",   # lavender     — residual block
    "LayerNorm":          "#90C97A",   # green        — normalisation
    "MultiheadAttention": "#F0C030",   # yellow/gold  — attention
    "LSTM":               "#E07070",   # salmon/red   — LSTM
    "Linear":             "#A8C8E8",   # light blue   — linear / dense
    "ReLU":               "#FFD580",   # light amber  — ReLU activation
    "ELU":                "#FFB347",   # darker amber — ELU activation (mLSTM gating)
    "Tanh":               "#B0E0E6",   # powder blue  — Tanh (attention scoring)
    "Dropout":            "#D0D0D0",   # light grey   — dropout
    "Sequential":         "#E8C8A8",   # peach        — sequential wrapper
    "ModuleList":         "#A8E8C8",   # mint         — module list
    "Output":             "#00A090",   # teal         — output node (synthetic)
    "default":            "#A8C8E8",   # fallback
}

LEGEND_LABELS: dict[str, str] = {
    "InputLayer":         "Input",
    "CausalConv":         "CausalConv1d",
    "Conv":               "Conv1d",
    "ResidualBlock":      "ResidualBlock",
    "LayerNorm":          "LayerNorm",
    "MultiheadAttention": "MultiHeadAttention",
    "LSTM":               "LSTM",
    "Linear":             "Linear / Dense",
    "ReLU":               "ReLU Activation",
    "ELU":                "ELU Activation (exp. gating)",
    "Tanh":               "Tanh (attention scoring)",
    "Dropout":            "Dropout",
    "Sequential":         "Sequential",
    "ModuleList":         "ModuleList (per-IMF TCNs)",
    "Output":             "Output",
    "default":            "Other",
}

def get_color_key(cls_name: str) -> str: # (Anthropic, 2026)
    """Return the COLOR_MAP key whose label is a case-insensitive substring of cls_name."""
    for key in COLOR_MAP:
        if key == "default":
            continue
        if key.lower() in cls_name.lower():
            return key
    return "default"

def get_color(cls_name: str) -> str: # (Anthropic, 2026)
    """Return the hex fill colour for cls_name from COLOR_MAP.

    Args:
        cls_name: PyTorch module class name string.

    Returns:
        A hex colour string such as '#2E6DA4'.
    """
    return COLOR_MAP[get_color_key(cls_name)]

# Layer-Level forward-hook tracing Registers hooks on every submodule not just top-level, that has no nn.Module children of its own. i.e. true leaf layers, this captures Conv1d, Linear, ReLU, Dropout etc. inside ResidualBlock.
# Modules to skip — pure containers add noise without meaning in the diagram
SKIP_TYPES = (nn.Sequential, nn.ModuleList, nn.ModuleDict)

# Maximum nesting depth to register hooks (avoids exploding graphs)
MAX_DEPTH = 6

def _is_leaf_layer(mod: nn.Module) -> bool: # (Anthropic, 2026)
    """Return True if mod has no nn.Module children (i.e. is a computational leaf)."""
    return not any(True for _ in mod.children())

def _module_depth(name: str) -> int: # (Anthropic, 2026)
    """Return the nesting depth of a dot-separated module path.

    Args:
        name: Dot-separated module name, e.g. 'network.0.conv1'.

    Returns:
        The number of path segments, or 0 for an empty string.
    """
    return len(name.split(".")) if name else 0

executed: list[tuple[str, str, object]] = []  # (full_name, class_name, output_shape)

def make_hook(full_name: str): # (Anthropic, 2026)
    """Create a forward hook that records the output shape of a named module.

    Args:
        full_name: Dot-separated qualified name of the module being hooked.

    Returns:
        A forward-hook callable that appends a tuple
        (full_name, class_name, shape) to the outer executed list after
        each forward pass through the hooked module.
    """
    def hook(mod, inp, out):
        try:
            shape = tuple(out.shape) if isinstance(out, torch.Tensor) else None
        except Exception:
            shape = None
        executed.append((full_name, mod.__class__.__name__, shape))
    return hook

hooks = []
for full_name, mod in model.named_modules():
    if full_name == "":
        continue
    depth = _module_depth(full_name)
    if depth > MAX_DEPTH:
        continue
    if isinstance(mod, SKIP_TYPES):
        continue
    if _is_leaf_layer(mod):
        hooks.append(mod.register_forward_hook(make_hook(full_name)))

# Dummy tensors
dummy_BTC  = torch.randn(1, sequence_length, n_features, device=device)   # [B, T, C] xLSTM
dummy_BCT  = torch.randn(1, 5, sequence_length, device=device)             # [B, C, T] TCN

forward_strategies = [
    # xLSTM-TS style
    lambda m: m(dummy_BTC),
    # MEMD-TCN style — forward(x, imf_idx)
    lambda m: m(dummy_BCT, 0),
    # Generic fallbacks
    lambda m: m(torch.randn(1, sequence_length, 5, device=device)),
    lambda m: m(torch.randn(1, n_features, sequence_length, device=device)),
]

trace_success = False
for strategy in forward_strategies:
    executed.clear()
    try:
        with torch.no_grad():
            _ = strategy(model)
        if executed:
            trace_success = True
            print(f"  Traced {len(executed)} leaf layer(s)")
            break
    except Exception as e:
        print(f"  [INFO] Strategy failed ({e.__class__.__name__}: {e}), trying next...")

for h in hooks:
    h.remove()

if not trace_success or not executed:
    print("[WARN] No layers traced. Diagram will only show input→output.")

# De-duplicate consecutive identical-class layers when ModuleList repeats the same block structure many times, collapse consecutive runs to keep the diagram readable.  Keeps first occurrence of each (class, parent-block) group.
def _parent(name: str) -> str: # (Anthropic, 2026)
    """Return the parent module path for a dot-separated name.

    Args:
        name: Dot-separated qualified module name.

    Returns:
        Everything before the final dot, or an empty string for top-level names.
    """
    parts = name.split(".")
    return ".".join(parts[:-1]) if len(parts) > 1 else ""

deduplicated: list[tuple[str, str, object]] = []
seen_class_under_parent: set[tuple[str, str]] = set()

for name, cls, shape in executed:
    parent = _parent(name)
    key    = (cls, parent)
    if key not in seen_class_under_parent:
        seen_class_under_parent.add(key)
        deduplicated.append((name, cls, shape))

print(f"  After dedup: {len(deduplicated)} nodes")

# Graphviz diagram
g = Digraph(f"{model_name}_layers", format="png")
g.attr(rankdir="LR", splines="ortho", nodesep="0.12", ranksep="0.25",
       size="36,6", dpi="150", bgcolor="white")
g.attr("node", margin="0", penwidth="0.6")

# Input node
g.node("__input__", label="", shape="box", style="filled",
       fillcolor=COLOR_MAP["InputLayer"],
       width="0.35", height="0.9", fixedsize="true",
       tooltip="Input")

# Track which color keys actually appear for legend filtering
used_color_keys: set[str] = {"InputLayer", "Output"}

prev = "__input__"
for i, (name, cls, shape) in enumerate(deduplicated):
    node_id = f"n{i}"
    color_key = get_color_key(cls)
    used_color_keys.add(color_key)

    # Node height scaled by last output dimension (channels/features)
    if shape and len(shape) >= 1:
        try:
            dim = int(shape[-1])
            height = max(0.3, min(3.0, dim / 64.0 * 1.2 + 0.25))
        except Exception:
            height = 0.6
    else:
        height = 0.6

    # Tooltip shows layer name + class + output shape for readability
    tooltip = f"{cls}\\n{name}\\n{shape}"

    g.node(node_id, label="", shape="box", style="filled",
           fillcolor=COLOR_MAP[color_key],
           width="0.25", height=str(round(height, 2)),
           fixedsize="true", tooltip=tooltip)
    g.edge(prev, node_id, arrowsize="0.4")
    prev = node_id

# Output node
g.node("__output__", label="", shape="box", style="filled",
       fillcolor=COLOR_MAP["Output"],
       width="0.2", height="0.35", fixedsize="true",
       tooltip="Output")
g.edge(prev, "__output__", arrowsize="0.4")

chart_base = charts_path / f"{model_name}_chart_modules_colored"
g.render(str(chart_base), cleanup=True)
print(f"  Graphviz diagram saved: {chart_base}.png")

# Pillow legend strip only show colours that appear in the diagram.
SQUARE_SIZE    = 36
TEXT_PADDING   = 10
ITEM_H_GAP     = 50
ITEM_V_GAP     = 14
LEGEND_PADDING = 24
LEGEND_BG      = (255, 255, 255, 255)
ITEMS_PER_ROW  = 5

try:
    font = ImageFont.truetype("arial.ttf", 22)
except Exception:
    font = ImageFont.load_default()

# Build legend items only keys that were actually encountered.
legend_items: list[tuple[str, str]] = []
for key in COLOR_MAP:
    if key == "default":
        continue
    if key in used_color_keys:
        legend_items.append((key, LEGEND_LABELS.get(key, key)))

def text_width(draw, text, fnt): # (Anthropic, 2026)
    """Return the pixel width of text rendered with fnt.

    Args:
        draw: A PIL.ImageDraw.ImageDraw instance used to measure text bounds.
        text: The string to measure.
        fnt: A PIL font object.

    Returns:
        Integer pixel width of the rendered text bounding box.
    """
    bbox = draw.textbbox((0, 0), text, font=fnt)
    return bbox[2] - bbox[0]

temp_img  = Image.new("RGBA", (1, 1))
temp_draw = ImageDraw.Draw(temp_img)

rows = [legend_items[i:i + ITEMS_PER_ROW]
        for i in range(0, len(legend_items), ITEMS_PER_ROW)]

item_pixel_widths = [
    SQUARE_SIZE + TEXT_PADDING + text_width(temp_draw, lbl, font)
    for _, lbl in legend_items
]

# Compute per-column max widths for alignment.
col_widths = []
for col in range(ITEMS_PER_ROW):
    col_ws = [
        item_pixel_widths[row_i * ITEMS_PER_ROW + col]
        for row_i in range(len(rows))
        if row_i * ITEMS_PER_ROW + col < len(item_pixel_widths)
    ]
    col_widths.append(max(col_ws) if col_ws else 0)

legend_content_w = sum(col_widths) + ITEM_H_GAP * max(0, ITEMS_PER_ROW - 1)
legend_content_h = len(rows) * (SQUARE_SIZE + ITEM_V_GAP) - ITEM_V_GAP

# Load diagram and combine
main_img = Image.open(str(chart_base.with_suffix(".png"))).convert("RGBA")

legend_strip_w = max(main_img.width,  legend_content_w + 2 * LEGEND_PADDING)
legend_strip_h = legend_content_h + 2 * LEGEND_PADDING

legend_strip = Image.new("RGBA", (legend_strip_w, legend_strip_h), LEGEND_BG)
draw         = ImageDraw.Draw(legend_strip)

for row_idx, row in enumerate(rows):
    x = LEGEND_PADDING
    y = LEGEND_PADDING + row_idx * (SQUARE_SIZE + ITEM_V_GAP)
    for col_idx, (key, label) in enumerate(row):
        hex_col = COLOR_MAP.get(key, COLOR_MAP["default"])
        r = int(hex_col[1:3], 16)
        gc = int(hex_col[3:5], 16)
        b  = int(hex_col[5:7], 16)
        draw.rectangle(
            [x, y, x + SQUARE_SIZE, y + SQUARE_SIZE],
            fill=(r, gc, b, 255), outline=(80, 80, 80, 255), width=1
        )
        draw.text(
            (x + SQUARE_SIZE + TEXT_PADDING, y + SQUARE_SIZE // 2),
            label, fill=(0, 0, 0, 255), font=font, anchor="lm"
        )
        x += col_widths[col_idx] + ITEM_H_GAP

total_w  = max(main_img.width, legend_strip_w)
total_h  = main_img.height + legend_strip_h
combined = Image.new("RGBA", (total_w, total_h), (255, 255, 255, 255))
combined.paste(main_img,     (0, 0))
combined.paste(legend_strip, (0, main_img.height))

output_path = charts_path / f"{model_name}_chart_modules_with_legend.png"
combined.save(str(output_path))
combined.show()
print(f"  Final diagram saved: {output_path}")