import sys
from pathlib import Path
import pandas as pd
from pathlib import Path
from PIL import Image, ImageFont
import torch
from torchviz import make_dot

# Add the xLSTM_TS_Model directory to sys.path
xLSTM_path = Path(__file__).resolve().parent.parent / "xLSTM_TS_Model"
sys.path.insert(0, str(xLSTM_path))

# import your PyTorch model class
from xLSTM_TS import xLSTM_TS_Model

# base project directory
project_root = Path(__file__).resolve().parent.parent.parent  # Move up 3 levels from AI_Model_Diagrams

# CSV location (Dataset_Modules/dataset_output)
csv_path = project_root / "Dataset_Modules" / "dataset_output" / "2015-2025_dataset_denoised.csv"
charts_path = project_root / "AI_Model_Diagrams" / "AI_Model_Diagram_output"
df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

# model input settings
sequence_length = 10
n_features = df.shape[1]  # number of columns used as features
input_shape = (sequence_length, n_features)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# instantiate model (adjust embedding_dim/output_size if needed)
model = xLSTM_TS_Model(input_shape, embedding_dim=64, output_size=1).to(device)
model.eval()

# create dummy batch for visualization: [batch, seq_len, features]
dummy = torch.randn(1, sequence_length, n_features, device=device, requires_grad=True)

# Register hooks only for top-level child modules (depth==1) to get a concise, module-level graph
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
    if name == "":
        continue
    # capture only first-level children (e.g. initial_projection, mlstm_block1, slstm_block, ...)
    if name.count('.') == 0:
        hooks.append(module.register_forward_hook(save_output(name)))

# Forward once to record execution order and output shapes
out = model(dummy)

# Remove hooks
for h in hooks:
    h.remove()

# Build a compact graph of modules using graphviz (module-level, TF-like)
from graphviz import Digraph
g = Digraph("xLSTM_TS_modules", format="png")
g.attr(rankdir='LR', splines='ortho')

# simple color palette keyed by module class name fragments
COLOR_MAP = {
    "Linear": "#CFE2F3",
    "Conv": "#0B5394",
    "LayerNorm": "#D9EAD3",
    "LayerNormalization": "#D9EAD3",
    "LSTM": "#F4CCCC",
    "MultiheadAttention": "#B4A7D6",
    "MultiHeadAttention": "#B4A7D6",
    "Attention": "#B4A7D6",
    "ELU": "#FCE5CD",
    "Sequential": "#F9CB9C",
    "Module": "#E6B8AF",
    "default": "#D9EAD3",
    "Input": "#FFF2CC",
    "Output": "#FCE5CD",
}

def get_color(cls_name):
    for key, color in COLOR_MAP.items():
        if key.lower() in cls_name.lower():
            return color
    return COLOR_MAP["default"]

def parse_shape(shape):
    # shape may be a tuple or a string; we expect tensor shapes like (b, seq, feat)
    if isinstance(shape, tuple):
        try:
            # prefer last dim (features/hidden channels) for vertical size
            channels = int(shape[-1]) if len(shape) >= 1 else 1
            seq = int(shape[1]) if len(shape) >= 2 else 1
            return seq, channels
        except Exception:
            return 1, 1
    else:
        # unknown shape string -> fallback
        return 1, 1

# input node
g.node("input", f"Input\n[batch, seq_len, features]", shape="box", style="filled", fillcolor=COLOR_MAP["Input"])

prev = "input"
for name, cls, shape in executed:
    seq, channels = parse_shape(shape)
    # map channels -> visual height (inches), clamp to reasonable bounds
    height = max(0.4, min(3.0, channels / 64.0 * 0.8 + 0.4))
    # map sequence length -> visual width (inches)
    width = max(0.6, min(4.0, seq / 10.0 * 0.8 + 0.6))
    color = get_color(cls)
    label = f"{name}\n{cls}\n{shape}"
    # use fixedsize so graphviz respects width/height, and rounded filled boxes
    g.node(name, label, shape="rectangle", style="rounded,filled", fillcolor=color,
           width=str(width), height=str(height), fixedsize="true")
    g.edge(prev, name)
    prev = name

# final output
g.node("output", f"Output\n{tuple(out.shape)}", shape="box", style="filled", fillcolor=COLOR_MAP["Output"])
g.edge(prev, "output")

chart_base = charts_path / "xLSTM-TS_chart_modules_colored"
g.render(str(chart_base), cleanup=True)  # writes xLSTM-TS_chart_modules_colored.png

# display the image (Windows)
Image.open(str(chart_base.with_suffix(".png"))).show()