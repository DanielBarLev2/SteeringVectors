# visualize_logit_compare.py
import json
import html
from pathlib import Path

IN_JSON = "logit_compare.json"
TEMPLATE_PATH = "steering_viz_template.html"
OUT_HTML = "sleep.html"


def safe(s: str) -> str:
    return html.escape(s, quote=True)


def chip_color_width(val: float, vmax: float) -> tuple[str, str]:
    """(background_css, width_css) for score chip."""
    if vmax <= 1e-12:
        return "rgba(42, 157, 143, 0.10)", "0%"
    alpha = 0.18 + 0.35 * (val / vmax)  # opacity 0.18..0.53
    width = max(6, int(100 * (val / vmax)))  # min width so label fits
    return f"rgba(42, 157, 143, {alpha:.3f})", f"{width}%"


def build_grid_html(response_tokens, per_step_top):
    blocks = []
    for i, token_text in enumerate(response_tokens):
        top_items = per_step_top[i]
        vmax = max((it["delta"] for it in top_items), default=1e-12)

        chips_html = []
        for it in top_items:
            label = it["label"]
            val = it["delta"]
            bg, w = chip_color_width(val, vmax)
            chips_html.append(
                f"""
                <div class="chip">
                  <div class="bar" style="width:{w}; background:{bg};"></div>
                  <div class="chip-text">{safe(label)}<span class="chip-score">(+{val:.3f})</span></div>
                </div>
                """
            )
        chips = "\n".join(chips_html)
        blocks.append(
            f"""
            <div class="cell">
              <div class="token">{safe(token_text) or "&nbsp;"}</div>
              <div class="under">
                {chips}
              </div>
            </div>
            """
        )
    return "\n".join(blocks)


# build html
data = json.loads(Path(IN_JSON).read_text(encoding="utf-8"))

prompt = data["prompt"]
response_tokens = data["response_tokens"]
per_step_top = data["per_step_top"]

template = Path(TEMPLATE_PATH).read_text(encoding="utf-8")
grid_html = build_grid_html(response_tokens, per_step_top)

html_out = template.format(
    prompt=safe(prompt),
    response=safe("".join(response_tokens)),
    grid=grid_html,
    topk=len(per_step_top[0]) if per_step_top else 5
)

Path(OUT_HTML).write_text(html_out, encoding="utf-8")
print(f"[INFO] Wrote visualization → {Path(OUT_HTML).resolve()}")
