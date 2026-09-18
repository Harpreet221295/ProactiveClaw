import csv
import json

from .filesystem import _validate_agent_path


def generate_chart(code: str, filename: str) -> str:
    """Execute matplotlib code and save the resulting chart as an image."""
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import pandas as pd

    if not filename.endswith((".png", ".jpg", ".jpeg", ".svg", ".pdf")):
        filename += ".png"

    target = _validate_agent_path(f"charts/{filename}")
    target.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Fresh figure so previous calls don't leak state
        plt.figure()
        exec(code, {"plt": plt, "pd": pd, "json": json, "csv": csv})
        plt.savefig(str(target), bbox_inches="tight", dpi=150)
        plt.close("all")
        return f"Chart saved: charts/{filename}"
    except Exception as e:
        plt.close("all")
        return f"Error generating chart: {e}"


SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "generate_chart",
            "description": "Generate a chart/plot by executing matplotlib Python code. Write the plotting code using `plt` (matplotlib.pyplot) and `pd` (pandas) — they are pre-imported. Do NOT call plt.show() or plt.savefig() — saving is handled automatically. Read data from agent_file_system first using fs_read_csv/fs_read_json, then pass the data directly in the code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code using matplotlib (plt) and optionally pandas (pd). Do NOT call plt.show() or plt.savefig(). Example: 'dates = [\"Jan\", \"Feb\"]\nweights = [80, 79.5]\nplt.plot(dates, weights, marker=\"o\")\nplt.title(\"Weight Trend\")\nplt.ylabel(\"kg\")'",
                    },
                    "filename": {
                        "type": "string",
                        "description": "Output filename for the chart (e.g. 'weight_trend.png'). Saved to agent_file_system/charts/.",
                    },
                },
                "required": ["code", "filename"],
            },
        },
    },
]
