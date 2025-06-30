#!/usr/bin/env python3
"""Aggregate inference results and generate simplified HTML for documentation.

The script searches for ``results_summary.json`` files and converts them
into an HTML structure with a main table that spans full width and
collapsible experiment details using native HTML <details>/<summary> tags.
It includes the facial recognition algorithm in the details.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from html import escape
from datetime import datetime


def collect_summaries(root: Path) -> list[dict]:
    """Traverse ``root`` and read all summary JSON files."""
    pattern = root.rglob("results_summary.json")
    summaries: list[dict] = []
    for summary_file_path in pattern:
        print(f"Reading {summary_file_path}")
        try:
            with summary_file_path.open("r") as f:
                data = json.load(f)
            summaries.append(data)
        except json.JSONDecodeError:
            print(f"Warning: Skipping malformed JSON file: {summary_file_path}")
            continue
    return summaries


def format_cell_value(value, is_numeric_col=False):
    """Helper to format values: date, rounded numbers, or default."""
    if isinstance(value, str):
        if 'T' in value and any(c.isdigit() for c in value.split('T')[0]):
            try:
                dt_object = datetime.fromisoformat(value)
                return dt_object.strftime("%Y-%m-%d")
            except ValueError:
                pass
    
    if is_numeric_col:
        try:
            if isinstance(value, bool):
                return str(value)
            return f"{float(value):.4f}" # Format to 4 decimal places
        except (ValueError, TypeError):
            pass
    
    return escape(str(value))


def summaries_to_html_plain(summaries: list[dict]) -> str:
    """Convert a list of summary dictionaries into HTML with a full-width main table and collapsible details."""
    if not summaries:
        return "<p>No results to display.</p>"

    all_experiments_html = []

    # --- Configuration for Main Table columns ---
    main_table_columns_config = [
        {"key": "timestamp", "header": "Date"},
        {"key": "model_name", "header": "Model"},
        {"key": "training_dataset", "header": "Train DS"},
        {"key": "inference_dataset", "header": "Inf. DS"},
        {"key": "bpp", "header": "BPP"},
        {"key": "watermark_lenght", "header": "WM Len."},
        {"key": "accuracy", "header": "Acc."},
        {"key": "psnr", "header": "PSNR"},
        {"key": "ssim", "header": "SSIM"},
        # Distance metrics - now with 'facenet_' prefix in keys
        {"key": "average_distances.facenet_avg_dist_cosine_genuine_before_watermark", "header": "Dist. Gen. Before"},
        {"key": "average_distances.facenet_avg_dist_cosine_impostor_before_watermark", "header": "Dist. Imp. Before"},
        {"key": "average_distances.facenet_avg_dist_cosine_genuine_after_watermark", "header": "Dist. Gen. After"},
        {"key": "average_distances.facenet_avg_dist_cosine_impostor_after_watermark", "header": "Dist. Imp. After"},
    ]

    # --- Configuration for Details (collapsible) content ---
    # Added "facial_recognition_algorithm"
    details_content_config = {
        "experiment_name": "Experiment Name",
        "facial_recognition_algorithm": "Face Rec. Algorithm",
        "fine_tuned_icao": "Fine-Tuned ICAO",
        "OFIQ_score": "OFIQ Score",
        "ICAO_compliance": "ICAO Compliance",
    }
    distance_metric_type = "Cosine Distance"

    # Numeric keys to round (used by format_cell_value)
    numeric_keys_to_round = {
        "accuracy", "psnr", "ssim",
        "facenet_avg_dist_cosine_genuine_before_watermark", # Updated key names
        "facenet_avg_dist_cosine_impostor_before_watermark", # Updated key names
        "facenet_avg_dist_cosine_genuine_after_watermark", # Updated key names
        "facenet_avg_dist_cosine_impostor_after_watermark", # Updated key names
    }

    for idx, row_data in enumerate(summaries):
        # Generate an ID/Title for each execution
        model_name = row_data.get("model_name", "Unknown Model")
        experiment_name = row_data.get("experiment_name", f"Experiment {idx + 1}")
        execution_id_title = f"{model_name} - {experiment_name}"

        all_experiments_html.append(f"<hr><h3>Experiment: {str(idx + 1)}</h3>")

        # --- Generate Main Metrics Table (full width) ---
        main_header_cells = "".join(f"<th>{escape(col['header'])}</th>" for col in main_table_columns_config)
        main_header_row = f"<tr>{main_header_cells}</tr>"
        
        main_cells_data = []
        for col_config in main_table_columns_config:
            key = col_config["key"]
            value = 'N/A' 
            if '.' in key: # Handles nested keys like "average_distances.facenet_avg_dist_..."
                main_key, sub_key = key.split('.', 1)
                nested_dict = row_data.get(main_key, {})
                value = nested_dict.get(sub_key, 'N/A')
            else: # Handles top-level keys
                value = row_data.get(key, 'N/A')
            
            # Use the full key from config to check if it's numeric for rounding
            is_numeric_for_rounding = key.split('.')[-1] in numeric_keys_to_round 
            main_cells_data.append(f"<td>{format_cell_value(value, is_numeric_col=is_numeric_for_rounding)}</td>")
        
        all_experiments_html.append("<h4>Main Metrics</h4>")
        all_experiments_html.append("<table class='table'") 
        all_experiments_html.append(f"<thead>{main_header_row}</thead>")
        all_experiments_html.append(f"<tbody><tr>{''.join(main_cells_data)}</tr></tbody>")
        all_experiments_html.append("</table>")

        # --- Generate Collapsible Experiment Details ---
        details_html_content = []
        
        # Determine Facial Recognition Algorithm from the distance keys
        # We assume if any facenet key exists, the algorithm is Facenet
        facial_rec_algo = "N/A"
        if row_data.get("average_distances"):
            for dist_key in row_data["average_distances"].keys():
                if dist_key.startswith("facenet_"):
                    facial_rec_algo = "Facenet"
                    break
                elif dist_key.startswith("arcface"):
                    facial_rec_algo = "ArcFace"
                    break
                else:
                    facial_rec_algo = "Unknown" 

        details_html_content.append(f"<p><strong>Distance Metric:</strong> {escape(distance_metric_type)}</p>")
        details_html_content.append(f"<p><strong>Face Rec. Algorithm:</strong> {escape(facial_rec_algo)}</p>") # NEW LINE
        
        for key, display_name in details_content_config.items():
            if key == "facial_recognition_algorithm": # Skip, handled above
                continue
            
            value = row_data.get(key, 'N/A')
            is_numeric_for_rounding = key in numeric_keys_to_round
            details_html_content.append(f"<p><strong>{escape(display_name)}:</strong> {format_cell_value(value, is_numeric_col=is_numeric_for_rounding)}</p>")
        
        # Using <details> and <summary> for native collapsible behavior
        all_experiments_html.append("<h4>Experiment Details</h4>")
        all_experiments_html.append("<details>")
        all_experiments_html.append("<summary>Click to show/hide details</summary>")
        all_experiments_html.append(f"<div>{''.join(details_html_content)}</div>")
        all_experiments_html.append("</details>")
        all_experiments_html.append("<br>")

    return "".join(all_experiments_html)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect inference summaries for documentation"
    )
    parser.add_argument(
        "root",
        nargs="?",
        default="experiments/output/watermarking",
        help="Root directory with watermarking outputs",
    )
    parser.add_argument(
        "--output",
        default="docs/_data",
        help="Destination directory for generated files",
    )
    args = parser.parse_args()

    root = Path(args.root)
    dest = Path(args.output)

    summaries = collect_summaries(root)
    dest.mkdir(parents=True, exist_ok=True)

    # Save summaries as JSON (includes all original data)
    (dest / "inference_runs.json").write_text(
        json.dumps(summaries, indent=2)
    )

    # Convert summaries to HTML with full-width main table and collapsible details
    html_output = summaries_to_html_plain(summaries)
    # Output name for clarity
    (dest / "inference_results.html").write_text(html_output)

    print(f"Wrote {len(summaries)} summaries as plain HTML to {dest / 'inference_results.html'}")


if __name__ == "__main__":
    main()