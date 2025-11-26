from fuzzywuzzy import fuzz
import os 
import json
from typing import Dict, Any


def read_dataset():
    train_data_path = [os.path.join("dataset/inputs", filename) for filename in os.listdir("dataset/inputs")]
    result_data_path = [os.path.join("dataset/results", filename) for filename in os.listdir("dataset/results")]
    train_data = []
    result_data = []
    for txt, result in zip(train_data_path, result_data_path):
        with open(txt, "r") as f:
            data = json.loads(f.read())
            train_data.append([obj.get("content") for obj in data["conversation"]])
        with open(result, "r") as f:
            data = json.loads(f.read())
            result_data.append(data)
    return train_data, result_data


def compare_contexts(groundtruth: Dict[str, Any], llm_result: Dict[str, Any], fuzzy_threshold=90):
    comparison = {}
    summary = {"total_fields": 0, "matched": 0, "missing_fields": [], "differences": []}

    for key in groundtruth.keys():
        gt_val = groundtruth[key]
        llm_val = llm_result.get(key, None)
        summary["total_fields"] += 1

        if isinstance(gt_val, dict):
            nested_comp, nested_summary = compare_contexts(gt_val, llm_val or {}, fuzzy_threshold)
            comparison[key] = nested_comp
            # merge nested summary
            summary["total_fields"] += nested_summary["total_fields"] - 1  # nested total already counted
            summary["matched"] += nested_summary["matched"]
            summary["missing_fields"].extend([f"{key}.{f}" for f in nested_summary["missing_fields"]])
            summary["differences"].extend([f"{key}.{f}" for f in nested_summary["differences"]])
        else:
            if llm_val is None:
                comparison[key] = False
                summary["missing_fields"].append(key)
                summary["differences"].append(key)
            else:
                gt_str = str(gt_val).strip().lower()
                llm_str = str(llm_val).strip().lower()
                # fuzzy match for long text
                if len(gt_str) > 50:
                    match_score = fuzz.ratio(gt_str, llm_str)
                    match = match_score >= fuzzy_threshold
                else:
                    match = gt_str == llm_str

                comparison[key] = match
                if match:
                    summary["matched"] += 1
                else:
                    summary["differences"].append(key)

    summary["accuracy"] = summary["matched"] / summary["total_fields"]
    return comparison, summary
