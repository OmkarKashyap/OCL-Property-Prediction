import json
import csv

PENULTIMATE_DINOSAUR_JSON_FILE = r'/mnt/data2/radhika/OCL-Property-Prediction/checkpoints/penultimate_layer_numbers/dinosaur/oc_eval/evaluation/downstream_prediction/loss/linear/train_original/test_original/results.json'
PENULTIMATE_FT_DINOSAUR_JSON_FILE = r'/mnt/data2/radhika/OCL-Property-Prediction/checkpoints/penultimate_layer_numbers/ft_dinosaur/oc_eval/evaluation/downstream_prediction/loss/linear/train_original/test_original/results.json'

FINAL_LAYER_DINOSAUR_JSON_FILE = r'/mnt/data2/radhika/OCL-Property-Prediction/checkpoints/final_layer_numbers/dinosaur/oc_eval/evaluation/downstream_prediction/loss/linear/train_original/test_original/results.json'
FINAL_LAYER_FT_DINOSAUR_JSON_FILE = r'/mnt/data2/radhika/OCL-Property-Prediction/checkpoints/final_layer_numbers/ft_dinosaur/oc_eval/evaluation/downstream_prediction/loss/linear/train_original/test_original/results.json'

CONSOLIDATED_RESULTS_CSV = r'/mnt/data2/radhika/OCL-Property-Prediction/consolidated_results.csv'



with open(PENULTIMATE_DINOSAUR_JSON_FILE, "r", encoding="utf-8") as f:
    penulimate_dinosaur_data = json.load(f)

with open(PENULTIMATE_FT_DINOSAUR_JSON_FILE, "r", encoding="utf-8") as f:
    penulimate_ft_dinosaur_data = json.load(f)

with open(FINAL_LAYER_DINOSAUR_JSON_FILE, "r", encoding="utf-8") as f:
    final_layer_dinosaur_data = json.load(f)

with open(FINAL_LAYER_FT_DINOSAUR_JSON_FILE, "r", encoding="utf-8") as f:
    final_layer_ft_dinosaur_data = json.load(f)


# print(type(penulimate_dinosaur_data[0]))

with open(CONSOLIDATED_RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    writer.writerow(["model", "layer", "top1_acc", "top5_acc", "coords_x", "coords_y"])

    writer.writerow([
        "dinosaur",
        "penultimate_layer",
        penulimate_dinosaur_data[0]["results"]["metric_value"],
        penulimate_dinosaur_data[1]["results"]["metric_value"],
        penulimate_dinosaur_data[6]["results"]["metric_value"],
        penulimate_dinosaur_data[7]["results"]["metric_value"],
    ])

    writer.writerow([
        "ft_dinosaur",
        "penultimate_layer",
        penulimate_ft_dinosaur_data[0]["results"]["metric_value"],
        penulimate_ft_dinosaur_data[1]["results"]["metric_value"],
        penulimate_ft_dinosaur_data[6]["results"]["metric_value"],
        penulimate_ft_dinosaur_data[7]["results"]["metric_value"],
    ])

    writer.writerow([
        "dinosaur",
        "final_layer",
        final_layer_dinosaur_data[0]["results"]["metric_value"],
        final_layer_dinosaur_data[1]["results"]["metric_value"],
        final_layer_dinosaur_data[6]["results"]["metric_value"],
        final_layer_dinosaur_data[7]["results"]["metric_value"],
    ])

    writer.writerow([
        "ft_dinosaur",
        "final_layer",
        final_layer_ft_dinosaur_data[0]["results"]["metric_value"],
        final_layer_ft_dinosaur_data[1]["results"]["metric_value"],
        final_layer_ft_dinosaur_data[6]["results"]["metric_value"],
        final_layer_ft_dinosaur_data[7]["results"]["metric_value"],
    ])