import pandas as pd
import sys


def parse_metadata_aml_mutation_trees(samples):

    metadata = (
        pd.read_csv(
            "../data/aml-mutation-trees/metadata_aml_mutation_trees.csv", sep=","
        )
        .fillna("")
        .map(str)
    )

    metadata_options = {}
    for key in metadata.keys():
        if key == "sample" or key == "N_subclone" or key == "Arm":
            continue
        metadata_options[key] = set()

    sample_metadata_map = {}
    for patient_name in samples:
        sample = patient_name.split("-")[0] + "-" + patient_name.split("-")[1]
        met = metadata.loc[metadata["sample"] == sample]
        sample_metadata_map[patient_name] = {}
        for key in metadata_options.keys():
            met_value = met[key].item()
            if key == "age":
                met_value = round(int(met_value), -1) - 10
            if key == "maxCCF":
                met_value = round(float(met_value), 1)
            sample_metadata_map[patient_name][key] = met_value
            metadata_options[key].add(met_value)

    metadata_colors = {}
    metadata_colors["PriorMalig"] = {"Yes": "#cc3b2e", "No": "#a0c0dd"}
    metadata_colors["Chemo"] = {"Yes": "#cc3b2e", "No": "#a0c0dd"}
    metadata_colors["XRT"] = {"Yes": "#cc3b2e", "No": "#a0c0dd"}
    metadata_colors["stAML"] = {"dnAML": "#cdbfcf", "stAML": "#e1d48e"}
    metadata_colors["untreated"] = {
        "treated": "#cdbfcf",
        "RR": "#e1d48e",
        "untreated": "#7b9a4a",
    }
    metadata_colors["Gender"] = {"Female": "#cc3b2e", "Male": "#a0c0dd"}
    metadata_colors["VitalStatus"] = {"Dead NOS": "#3d3b48", "Alive NOS": "#f9cb9c"}
    metadata_colors["age"] = {
        10: "#f5f5f0",
        20: "#e0e0d2",
        30: "#cbcbb3",
        40: "#b6b695",
        50: "#a2a277",
        60: "#88885d",
        70: "#6a6a49",
        80: "#4c4c34",
    }
    metadata_colors["Diagnosis"] = {
        "AEL": "#cdbfcf",
        "AML": "#e1d48e",
        "AMOL": "#d8b74b",
        "AMML": "#008080",
        "AUL": "#014421",
    }
    metadata_colors["Tx_group"] = {
        "others": "#cdbfcf",
        "IA-based": "#e1d48e",
        "DAC_VEN": "#d8b74b",
        "HMA": "#008080",
        "AraC-based": "#014421",
    }
    metadata_colors["Response"] = {
        "": "white",
        "CRi": "#e1d48e",
        "CR": "#d8b74b",
        "Died": "#014421",
        "TE": "#008080",
        "NE": "#e36745",
        "NR": "#9cc3fa",
        "HI": "#cdbfcf",
    }
    metadata_colors["maxCCF"] = {
        0.2: "#f9e0c1",
        0.3: "#f9cb9c",
        0.4: "#e3b089",
        0.5: "#ddba93",
        0.6: "#cfdcce",
        0.7: "#abc2aa",
        0.8: "#7a9778",
        0.9: "#516450",
        1: "#283228",
    }

    filtered_metadata_colors = {}
    for key in metadata_options:
        filtered_metadata_colors[key] = metadata_colors[key]

    all_color_labels = []
    for key in set(metadata_options.keys()):
        color_map = {sample: "white" for sample in samples}
        for sample in samples:
            color_map[sample] = metadata_colors[key][sample_metadata_map[sample][key]]
        color_series = pd.Series(color_map)
        color_series.name = key
        all_color_labels.append(color_series)

    return pd.concat(all_color_labels, axis=1), filtered_metadata_colors
