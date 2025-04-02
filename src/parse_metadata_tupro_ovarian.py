import pandas as pd
import sys


def parse_metadata_tupro_ovarian(samples):

    metadata = (
        pd.read_csv("../data/tupro_ovarian/metadata_tupro_ovarian.csv", sep=",")
        .fillna("")
        .applymap(str)
    )

    metadata_options = {}
    for key in metadata.keys():
        if key in [
            "sample_name",
            "Response_second-line_chemotherapy",
            "Age_range",
            "Histology",
            "ER_Patho_grouped",
            "F1_LOH_grouped",
            "blacklisted",
        ]:
            continue
        metadata_options[key] = set()

    sample_metadata_map = {}
    for sample in samples:
        met = metadata.loc[metadata["sample_name"] == sample]
        sample_metadata_map[sample] = {}
        for key in metadata_options.keys():
            met_value = met[key].item()
            if key == "Residual_disease" and met_value:
                met_value = int(float(met_value))
            sample_metadata_map[sample][key] = str(met_value)
            metadata_options[key].add(met_value)

    metadata_colors = {}
    metadata_colors["sampling_site"] = {
        "ascites": "#A6CEE3",
        "breast": "#1F78B4",
        "lymph_node": "#B2DF8A",
        "omentum": "#33A02C",
        "ovary": "#FB9A99",
        "peritoneum": "#E31A1C",
        "relapse": "#FDBF6F",
    }
    metadata_colors["Origin"] = {
        "ovary": "#1B9E77",
        "peritoneal": "#D95F02",
        "tubal": "#7570B3",
    }
    metadata_colors["primary_neo_relapse"] = {
        "chemo_naive": "#25C8EC",
        "neo-adjuvant": "#FFC71E",
        "relapse": "#A60661",
    }
    metadata_colors["Platinum_status"] = {
        "refractory": "#BEAED4",
        "resistant": "#FDC086",
        "sensitive": "#FFFF99",
        "no_chemotherapy": "#386CB0",
        "not_applicable": "#F2F2F2",
        "TBD": "#F2F2F2",
    }
    metadata_colors["Residual_disease"] = {
        "0": "#8DD3C7",
        "1": "#FFFFB3",
        "": "#F2F2F2",
    }
    metadata_colors["No_of_prior_chemotherapy_before_TuPro"] = {
        "0": "#EFF3FF",
        "1": "#C6DBEF",
        "2": "#9ECAE1",
        "3": "#6BAED6",
        "4": "#3182BD",
        "5": "#08519C",
    }
    metadata_colors["Response_first-line_chemotherapy"] = {
        "complete_response": "#00A600",
        "progressive_disease": "#E6E600",
        "stable_disease": "#EAB64E",
        "partial_response": "#FB6A4A",
        "unknown": "#F2F2F2",
        "TBD": "#F2F2F2",
    }
    metadata_colors["F1_BRCAness"] = {
        "wildtype": "#25C8EC",
        "BRCA1": "#FFC71E",
        "BRCA2": "#A60661",
        "-": "#F2F2F2",
    }
    metadata_colors["F1_LOH_PARPsens"] = {
        "01_resistance": "#FF0099",
        "02_response": "#00FF66",
        "-": "#F2F2F2",
    }
    metadata_colors["OncCassette_3q26"] = {"Yes": "#FF0099", "-": "#F2F2F2"}
    metadata_colors["FIGO_01"] = {
        "I": "#00A600",
        "II": "#E6E600",
        "III": "#FB6A4A",
        "IV": "#0054FF",
    }
    metadata_colors["has_wgd"] = {
        "yes": "#FC6238",
        "no": "#00CDAC",
        "uncertain": "#74737A",
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
