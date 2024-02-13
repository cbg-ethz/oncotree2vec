function getGeneDropDownId(div_id) {
  return div_id + "_drop-down-genes"
}

function getDrugDropDownId(div_id) {
  return div_id + "_drop-down-drugs"
}

function getInnerDivId(div_id) {
  return div_id + "_inner_div_gene"
}

function getInnerDivDrugId(div_id) {
  return div_id + "_inner_div_drug"
}

function addOneSample(sample_data) {

  sample_name = sample_data.get("sample_name")

  var body = document.getElementsByTagName("BODY")[0];
  var sample_div = document.createElement("div");
  sample_div.setAttribute("style", "border: 1px solid gray; border-radius: 5px; float:left; width:25%;  padding-top: 3px; padding-right: 3px; padding-left: 3px;")
  var div_id = sample_name + "_" + Date.now()
  sample_div.setAttribute("id", div_id)
  sample_div.innerHTML += '<div align="right"><a onclick="removeDiv(\'' + div_id + '\')"><i class="fa fa-times-circle fa-xs" ></i></a></div>'    
  body.appendChild(sample_div);                 

  var info_div_id = div_id + "_info"
  var tree_div_id = div_id + "_tree"  

  var div = document.createElement("div");
  div.setAttribute("id", info_div_id)    
  sample_div.appendChild(div);      
  div.innerHTML += ""
  
  var tree_div = document.createElement("div");
  tree_div.setAttribute("id", tree_div_id)  
  tree_div.innerHTML += "<strong style='background-color:#e6e2d3; display:block;'>&nbsp;scDNA Tree " + sample_name + ":<br></strong>"
  node_states = sample_data.get("node_states")
  for (let i = 0; i < node_states.length; i++) {
    node_state = objectToMap(node_states[i])
    gene_states = node_state.get("gene_state")
    amplified_genes = new Set()  
    deleted_genes = new Set()
    if(JSON.stringify(gene_states) != JSON.stringify({})) {
      objectToMap(gene_states).forEach((value, key) => {
        if(value > 0) {
	    amplified_genes.add(key)
        }
        else {
            deleted_genes.add(key)
        }
      })
    }   
    if(node_state.get("node_label") != '' && amplified_genes.size + deleted_genes.size != 0) {
      tree_div.innerHTML += "<font size=1><b>node " + node_state.get("node_label") + "</b>: <font color=tomato>amplified: " + [...Array.from(amplified_genes).sort()].join(', ') + "</font>, <font color=slateblue>deleted: " + [...Array.from(deleted_genes).sort()].join(', ') + "</font></font><br/>"   
    }
  } 
  tree_div.innerHTML += "<br/>"
  sample_div.appendChild(tree_div); 
  display_tree(tree_div_id, sample_data, "", "");   
}

function get_cell_count_for_gene(dict, target_gene) {
  if (target_gene in dict) {
    return dict[target_gene]
  }
  return 0
}

function get_cell_count_for_drug(node_states, target_drug, drug_gene_map) {
  var num_cells = 0
  if (target_drug in drug_gene_map) {
    var gene_list = drug_gene_map[target_drug]
    for (var node of node_states){ 
      // if a gene that interacts with the drug i in the node
      for (var gene of gene_list){
        if(gene in node.gene_state) {
          num_cells += node.num_cells
          break
        } 
      }
    }
  }
  return num_cells
}

function populateGeneInnerDiv(inner_div_id, tree_data, target_gene, gene_to_drug_map) {

  var inner_div = document.getElementById(inner_div_id);
  var html_gene_info = populateGeneInfoHTML(tree_data, target_gene)
  html_gene_info += populateGeneDrugInfoHTML(target_gene, gene_to_drug_map, inner_div_id) 
  html_gene_info +=  "<br/><br/>" 
  inner_div.innerHTML = html_gene_info
}

function populateDrugInnerDiv(inner_div_id, tree_data, target_drug, drug_gene_map, target_gene, gene_to_drug_map) {

  var num_cells_interactions = get_cell_count_for_drug(tree_data.node_states, target_drug, drug_gene_map)
  var inner_div = document.getElementById(inner_div_id);  
  html = "<br/>The selected drug has a theoretical interaction with " + num_cells_interactions + " cells (" + Math.round(100*num_cells_interactions / tree_data["total_num_cells"]) +"%)<br/>"
  inner_div.innerHTML = html
}

function populateGeneInfoHTML(tree_data, target_gene) {

  var num_amplified_cells = get_cell_count_for_gene(tree_data.cell_count_amplified_genes, target_gene)
  var num_deleted_cells =  get_cell_count_for_gene(tree_data.cell_count_deleted_genes, target_gene) 

  html_string = "Num cells with target gene amplified: " + num_amplified_cells + " (" + Math.round(100*num_amplified_cells / tree_data["total_num_cells"]) +"%)<br/>"
  html_string +=  "Num cells with target gene deleted: " + num_deleted_cells + " (" + Math.round(100*num_deleted_cells / tree_data["total_num_cells"] )+"%)<br/><br/>"

  return html_string
}

function populateGeneDrugInfoHTML(target_gene, gene_to_drug_map, inner_div_id) {
  var drugs = gene_to_drug_map[target_gene]

  html_string = ""
  if(drugs.length == 0) {
      html_string += "<b>No drugs associated with target gene.</b><br/><br/>"
  } else {
      drugs.sort(function(a, b){return b["drug_score"] - a["drug_score"]});

      var data_target_id = "drugs_" + inner_div_id
      html_string += "<b>Top DGIdb drugs associated with target gene:</b> <button class='btn' style='background-color:transparent' type='button' data-toggle='collapse' data-target=#" + data_target_id + "><i class='fa fa-angle-double-down' style='font-size:24px'></i></button>"
      html_string += "<div id=" + data_target_id + " class='collapse'>"
      html_string += "<font size=2><i>(<font color=tomato>inhibitors are red</font>, <font color=lightsteelblue>activators are blue</font>)</i></font><br/>"
      for(i=0; i<drugs.length; i++) {
          drug = drugs[i]
          if (drug.drug_score < 3) {
            continue
          }
          color = "black"
          if(isInhibitor(drug.interaction_types)) {
            color = "tomato"  
          }
          else if (isActivator(drug.interaction_types)) {
            color = "lightsteelblue"  
          }
          html_string += "-  <font color=" + color + ">" +drug.drug_name + "</font>: " + drug.interaction_types + " (" + drug.drug_score + " citations)<br/>"
      }
      html_string += "</div>"
  }
  return html_string
}

function populateFastdrugsInfoHTML(tree_data, div_id) {

  html_string = ""
  if("fastdrugs" in tree_data && tree_data["fastdrugs"].length != 0) {
      fastdrugs = tree_data["fastdrugs"]
      var data_target_id = "fastdrugs_" + div_id
      html_string += "<b>Significant fastDrugs results:</b> <button class='btn' style='background-color:transparent' type='button' data-toggle='collapse' data-target=#" + data_target_id + "><i class='fa fa-angle-double-down' style='font-size:24px'></i></button>"
      html_string += "<div id=" + data_target_id + " class='collapse'>"
      for(i=0; i<fastdrugs.length; i++) {
        html_string += fastdrugs[i].drug + "<br/>"
      }
      html_string += "</div>"
  }
  else {
    html_string += "<b>No fastDrugs results.</b><br/><br/>"
  }
  return html_string
}
