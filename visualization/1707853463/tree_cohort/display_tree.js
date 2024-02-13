////////// Help functions  ///////////

function get_cn_event_for_gene(node, target_gene) {
  if (target_gene in node.data.gene_cn_events) {
    return node.data.gene_cn_events[target_gene];
  }
  return 0;
}

function get_color_for_cn_event(cn_event) {
  if (cn_event > 0) {
    return "tomato";
  } else if (cn_event < 0) {
    return "lightsteelblue";
  }
  return "";
}

function get_color_for_gene_state(node, target_gene) {
  if (target_gene in node.data.gene_state) {
    gene_state = node.data.gene_state[target_gene];
    if (gene_state > 0) {
      return "tomato";
    } else if (gene_state < 0) {
      return "lightsteelblue";
    }
  }
  return "";
}

function removeElement(id){
  d3.selectAll(id).remove();
}

function getSvgId(div_id) {
  return div_id + "_svg"
}

html_colors = ["#FF0000", "#800000", "#E67E22", "#808000", "#00FF00", "#008000", "#00FFFF", "#008080", "#0000FF", "#000080", "#FF00FF", "#800080"]

////////// Display tree ///////////
function display_tree(tree_label, json_tree, highlight_node_ids=[], distance=1) {

  /*****
  var elem = document.getElementById(getSvgId(div_id));
  if(elem) {
    elem.parentNode.removeChild(elem);
  }
  *****/

  var color = d3.scaleLinear()
    .domain([0,1])
    .range(["red", "white"])
    //.domain([0, 0.5, 1])
    //.range(["red", "violet", "blue"]);

  // set the dimensions and margins of the diagram
  var margin = { top: 30, right: 30, bottom: 70, left: 50 };
  var width = 200 //screen.width / 6 - margin.left - margin.right;
  var height = 400 - margin.top - margin.bottom;

  // declares a tree layout and assigns the size
  var treemap = d3.tree().size([width, height]);
  //  assigns the data to a hierarchy using parent-child relationships
  var nodes = d3.hierarchy(json_tree);
  // maps the node data to the tree layout
  nodes = treemap(nodes);

  // append the svg obgect to the body of the page
  // appends a 'group' element to 'svg'
  // moves the 'group' element to the top left margin
  var svg = d3.select("body")
      //.select("div#" + div_id)
      .append("svg")
      .attr("width", width + margin.left + margin.right)
      .attr("height", height + margin.top + margin.bottom)
 
  var g = svg.append("g")
      .attr("transform",
            "translate(" + margin.left + "," + margin.top + ")");

  // adds the links between the nodes
  var link = g.selectAll(".link")
    .data( nodes.descendants().slice(1))
    .enter().append("path")
    .attr("class", "link")
    .attr("d", function(d) {
       return "M" + d.x + "," + d.y
         + "C" + d.x + "," + (d.y + d.parent.y) / 2
         + " " + d.parent.x + "," +  (d.y + d.parent.y) / 2
         + " " + d.parent.x + "," + d.parent.y;
       })
    .style("stroke", function(d){ 
        if(d.data.node_id == 0) {return 'white'} else {return 'lightgray'} 
    }) 

  // adds each node as a group
  var node = g.selectAll(".node")
    .data(nodes.descendants())
    .enter().append("g")
    .attr("class", function(d) { 
      return "node" + 
        (d.children ? " node--internal" : " node--leaf"); })
    .attr("transform", function(d) { 
      return "translate(" + d.x + "," + d.y + ")"; })
    .on("mouseover", function(d) {
      d3.select(this)
        .append("text")
        .classed("info", true)
        .attr("x", -10)
        .attr("y", 5)    
        .text(function(d) {
          return d.data.node_id
          if (d.data.node_id != 0) {    
            if (d.data.size_percent) {
              cell_fraction = Math.round(d.data.size_percent * 100)
            } 
            else {
              cell_fraction = Math.round(0.2 * 100)
            }
            return cell_fraction + "%";
          }
        })
    })
    .on("mouseout", function() {      
      // Remove the info text on mouse out.
      d3.select(this)
        .select("text.info")
        .remove();
    })

  g.append("text")
    .attr("id", "rectangleText")
    .attr("class", "visible")
    .attr("x", -35)
    .attr("y", 10)
    .attr("width",10)
    .style("font-size", "13px")
    .text(tree_label);
  

  // adds the circle to the node
  node
    .append("circle")
    .attr("r", function(d) {
      if (d.data.node_id == 0)
        return 5
      //if (d.data.node_id != 0) {
        //if (d.data.size_percentage) {
          //return Math.max(10, d.data.size_percent * 70)
          //return d.data.size_percent * 70
        //}
        //else {
          return Math.max(10, 0.2 * 70)
        //}
      //}
      return 1;
    })
    .style("fill", function(d){ 
        /*for (i = 0; i < highlight_node_ids.length; i++) {
          clone_info = highlight_node_ids[i]
          if (d.data.node_id == clone_info["clone_id"]) {
            return color(clone_info["distance"])
          }
        }
        if (!d.data.node_label) {
          return "gray";
        }
    
        if (d.data.node_id && d.data.num_cells != 0 && !d.data.is_malignant) {
          return "lightyellow"
        }*/

        if (d.data.node_id == 0) {
          return "white"
        }        

        if (d.data.node_color) {     
          return d.data.node_color
        }

    })
    /*.style("opacity", function(d){ 
        for (i = 0; i < highlight_node_ids.length; i++) {
          clone_info = highlight_node_ids[i]
          if (d.data.node_id == clone_info["clone_id"]) {
            return clone_info["opacity"]
          }
        }
    })*/

  node.append("text")
    .attr("x", -15)
    .attr("y", 4)
    .attr("dx", 12)
    .attr("dy", ".35em")
    .text(function(d) {
        if (d.data.node_id != 0) {
          return d.data.label //d.data.label //matching_label //label
        }
    })
}


