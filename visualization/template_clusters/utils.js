
const sleep = (milliseconds) => {
  return new Promise(resolve => setTimeout(resolve, milliseconds))
}

// Read sample_map
function loadSampleMap(filename){
  var script = document.createElement("script");
  script.src = filename;
  document.head.appendChild(script);
  sleep(2000).then(() => {})
}

const objectToMap = obj => {
   console.log(obj)
   const keys = Object.keys(obj);
   const map = new Map();
   for(let i = 0; i < keys.length; i++){
      //inserting new key value pair inside map
      map.set(keys[i], obj[keys[i]]);
   };
   return map;
};

function getSampleNames(sample_data) {
  return sample_data.map(s => s.sample_name)
}

function getCancerTypesSet(sample_data) {
  var cancer_types_set = new Set()
  for (key in sample_data) {
    cancer_types_set.add(sample_data[key].cancer_type)
  }
  return cancer_types_set
}

function getSamples(sample_data){
  array = []
  for (key in sample_data) {
    array.push(sample_data[key])
  }
  return array
}

function getSamplesByCancerType(sample_data, cancer_type){
  filtered_output = []
  for (key in sample_data) {
    var sample = sample_data[key]
    if(sample.cancer_type == cancer_type) {
      filtered_output.push(sample)
    }
  }
  return filtered_output
}

function getAllAffectedGenes(all_samples){

  var affected_genes = new Set()
  for (key in all_samples) {
    all_samples[key]["all_affected_genes"].forEach(affected_genes.add, affected_genes)
  }
  return Array.from(affected_genes)
}

function getMapKeys(map){
  list = []
  for (key in map) {
    list.push(key)
  }
  return list
}
