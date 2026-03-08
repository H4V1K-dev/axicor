# Data Credits — GNM-Library

The **GNM-Library** neuron type parameters were derived from the following publicly available neuroscience databases. Raw biological measurements were transformed into **Genesis Neuron Model (GNM)** integer-domain invariants via high-fidelity approximation formulas; no original data files are distributed with this library.

---

### 🧠 Allen Cell Types Database
**Provider:** Allen Institute for Brain Science  
**URL:** [celltypes.brain-map.org](https://celltypes.brain-map.org)  
**Data used:** Electrophysiology (ephys) recordings and morphological reconstructions (SWC) of neocortical neurons from *Homo sapiens* and *Mus musculus*.  
**License:** [Allen Institute Terms of Use](https://alleninstitute.org/terms-of-use)  
*Commercial use of derivative works is permitted with attribution.*

> **Citation:**
> Allen Institute for Brain Science (2015). Allen Cell Types Database. Available from: https://celltypes.brain-map.org

---

### 🧬 NeuroMorpho.Org
**Provider:** George Mason University  
**URL:** [neuromorpho.org](https://neuromorpho.org)  
**Data used:** Morphological reconstructions (SWC format) for subcortical structures including hippocampus, thalamus, cerebellum, and striatum. Used to approximate axon length, dendritic radius, and branching parameters.  
**License:** Open access. Individual reconstructions were contributed by their respective originating laboratories and are subject to per-contributor attribution requirements.

> **Attribution:** Morphological reconstructions sourced from NeuroMorpho.Org. Original laboratory contributors are acknowledged per [neuromorpho.org/p-acknowledgments.jsp](https://neuromorpho.org/p-acknowledgments.jsp)

> **Citations:**
> 1. Ascoli, G. A., Donohue, D. E., & **Bhatt, D. L.** (2007). NeuroMorpho.Org: A Central Resource for Neuronal Morphologies. *Journal of Neuroscience*, 27(35), 9247–9251. DOI: [10.1523/JNEUROSCI.2055-07.2007](https://doi.org/10.1523/JNEUROSCI.2055-07.2007)
> 2. Ascoli, G. A., **Halavi, M.**, et al. (2012). NeuroMorpho.Org: A Decade of Neuronal Morphologies. *Nature Methods*, 9, 778. DOI: [10.1038/nmeth.2114](https://doi.org/10.1038/nmeth.2114)

---

### 🔬 Note on Heuristic Constants
Biophysical reference values used in approximation formulas (resting potentials, time constants, firing thresholds for specific cell classes such as Purkinje cells and Medium Spiny Neurons) are derived from general neuroscience literature. Factual biophysical constants published in scientific papers are generally not subject to copyright protection.

---

### 🛠️ How Source Data Was Used
Source data was used exclusively during the library generation process (offline, locally). The generation scripts consumed raw JSON/CSV exports from the databases listed above and produced the `.toml` parameter files in this library. 

**No raw database files, SWC reconstructions, or original recordings are included in this repository or any distribution of Genesis Engine.**

---
*For more information, visit the [Genesis Project Repository](https://github.com/H4V1K-dev/genesis-agi).*