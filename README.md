# ContactWalker_OXA51
ContactWalker compares wild-type and mutant MD replicates, computes per-contact occupancy changes, ranks them against an empirical noise floor and renders the disruption as a publication-style network diagram. It was developed and validated on the OXA-51 β-lactamase, but generalizes to any protein system with WT and mutant MD replicates.
**Background**
Point mutations can reshape a protein's intramolecular contact network, stabilizing or destabilizing specific residue-residue contacts. ContactWalker provides a reproducible, statistics-aware workflow for detecting these changes from MD trajectory snapshots:
1.	Detect contacts in every PDB frame using distance cutoffs, and compute per-contact occupancy across replicates.
2.	Quantify the change as the minimum (by absolute value) of all WT–MUT pairwise occupancy differences — the most conservative signal estimate.
3.	Set an empirical noise floor from within-replicate variability, so that only changes exceeding replicate noise are called disruptions.
4.	Visualize the disruption network, linking disrupted clusters back to the mutation site through the protein's stable structural scaffold.
The pipeline uses a consistent sign convention throughout:
change = WT_occupancy − MUT_occupancy

  Negative → mutant GAINS contact   (MUT > WT, stabilized in mutant)
  Positive → mutant LOSES contact   (MUT < WT, destabilized in mutant)
________________________________________
**Install
Prerequisites**
•	Python 3.8+
•	Graphviz system binaries (required only for Step 4, the network figure)
**Python dependencies**
pip install MDAnalysis numpy pandas tqdm networkx matplotlib pydot
Or via conda:
conda install -c conda-forge mdanalysis numpy pandas tqdm networkx matplotlib pydot graphviz

**Usage**
**Quick start**
# Step 1: Detect contacts from MD snapshots → full data CSV
python contactwalker.py

# Step 2: Merge split residues → analysis tables (Table 1 + Table 2)

python contactwalker_split_tables.py <input_csv> [output_dir]

# Step 3: Compute noise floor → percentile ranking
python noise_ranking.py --input contactwalker_full_data.csv --output noise_ranking.csv

# Step 4: Render disruption network 
python contactwalker_network_graphviz.py <input_csv> --mutation ILE129 --mutation-label LEU129 --gv-engine dot --scale 1.0 --edge-length 1.4 --node-size 1000 --font-size 10 --fig-inch 8.27 11.69 --dpi 300
Pipeline data flow
 MD snapshots (PDB)          Full data CSV              Analysis tables
 (WT ×3, MUT ×3)    ──►     contactwalker_full_data.csv  ──►  Table 1 & Table 2
       │                              │                          │
       │                              ▼                          │
       │                     noise_ranking.csv                   │
       │                     (percentile cutoff)                 │
       │                              │                          │
       └──────────────────────────────┴──────────────────────────┘
                                      ▼
                          Disruption network (PNG)
