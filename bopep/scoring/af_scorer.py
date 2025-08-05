import os
import json
import re
from Bio.PDB import PDBParser, NeighborSearch, Selection
from collections import defaultdict

# TODO UPDATE THIS!

class AFScorer:
    """
    Class for scoring AlphaFold/ColabFold predictions.
    """
    
    def __init__(self, colab_dir, rank_num=1):
        """
        Initialize the AFScorer with a ColabFold output directory.
        """
        self.colab_dir = colab_dir
        self.rank_num = rank_num
        self.initialized = False
        self.data = None
        self.json_file = None
        self.pdb_file = None
        self.residue_chain_list = None
        
    def _initialize(self):
        """
        Parse the JSON file and PDB file from ColabFold output once and store the data.
        """
        if not self.initialized:
            # Find the JSON file
            json_pattern = re.compile(rf".*_scores_rank_00{self.rank_num}_.*\.json")
            json_files = []
            for root, _, files in os.walk(self.colab_dir):
                for f in files:
                    if json_pattern.search(f):
                        json_files.append(os.path.join(root, f))

            if not json_files:
                raise FileNotFoundError(f"No matching JSON file found in {self.colab_dir}")
            
            self.json_file = json_files[0]
            
            # Find the corresponding PDB file
            pdb_pattern = re.compile(rf".*_relaxed_rank_00{self.rank_num}_.*\.pdb")
            pdb_files = []
            for root, _, files in os.walk(self.colab_dir):
                for f in files:
                    if pdb_pattern.search(f):
                        pdb_files.append(os.path.join(root, f))

            if not pdb_files:
                raise FileNotFoundError(f"No matching PDB file found in {self.colab_dir}")
                
            self.pdb_file = pdb_files[0]
            
            # Parse the JSON file
            with open(self.json_file, "r") as f:
                self.data = json.load(f)
            
            # Parse the PDB file to build residue chain list
            self._parse_pdb_residues()
                
            self.initialized = True
    
    def _parse_pdb_residues(self):
        """
        Parse the PDB file to build a list of residue chains in order.
        Used for mapping pLDDT and PAE values to the correct residues.
        """
        self.residue_chain_list = []
        
        try:
            with open(self.pdb_file, "r") as f:
                last_chain_resid = None
                for line in f:
                    if line.startswith("ATOM"):
                        chain_id = line[21]       # chain ID is at position 21
                        residue_num = line[22:26] # residue number is at positions 22-25
                        residue_num = residue_num.strip()

                        # Combine chain + resid to check if this is a new residue
                        chain_resid = (chain_id, residue_num)
                        if chain_resid != last_chain_resid:
                            self.residue_chain_list.append(chain_resid)
                            last_chain_resid = chain_resid
        except IOError as e:
            print(f"Error reading PDB file: {e}")
            raise
    
    def get_iptm(self):
        if not self.initialized:
            self._initialize()
        
        return self.data.get("iptm")
    
    def get_peptide_plddt(self, chain_id="B"):
        if not self.initialized:
            self._initialize()
        
        # Get pLDDT array from data
        plddt_array = self.data.get("plddt")
        if not plddt_array:
            print("JSON does not contain a 'plddt' array.")
            return None
            
        # Extract indices for the specified chain
        chain_indices = [
            i for i, (chain, _) in enumerate(self.residue_chain_list) 
            if chain.upper() == chain_id.upper()
        ]
        
        if not chain_indices:
            print(f"No residues found for chain {chain_id} in the PDB file.")
            return None
            
        # Calculate mean pLDDT for the chain
        chain_plddt_vals = [plddt_array[i] for i in chain_indices]
        return sum(chain_plddt_vals) / len(chain_plddt_vals)
    
    def get_weighted_plddt(self, chain_id="B", protein_chain="A", distance_threshold=5.0):
        """
        Calculate weighted pLDDT for peptide chain based on contact residues.
        Uses B-factors (pLDDT values) from the PDB file and weights them by 
        contact with the protein chain.
        
        Args:
            chain_id: Chain ID of the peptide (default: "B")
            protein_chain: Chain ID of the protein (default: "A") 
            distance_threshold: Distance threshold for defining contacts (default: 5.0 Å)
            
        Returns:
            tuple: (overall_weighted_avg, residue_weighted_avg)
        """
        if not self.initialized:
            self._initialize()
            
        if not self.pdb_file or not os.path.exists(self.pdb_file):
            print(f"PDB file not found: {self.pdb_file}")
            return None, None
            
        try:
            # Initialize the PDB parser
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure('structure', self.pdb_file)
            
            # Extract chains A and B
            chain_a = None
            chain_b = None
            for model in structure:
                try:
                    chain_a = model[protein_chain]
                    chain_b = model[chain_id]
                    break  # Assuming only one model is needed
                except KeyError as e:
                    print(f"Chain not found: {e}")
                    return None, None
            
            if not chain_a or not chain_b:
                print(f"Could not find chains {protein_chain} and/or {chain_id}")
                return None, None
            
            # Identify contact residues in chain B (within distance threshold to any residue in chain A)
            atoms_in_chain_a = Selection.unfold_entities(chain_a, 'A')  # List of all atoms in chain A
            atoms_in_chain_b = Selection.unfold_entities(chain_b, 'A')  # List of all atoms in chain B

            # Using NeighborSearch to find contacts
            ns = NeighborSearch(atoms_in_chain_a)
            contact_residues_in_chain_b = set()
            
            for atom in atoms_in_chain_b:
                neighbors = ns.search(atom.coord, distance_threshold, 'R')  # 'R' for residues
                if neighbors:
                    contact_residues_in_chain_b.add(atom.get_parent().get_id())

            # To store B-factors for each residue in chain B (including zeros for non-contact residues)
            residue_bfactors = defaultdict(list)

            # Loop over all residues in chain B
            for residue in chain_b:
                residue_id = residue.get_id()
                if residue_id in contact_residues_in_chain_b:
                    # Get B-factors for all atoms in the residue
                    for atom in residue:
                        residue_bfactors[residue_id].append(atom.get_bfactor())
                else:
                    # If not in contact, add a zero for the residue
                    residue_bfactors[residue_id].append(0)

            # Calculate average B-factor per residue (averaging over all atoms within a residue)
            avg_bfactors_per_residue = {res_id: sum(bfactors)/len(bfactors) 
                                        for res_id, bfactors in residue_bfactors.items()}
            
            # Calculate the overall average (Direct average of all atoms in contact and non-contact residues)
            all_bfactors = [bfactor for bfactors in residue_bfactors.values() for bfactor in bfactors]
            overall_avg_bfactor = sum(all_bfactors) / len(all_bfactors) if all_bfactors else 0
            
            # Calculate the average of residue averages
            avg_bfactor_from_residues = sum(avg_bfactors_per_residue.values()) / len(avg_bfactors_per_residue) if avg_bfactors_per_residue else 0
            
            return overall_avg_bfactor, avg_bfactor_from_residues
            
        except Exception as e:
            print(f"Error calculating weighted pLDDT: {e}")
            return None, None
    
    def get_peptide_pae(self, chain_id="B"):
        if not self.initialized:
            self._initialize()
        
        # Get PAE array from data (this is a 2D matrix)
        pae_array = self.data.get("pae")
        if not pae_array:
            print("JSON does not contain a 'pae' array.")
            return None
            
        # Extract indices for the specified chain
        chain_indices = [
            i for i, (chain, _) in enumerate(self.residue_chain_list) 
            if chain.upper() == chain_id.upper()
        ]
        
        if not chain_indices:
            print(f"No residues found for chain {chain_id} in the PDB file.")
            return None
        
        # For PAE we need to consider the full 2D matrix 
        # We want the PAE between chain B and all other residues
        all_pae_values = []
        
        # Extract all PAE values involving the chain B residues
        for i in chain_indices:
            # Get PAE values where this residue interacts with all others
            for j in range(len(self.residue_chain_list)):
                all_pae_values.append(pae_array[i][j])
        
        # Calculate the mean PAE
        if all_pae_values:
            return sum(all_pae_values) / len(all_pae_values)
        else:
            return None
    
    def get_all_metrics(self):
        if not self.initialized:
            self._initialize()
            
        metrics = {}
        
        # Add available metrics to the dictionary
        iptm = self.get_iptm()
        if iptm is not None:
            metrics['iptm'] = iptm
            
        peptide_plddt = self.get_peptide_plddt()
        if peptide_plddt is not None:
            metrics['peptide_plddt'] = peptide_plddt
            
        peptide_pae = self.get_peptide_pae()
        if peptide_pae is not None:
            metrics['peptide_pae'] = peptide_pae
            
        # Add weighted pLDDT metrics
        weighted_plddt_overall, weighted_plddt_residues = self.get_weighted_plddt()
        if weighted_plddt_overall is not None:
            metrics['weighted_plddt_overall'] = weighted_plddt_overall
        if weighted_plddt_residues is not None:
            metrics['weighted_plddt_residues'] = weighted_plddt_residues
            
        return metrics


if __name__ == "__main__":
    # Example usage
    colab_dir_path = "/srv/data1/general/immunopeptides_data/databases/benchmark_data/pdbs_erik/docked_peptides/1ydi_VGWEQLLTTIARTINEVENQILTR"
    
    scorer = AFScorer(colab_dir_path)
    #print(f"All metrics: {scorer.get_all_metrics()}")
    print(f"ipTM: {scorer.get_iptm()}")
    print(f"Peptide pLDDT: {scorer.get_peptide_plddt()}")
    print(f"Peptide PAE: {scorer.get_peptide_pae()}")
    
    # Test weighted pLDDT
    weighted_overall, weighted_residues = scorer.get_weighted_plddt()
    print(f"Weighted pLDDT (overall): {weighted_overall}")
    print(f"Weighted pLDDT (residues): {weighted_residues}")
