import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns

from bopep.scoring.is_peptide_in_binding_site import (
    is_peptide_near_binding_site_by_centroid,
    n_peptides_in_binding_site_colab_dir,
)

sns.set_context("paper")

binding_site_residue_indices = [22, 23, 24, 42, 43, 44, 45, 46, 47, 48, 49, 
                                50, 51, 52, 53, 69, 70, 71, 72,
                                 73, 74, 75, 76, 77, 81, 82, 83, 84, 85, 86, 87, 
                                 88, 89, 90, 104, 105, 106, 107, 108, 109, 110] 
binding_site_residue_indices = [residue - 19 for residue in binding_site_residue_indices]

# base directory containing one folder per complex
base_dir = "/srv/data1/er8813ha/docking-peptide/output_v2/run_cd14/docked_pdbs"

distances = []
n_contacts = []
labels = []
n_models = []

# loop over each complex directory
files = os.listdir(base_dir)
for i, entry in enumerate(files, start=1):
    print(f"{i}/{len(files)}: {entry}")
    colab_dir = os.path.join(base_dir, entry)
    if not os.path.isdir(colab_dir):
        continue

    # find the top‐ranked pdb file (here we grab rank_001 model)
    pdb_paths = glob.glob(os.path.join(colab_dir, "*rank_001*.pdb"))
    if not pdb_paths:
        print(f"  no rank_001 pdb in {entry}, skipping")
        continue
    pdb_file = pdb_paths[0]

    # compute centroid distance
    try:
        dist, _near = is_peptide_near_binding_site_by_centroid(
            pdb_file,
            binding_site_residue_indices,
            receptor_chain="A",
            peptide_chain="B",
            cutoff=10.0,
        )
    except Exception as e:
        print(f"  error computing centroid distance for {entry}: {e}")
        continue

    try:
        frac_contacts, _in_site, nr_contacts = n_peptides_in_binding_site_colab_dir(
            colab_dir,
            binding_site_residue_indices=binding_site_residue_indices,
            threshold=5.0,
            required_n_contact_residues=8,  # we only need the count
        )
    except Exception as e:
        print(f"  error computing fraction contacts for {entry}: {e}")
        continue

    distances.append(dist)
    n_contacts.append(nr_contacts)
    labels.append(entry)
    n_models.append(frac_contacts)

# plot
plt.figure(figsize=(4,4))
sns.jointplot(
    x=n_contacts,
    y=distances,
    kind="scatter",
    marginal_kws=dict(bins=30, fill=True),
    s=15
)
plt.xlabel("Number of contact residues")
plt.ylabel("Centroid-to-centroid distance (Å)")


plt.tight_layout()
plt.savefig("/home/er8813ha/bopep/examples/figures/centroid_vs_contacts.png", dpi=300)
plt.show()

plt.figure(figsize=(4,4))
sns.jointplot(x=n_models, y=distances, kind="scatter",
              marginal_kws=dict(bins=30, fill=True), s=15)

plt.ylabel("Centroid-to-centroid distance (Å)")
plt.xlabel("Fraction of contact models (n_required=8)")
plt.tight_layout( )
plt.savefig("/home/er8813ha/bopep/examples/figures/centroid_vs_fraction_contacts.png", dpi=300)
