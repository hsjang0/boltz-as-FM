from rdkit.Chem import AllChem, rdmolops
from rdkit import Chem
from rdkit.Chem.rdchem import BondStereo, Conformer, Mol

def compute_3d_conformer(mol: Mol, version: str = "v3") -> bool:
    """Generate 3D coordinates using EKTDG method.

    Taken from `pdbeccdutils.core.component.Component`.

    Parameters
    ----------
    mol: Mol
        The RDKit molecule to process
    version: str, optional
        The ETKDG version, defaults ot v3

    Returns
    -------
    bool
        Whether computation was successful.

    """
    if version == "v3":
        options = AllChem.ETKDGv3()
    elif version == "v2":
        options = AllChem.ETKDGv2()
    else:
        options = AllChem.ETKDGv2()

    options.clearConfs = False
    conf_id = -1

    try:
        conf_id = AllChem.EmbedMolecule(mol, options)

        if conf_id == -1:
            print(
                f"WARNING: RDKit ETKDGv3 failed to generate a conformer for molecule "
                f"{Chem.MolToSmiles(AllChem.RemoveHs(mol))}, so the program will start with random coordinates. "
                f"Note that the performance of the model under this behaviour was not tested."
            )
            options.useRandomCoords = True
            conf_id = AllChem.EmbedMolecule(mol, options)

        AllChem.UFFOptimizeMolecule(mol, confId=conf_id, maxIters=1000)

    except RuntimeError:
        pass  # Force field issue here
    except ValueError:
        pass  # sanitization issue here

    if conf_id != -1:
        conformer = mol.GetConformer(conf_id)
        conformer.SetProp("name", "Computed")
        conformer.SetProp("coord_generation", f"ETKDG{version}")

        return True

    return False

from tdc.single_pred import ADME
data = ADMET(name = 'Caco2_Wang')

mol = Chem.MolFromSmiles("C[C@H]1CC@HCC1")
print(compute_3d_conformer(mol))