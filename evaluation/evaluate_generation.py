import torch

@torch.no_grad()
def molecule_evaluate(datamodule, molecule_list, evaluate_3D=True, evaluate_2D=False, evaluate_moses=False, evaluate_align=False):
    results = {}

    if evaluate_3D:
        print("Evaluating 3D EDM mertrics...")
        results_stability, results_rdkit, reconstructed_rdmols_3D = datamodule.evaluate_3D_edm(molecule_list)

        results_3D = {
            '3D/MolStable': results_stability['mol_stable'],
            '3D/AtomStable': results_stability['atom_stable'],
            '3D/Validity': results_rdkit['Validity'],
            '3D/Novelty': results_rdkit['Novelty'],
            '3D/Complete': results_rdkit['Complete'],
            '3D/Unique': results_rdkit['Unique'],
        }
        results.update(results_3D)

        if evaluate_moses:
            print("Evaluating 3D MOSES metrics...")
            evaluation_moses = datamodule.evaluate_moses(reconstructed_rdmols_3D)
            results_moses = {
                '3D/FCD': evaluation_moses['FCD']
            }
            results.update(results_moses)
    else:
        reconstructed_rdmols_3D = None

    if evaluate_2D:
        print("Evaluating 2D EDM metrics...")
        results_stability, results_rdkit, reconstructed_rdmols_2D = datamodule.evaluate_2D_edm(molecule_list)

        results_2D = {
            '2D/MolStable': results_stability['mol_stable'],
            '2D/AtomStable': results_stability['atom_stable'],
            '2D/Validity': results_rdkit['Validity'],
            '2D/Novelty': results_rdkit['Novelty'],
            '2D/Complete': results_rdkit['Complete'],
            '2D/Unique': results_rdkit['Unique'],
        }
        results.update(results_2D)

        if evaluate_moses:
            print("Evaluating 2D MOSES metrics...")
            evaluation_moses = datamodule.evaluate_moses(reconstructed_rdmols_2D)
            results_moses = {
                '2D/FCD': evaluation_moses['FCD'],
                '2D/SNN': evaluation_moses['SNN'],
                '2D/Frag': evaluation_moses['Frag'],
                '2D/Scaf': evaluation_moses['Scaf'],
                '2D/IntDiv': evaluation_moses['IntDiv'],
            }
            results.update(results_moses)
    else:
        reconstructed_rdmols_2D = None

    if evaluate_align:
        assert evaluate_2D and evaluate_3D, "The evaluation of 3D alignment requires the reconstruction RDKit molecules from both 2D evaluation."
        print("Evaluating 3D alignment of substructure geometries...")
        evaluation_align = datamodule.evaluate_sub_geometry(reconstructed_rdmols_2D)
        results_align = {
            '3D/Bond_Length_Mean': evaluation_align['bond_length_mean'],
            '3D/Bond_Angle_Mean': evaluation_align['bond_angle_mean'],
            '3D/Dihedral_Angle_Mean': evaluation_align['dihedral_angle_mean'],
        }
        results.update(results_align)

    return results, reconstructed_rdmols_3D, reconstructed_rdmols_2D

def add_evaluation_specific_args(parser):
    evaluation_group = parser.add_argument_group("Evaluation")
    evaluation_group.add_argument('--evaluate_3D', action='store_true', default=True)
    evaluation_group.add_argument('--not_evaluate_3D', action='store_false', dest='evaluate_3D')

    evaluation_group.add_argument('--evaluate_align', action='store_true', default=False)
    evaluation_group.add_argument('--not_evaluate_align', action='store_false', dest='evaluate_align')

    evaluation_group.add_argument('--evaluate_2D', action='store_true', default=False)
    evaluation_group.add_argument('--not_evaluate_2D', action='store_false', dest='evaluate_2D')

    evaluation_group.add_argument('--evaluate_moses', action='store_true', default=False)
    evaluation_group.add_argument('--not_evaluate_moses', action='store_false', dest='evaluate_moses')

    return evaluation_group