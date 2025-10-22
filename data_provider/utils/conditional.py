import torch
from torch_geometric.data import Data

class ConditionTransform(object):
    def __init__(self, atom_type_list, *property_idx):
        super().__init__()
        self.atom_type_list = torch.tensor(list(atom_type_list))

        if len(property_idx) > 2:
            raise NotImplementedError(f"Only 2 properties are supported, but got {len(property_idx)}")

        self.property_idx_list = property_idx

    def __call__(self, data: Data):
        properties = data.y
        property_data = []
        for property_idx in self.property_idx_list:
            if property_idx == 11:
                Cv_atomref = [2.981, 2.981, 2.981, 2.981, 2.981]
                atom_types = data.atom_type
                atom_counts = torch.bincount(atom_types, minlength=len(Cv_atomref))
                property_data.append(properties[0, property_idx:property_idx+1] - torch.sum((atom_counts * torch.tensor(Cv_atomref))))
            else:
                property = properties[0, property_idx:property_idx+1]
                property_data.append(property)
        data.property = torch.cat(property_data).unsqueeze(0)

        return data
