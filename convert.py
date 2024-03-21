"""
A small collection of functions which parse data from lammps dump file
into a PyTorch Geometric `Data` object and back.

Compared to the previous data parcing pipeline, introduces two extra attribiutes
into the PyTroch Data object, namely `box` and `atom_ids`, which are needed for
the lammps trajectory file format.

Intended as a substite for the old helpers.py script.
"""
import os

import numpy as np
import torch
from torch_geometric.data import Data

from network import Atom, Bond, Box, Header, Network

def assemble_data(
    atoms: list[Atom],
    bonds: list[Bond],
    box: Box,
    node_features: str  = "coord",
    original_file_path: str | None = None,
    step: int | None = None
) -> Data:
    """A helper function which assembles the PyTorch_Geometric `Data` object.
    
    Parameters
    ----------
    `atoms` : list
        list of `Atom` objects to transform into nodes
    `bonds` : list
        list of `Bond` objects to transform into edges
    `box` : Box
        Box object to be used later for the purposes of calculating edges
    `node_features` : str, optional
        which node features to include, by default "coord"
    `original_file_path` : str | None, optional
        path to the original lammps simulation directory, by default None
    `step` : int | None, optional
        timestep of the original trajectory from which this Data was created,
        by default None


    Returns
    -------
    Data
        PyTorch_Geometric `Data` object
    """
    # mapping of atomic IDs to their list indices
    atom_ids = torch.tensor([atom.atom_id for atom in atoms])
    id_to_index_map = {atom.atom_id: i for atom, i in zip(atoms, range(len(atoms)))}

    # edges as defined with lammps IDs
    edges_with_ids = [(bond.atom1.atom_id, bond.atom2.atom_id) for bond in bonds]

    # edges as defined with indices
    edges_with_indices = [
        (id_to_index_map[node1], id_to_index_map[node2])
        for node1, node2 in edges_with_ids
    ]

    edge_index = torch.tensor(np.array(edges_with_indices).T)

    #NOTE: edge vectors will always be defined as vectors between the nodes in the SAME simulation box.
    # They will be wrond for bonds that cross the simulation box boundary.
    # On the other hand, edge lengths will always be correct.
    edge_vectors = [torch.tensor([bond.atom1.x - bond.atom2.x, bond.atom1.y - bond.atom2.y]) for bond in bonds] 
    edge_vectors = torch.stack(edge_vectors)
    edge_lengths = torch.tensor([bond.length for bond in bonds])
    edge_stiffnesses = torch.tensor([bond.bond_coefficient for bond in bonds])

    # edge vector and its length
    edge_attr = [
        torch.cat((v, torch.tensor([length]), torch.tensor([stiffness]))) 
        for v, length, stiffness in zip(edge_vectors, edge_lengths, edge_stiffnesses)
    ]
    edge_attr = torch.stack(edge_attr)

    match node_features:
        case "dummy":
            x = torch.tensor([[0] for i in range(len(atoms))])
        case "vel":
            try:
                x = torch.tensor([[atom.vx, atom.vy] for atom in atoms])
            except AttributeError:
                raise Exception("Atoms don't have velocity info")
        case "coord":
            x = torch.tensor([[atom.x, atom.y] for atom in atoms])
        case "full":
            x = torch.tensor(
                [[atom.x, atom.y, atom.vx, atom.vy] for atom in atoms]
            )
        case _:
            raise Exception(f"{node_features} not recognized as feature!")

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        box=box,
        atom_ids=atom_ids,
        file=original_file_path,
        step=step
    )


def parse_dump(
    dump_filepath: str,
    original_network: Network,
    node_features: str = "coord",
    skip: int = 1,
) -> list[Data]:
    """Parses a lammps dump file. By default returns x and y as node features.

    Parameters
    ----------
    `dump_filepath` : str
        Path to the lammps trajectory files
    `original_network` : Network
        simulated network to get the accurate information about periodic bonds
    `node_features` : str, optional
        node features to include into Data object, by default "coord".
        See `assemble_data()` function for more info.
    `skip` : int, optional
        load each n-th step starting from the first, by default 1 (skip none)

    Returns
    -------
    list[Data]
        list of pytorch_geometric Data objects
    """
    with open(dump_filepath, "r", encoding="utf8") as f:
        content = f.readlines()

    timesteps: list[int] = []
    for index, line in enumerate(content):
        if "ITEM: TIMESTEP" in line:
            timesteps.append(index)

    original_network.bonds
    original_edge_index = [(bond.atom1.atom_id, bond.atom2.atom_id) for bond in original_network.bonds]
    bond_map = {(bond.atom1.atom_id, bond.atom2.atom_id): bond for bond in original_network.bonds}

    data_list = []
    for i in range(0, len(timesteps) - 1, skip):
        timestep = i
        timestep_data = content[timesteps[i] : timesteps[i + 1]]

        # get box info
        x1, x2 = (
            float(timestep_data[5].split(" ")[0]),
            float(timestep_data[5].split(" ")[1]),
        )
        y1, y2 = (
            float(timestep_data[6].split(" ")[0]),
            float(timestep_data[6].split(" ")[1]),
        )
        z1, z2 = (
            float(timestep_data[7].split(" ")[0]),
            float(timestep_data[7].split(" ")[1]),
        )
        box = Box(x1, x2, y1, y2, z1, z2)

        # get atoms info
        new_atoms = []
        for atom_data in timestep_data[9:]:
            atom_data = atom_data.split()
            atom_diameter = [
                atom
                for atom in original_network.atoms
                if atom.atom_id == int(atom_data[0])
            ][0].diameter
            # print(atom_diameter)
            atom = Atom(
                atom_id=int(atom_data[0]),
                atom_diameter=atom_diameter,
                x=float(atom_data[1]),
                y=float(atom_data[2]),
                z=float(atom_data[3]),
            )
            if node_features == 'full' or node_features == 'vel':
                atom.vx = float(atom_data[4])
                atom.vy = float(atom_data[5])
                atom.vz = float(atom_data[6])
            new_atoms.append(atom)

        new_atom_map = {atom.atom_id: atom for atom in new_atoms}
        new_bonds = []
        for edge_index in original_edge_index:
            id1, id2 = edge_index[0], edge_index[1]
            new_atom1 = new_atom_map[id1]
            new_atom2 = new_atom_map[id2]
            bond_stiffness = bond_map[(id1, id2)].bond_coefficient

            # calculate the proper distance between two bonded atoms 
            # keeping periodicity in mind
            if abs(new_atom1.x - new_atom2.x) > original_network.box.x / 2:
                real_x2 = max(new_atom1.x, new_atom2.x)
                real_x1 = min(new_atom1.x, new_atom2.x) + original_network.box.x
            else:
                real_x1 = new_atom1.x
                real_x2 = new_atom2.x
            if abs(new_atom1.y - new_atom2.y) > original_network.box.y / 2:
                real_y2 = max(new_atom1.y, new_atom2.y)
                real_y1 = min(new_atom1.y, new_atom2.y) + original_network.box.y
            else:
                real_y1 = new_atom1.y
                real_y2 = new_atom2.y

            new_length = ((real_x2 - real_x1) ** 2 + (real_y2 - real_y1) ** 2) ** 0.5
            new_bond = Bond(new_atom1, new_atom2)
            new_bond.length = new_length
            new_bond.bond_coefficient = bond_stiffness
            new_bonds.append(new_bond)

        data_list.append(assemble_data(new_atoms, new_bonds, box, node_features=node_features, original_file_path=dump_filepath, step=timestep))

    return data_list


def bulk_load(
    data_dir: str,
    n_networks: int,
    node_features: str = "coord",
    skip: int = 1
) -> list[list[Data]]:
    """Loads data from a provided directory.
    By default returns x and y as node features.
    Assumes that each directory in the `data_dir` contains a network simulation.

    Parameters
    ----------
    data_dir : str
        Path to the data directory

    Returns
    -------
    list[list[Data]]
    """
    sim_dirs = [
        os.path.abspath(os.path.join(data_dir, directory))
        for directory in os.listdir(data_dir)
    ]
    data = []
    for index, sim_dir in enumerate(sim_dirs):
        # reading network from `coord.dat` instead of `*.lmp` to get the accurate information about periodic bonds
        current_network = Network.from_atoms(
            os.path.join(sim_dir, "coord.dat"),
            include_angles=False,
            include_dihedrals=False,
        )
        current_network.write_to_file(os.path.join(sim_dir, "true_network.lmp"))
        dump_file = os.path.join(sim_dir, "dump.lammpstrj")
        print(f"{index+1}/{len(sim_dirs)} : {dump_file}")
        data.append(parse_dump(dump_file, current_network, node_features=node_features, skip=skip))

        # stop loading of desired number of network simulation is parsed
        if index + 1 >= n_networks:
            break
    return data


def network_from_data(data_object: Data) -> Network:
    """Transforms PyTorch Data object into Network object.

    Parameters
    ----------
    `data_object` : Data
        PyTorch Data object containing the graph
    `template` : Network
        original network compressed with lammps from which data object was made

    Returns
    -------
    Network
        updated network
    """
    # create atoms 
    atoms: list[Atom] = []
    for index, node in enumerate(data_object.x):
        atom_id = index + 1
        x = float(node[0])
        y = float(node[1])
        atoms.append(Atom(atom_id=atom_id, atom_diameter=0.0, x=x, y=y, z=0.0))
    
    # update bonds
    bonds: list[Bond] = []
    for index, ((source_id, target_id), (ux, uy, length, k)) in enumerate(zip(data_object.edge_index.T, data_object.edge_attr)):
        atom1 = atoms[int(source_id)]
        atom2 = atoms[int(target_id)]
        
        bond = Bond(atom1, atom2)
        bond.bond_coefficient = float(k)
        bond.length = float(length)
        
        bonds.append(bond)

    box = data_object.box
    
    header = Header(
        atoms=atoms,
        bonds=bonds,
        box=box,
    )

    return Network(atoms, bonds, box, header, masses={1: 100000.0})


def dump(data: list[Data], filedir: str = "", filename: str = "dump_custom.lammpstrj"):
    """Writes a lammps trajctory file from the list of PyTorch Geometric
    Data objects.

    Parameters
    ----------
    `data` : list[Data]
        list of Data objects to dump into a file as trajectory
    `filedir` : str, optional
        directory to write the output file to, by default ""
    `filename` : str, optional
        output file name, by default "dump_custom.lammpstrj"
    """
    filepath = os.path.join(filedir, filename)
    print(filepath)
    with open(filepath, "w", encoding="utf8") as f:
        for index, data_object in enumerate(data):
            f.write("ITEM: TIMESTEP\n")
            f.write(f"{index}\n")
            f.write("ITEM: NUMBER OF ATOMS\n")
            f.write(f"{data_object.x.shape[0]}\n")
            f.write("ITEM: BOX BOUNDS pp pp pp\n")
            f.write(f"{data_object.box.x1} {data_object.box.x2}\n")
            f.write(f"{data_object.box.y1} {data_object.box.y2}\n")
            f.write(f"{data_object.box.z1} {data_object.box.z2}\n")
            f.write(
                "ITEM: ATOMS id x y z vx vy vz\n"
            )  # TODO: make this atoms header dynamic
            for node_index, node in enumerate(data_object.x):
                atom_line = f"{data_object.atom_ids[node_index]} {node[0]} {node[1]} {0} {node[2]} {node[3]} {0}\n"
                f.write(atom_line)


def bulk_dump(data_dir: str):
    # TODO: write this function
    raise NotImplementedError


if __name__ == "__main__":
    pass