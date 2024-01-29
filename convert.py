"""
A small collection of functions which parse data from lammps dump file
into a PyTorch Geometric `Data` object and back.

Compared to the previous data parcing pipeline, introduces two extra attribiutes
into the PyTroch Data object, namely `box` and `atom_ids`, which are needed for
the lammps trajectory file format.

Intended as a substite for the helpers.py file in the future.
"""
from copy import deepcopy
import numpy as np
import os
from sympy import per
import torch
from torch_geometric.data import Data

import network


def assemble_data(
    atoms: list[network.Atom], bonds: list[network.Bond], box: network.Box, node_features: str = "full"
) -> Data:
    """Helper function, part of the `parse_dump()`. Assembles the pytroch_geometric `Data` object.
    By default returns only x and y as node features.

    Parameters
    ----------
    count : int
        Index of the current `Data` object
    atoms : list
        list of `Atom` objects to become node features
    bonds : list
        list of `Bond` objects to become edge features
    node_features : str, optional
        what node feature to include, by default "coord"

    Returns
    -------
    Data
        pytorch_geometric `Data` object
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
            node_features = torch.tensor([[0] for i in range(len(atoms))])
        case "vel":
            try:
                node_features = torch.tensor([[atom.vx, atom.vy] for atom in atoms])
            except AttributeError:
                raise Exception("Atoms don't have velocity info")
        case "coord":
            node_features = torch.tensor([[atom.x, atom.y] for atom in atoms])
        case "full":
            node_features = torch.tensor(
                [[atom.x, atom.y, atom.vx, atom.vy] for atom in atoms]
            )
        case _:
            raise Exception(f"{node_features} not recognized as feature!")

    return Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        box=box,
        atom_ids=atom_ids,
    )


def parse_dump(
    dump_filepath: str,
    original_network: network.Network,
    node_features: str = "full",
    skip: int = 1,
) -> list[Data]:
    """Parses a lammps dump file. By default returns x, y, vx, and vy as node features.

    Parameters
    ----------
    dump_filepath : str
        Path to the lammps trajectory files
    original_network : network.Network
        simulated network to get the accurate information about periodic bonds
    node_features : str, optional
        what node features to parse to parse from trajectory file, by default "full"
    skip : int, optional
        load each n-th step starting from the first. 1 means load each step and skip none save time by increasing

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

    original_connections = original_network.bonds

    data_list = []
    for i in range(0, len(timesteps) - 1, skip):
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
        box = network.Box(x1, x2, y1, y2, z1, z2)

        # get atoms info
        atoms = []
        for atom_data in timestep_data[9:]:
            atom_data = atom_data.split()
            atom_diameter = [atom for atom in original_network.atoms if atom.atom_id == int(atom_data[0])][0].diameter
            # print(atom_diameter)
            atom = network.Atom(
                atom_id=int(atom_data[0]),
                diameter=atom_diameter,
                x=float(atom_data[1]),
                y=float(atom_data[2]),
                z=float(atom_data[3]),
            )
            atom.vx = float(atom_data[4])
            atom.vy = float(atom_data[5])
            atom.vz = float(atom_data[6])
            atoms.append(atom)

        new_connections = deepcopy(original_connections)
        for connection in new_connections:
            atom1, atom2 = connection.atom1, connection.atom2
            
        
        data_list.append(assemble_data(atoms, bonds, box, node_features=node_features))

    return data_list


def bulk_load(
    data_dir: str, n_networks: int, node_features: str = "full", skip: int = 1
) -> list[list[Data]]:
    """Loads data from a provided directory. By default returns x, y, vx, and vy as node features.
    Assumes that each directory in the `data_dir` contains a network simulation.

    Parameters
    ----------
    data_dir : str
        Path to the data directory

    Returns
    -------
    list[list[Data]]
    """
    network_sims = [
        os.path.abspath(os.path.join(data_dir, directory, "sim"))
        for directory in os.listdir(data_dir)
    ]

    data = []
    for index, sim in enumerate(network_sims):
        # reading network from `coord.dat` instead of `*.lmp` to get the accurate information about periodic bonds
        current_network = network.Network.from_atoms(
            os.path.join(sim, "../", "coord.dat"),
            include_angles=False,
            include_dihedrals=False,
        )
        current_network.write_to_file(os.path.join(sim, "true_network.lmp"))
        dump_file = os.path.join(sim, "dump.lammpstrj")
        print(f"{index+1}/{len(network_sims)} : {dump_file}")
        data.append(
            parse_dump(
                dump_file, current_network, node_features=node_features, skip=skip
            )
        )

        # stop loading of desired number of network simulation is parsed
        if index+1 >= n_networks:
            break
    return data


def dump(data: list[Data], filedir: str = "", filename: str = "dump_custom.lammpstrj"):
    """Writes a lammps trajctory file from the list of PyTorch Geometric
    Data objects.

    Parameters
    ----------
    data : list[Data]
        list of Data objects to dump into a file as trajectory
    filedir : str, optional
        directory to write the output file to, by default ""
    filename : str, optional
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
    example = network.Network.from_data_file(
        "true_network.lmp", include_angles=False, include_dihedrals=False
    )
    data = parse_dump("dump.lammpstrj", example)
    dump(data)
