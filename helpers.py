"""
A number of helper functions not relevant to the main logic.
"""

from os import listdir, path

import matplotlib.pyplot as plt
import networkx as nx
# import numpy as np
# import torch
from torch_geometric.data import Data

# import network


# def add_spaces(string: str, width: int, indent: str = "right"):
#     """
#     If the string is longer than provided width,
#     returns the original string without change.
#     """
#     if width <= len(string):
#         return string
#     spaces_to_add = (width - len(string)) * " "
#     if indent == "right":
#         return spaces_to_add + string
#     if indent == "left":
#         return string + spaces_to_add


# def table_row(items: list, widths: list, indent: str = "right") -> str:
#     """
#     Creates a string with the certain number of spaces between words
#     alligned to either right or left
#     """
#     line = []
#     for item, width in zip(items, widths):
#         line.append(add_spaces(str(item), width, indent))

#     return "".join(line) + "\n"


# def assemble_data(count: int, atoms: list, bonds: list, node_features: str = "full") -> Data:
#     """Helper function, part of the `parse_dump()`. Assembles the pytroch_geometric `Data` object.
#     By default returns only x and y as node features.

#     Parameters
#     ----------
#     count : int
#         Index of the current `Data` object
#     atoms : list
#         list of `Atom` objects to become node features
#     bonds : list
#         list of `Bond` objects to become edge features
#     node_features : str, optional
#         what node feature to include, by default "coord"

#     Returns
#     -------
#     Data
#         pytorch_geometric `Data` object
#     """
#     # mapping of atomic IDs to their list indices
#     id_to_index_map = {atom.atom_id: i for atom, i in zip(atoms, range(len(atoms)))}

#     # edges as defined with lammps IDs
#     edges_with_ids = [(bond.atom1.atom_id, bond.atom2.atom_id) for bond in bonds]

#     # edges as defined with indices
#     edges_with_indices = [
#         (id_to_index_map[node1], id_to_index_map[node2])
#         for node1, node2 in edges_with_ids
#     ]

#     edge_index = torch.tensor(np.array(edges_with_indices).T)
#     edge_vectors = torch.stack([torch.tensor([bond.atom1.x - bond.atom2.x, bond.atom1.y-bond.atom2.y]) for bond in bonds])
#     edge_lengths = torch.tensor([bond.length for bond in bonds])
#     # edge vector and its length
#     edge_attr = [torch.cat((v, torch.tensor([length]))) for v, length in zip(edge_vectors, edge_lengths)]
#     edge_attr = torch.stack(edge_attr)

    
#     match node_features:
#         case "dummy":
#             node_features = torch.tensor([[0] for i in range(len(atoms))])
#         case "vel":
#             try:
#                 node_features = torch.tensor([[atom.vx, atom.vy] for atom in atoms])
#             except AttributeError:
#                 raise Exception("Atoms don't have velocity info")
#         case "coord":
#             node_features = torch.tensor([[atom.x, atom.y] for atom in atoms])
#         case "full":
#             node_features = torch.tensor([[atom.x, atom.y, atom.vx, atom.vy] for atom in atoms])
#         case _:
#             raise Exception(f"{node_features} not recognized as feature!")

#     return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor([1]))


# def parse_dump(dump_file: str, original_network: network.Network, node_features: str = "full", skip: int = 1) -> list[Data]:
#     """Parses a lammps dump file. By default returns only x and y as node features. 

#     Parameters
#     ----------
#     dump_file : str
#         Path to the lammps trajectory files
#     original_network : network.Network
#         simulated network to get the accurate information about periodic bonds
#     node_features : str, optional
#         what node features to parse to parse from trajectory file, by default "full"
#     skip : int, optional
#         load each n-th step starting from the first. 1 means load each step and skip none save time by increasing

#     Returns
#     -------
#     list[Data]
#         list of pytorch_geometric Data objects
#     """
#     with open(dump_file, "r", encoding="utf8") as f:
#         content = f.readlines()

#     timesteps: list[int] = []
#     for index, line in enumerate(content):
#         if "ITEM: TIMESTEP" in line:
#             timesteps.append(index)

#     data_list = []
#     count = -1

#     for i in range(0, len(timesteps), skip):
#         count +=1
#         timestep_data = content[timesteps[i] : timesteps[i+1]]
#     # for index, step in enumerate(timesteps[:-1]):
#     #     count += 1
#     #     timestep_data = content[step : timesteps[index + 1]]
#         atoms = []
#         for atom_data in timestep_data[9:]:
#             atom_data = atom_data.split()
#             atom = network.Atom(
#                 atom_id=int(atom_data[0]),
#                 diameter=1.0,
#                 x=float(atom_data[1]),
#                 y=float(atom_data[2]),
#                 z=float(atom_data[3]),
#             )
#             atom.vx = float(atom_data[4])
#             atom.vy = float(atom_data[5])
#             atom.vz = float(atom_data[6])

#             atoms.append(atom)
        
#         bonds = original_network.bonds

#         data_list.append(assemble_data(count, atoms, bonds, node_features=node_features))
#     return data_list


# def bulk_load(data_dir: str, node_features: str = "full", skip: int = 1) -> list[list[Data]]:
#     """Loads data from a provided directory.

#     Parameters
#     ----------
#     data_dir : str
#         Path to the data directory

#     Returns
#     -------
#     list[list[Data]]
#     """
#     network_sims = [
#         path.abspath(path.join(data_dir, directory, "sim"))
#         for directory in listdir(data_dir)
#     ]

#     data = []
#     for sim in network_sims:
#         # reading network from `coord.dat` instead of `*.lmp` to get the accurate information about periodic bonds
#         current_network = network.Network.from_atoms(path.join(sim, "../", "coord.dat"), include_angles=False, include_dihedrals=False)
#         current_network.write_to_file(path.join(sim, "true_network.lmp"))
#         dump_file = path.join(sim, "dump.lammpstrj")
#         print(dump_file)
#         data.append(parse_dump(dump_file, current_network, node_features=node_features, skip=skip))

#     return data


def visualize_graph(data: Data):
    # Create a NetworkX graph from the Data object
    G = nx.Graph()

    # Add nodes
    for i in range(data.x.shape[0]):
        G.add_node(i)

    # Add edges
    edge_index = data.edge_index.cpu().numpy()
    for edge in edge_index.T:
        G.add_edge(edge[0], edge[1])

    # Visualize the graph
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color="skyblue", node_size=200, font_size=10)
    plt.show()


def visualize_combined_graph(original_data, predicted_data):
    # Create a NetworkX graph for the combined data
    G_combined = nx.Graph()

    # Add nodes and edges for the original graph
    for i in range(original_data.x.shape[0]):
        x, y = original_data.x[i][
            :2
        ].tolist()  # Get x and y coordinates from the original data
        G_combined.add_node(i, pos=(x, y))  # Use x and y as node positions

    edge_index_original = original_data.edge_index.cpu().numpy()
    for edge in edge_index_original.T:
        G_combined.add_edge(
            edge[0], edge[1], weight=1
        )  # Weight is set to 1 for the original edges

    # Add nodes and edges for the predicted graph
    offset = original_data.x.shape[0]  # Offset to differentiate nodes between graphs
    for i in range(predicted_data.x.shape[0]):
        x, y = predicted_data.x[i][
            :2
        ].tolist()  # Get x and y coordinates from the predicted data
        G_combined.add_node(i + offset, pos=(x, y))  # Use x and y as node positions

    edge_index_predicted = predicted_data.edge_index.cpu().numpy()
    for edge in edge_index_predicted.T:
        G_combined.add_edge(
            edge[0] + offset, edge[1] + offset, weight=2
        )  # Weight is set to 2 for the predicted edges

    # Separate nodes of the two graphs for coloring
    nodes_original = list(range(original_data.x.shape[0]))
    nodes_predicted = list(
        range(original_data.x.shape[0], G_combined.number_of_nodes())
    )

    # Get the colors for nodes in the predicted graph
    node_colors = [
        "lightcoral" if n in nodes_predicted else "skyblue" for n in G_combined.nodes()
    ]

    # Get edge weights for adjusting edge thickness
    edge_weights = [data["weight"] for _, _, data in G_combined.edges(data=True)]

    # Get node positions for plotting
    pos = {node: pos for node, pos in nx.get_node_attributes(G_combined, "pos").items()}

    # Visualize the combined graph with bold edges
    plt.figure(figsize=(8, 6))  # Set the figure size

    # Draw nodes and edges with adjusted edge thickness and positions
    nx.draw(
        G_combined,
        pos,
        with_labels=True,
        node_color=node_colors,
        node_size=200,
        font_size=10,
        width=edge_weights,
    )

    # Add a header indicating color meanings
    plt.text(
        0.05,
        1.05,
        "Original Graph: Blue\nPredicted Graph: Red",
        transform=plt.gca().transAxes,
        fontsize=12,
    )

    plt.title("Combined Graph with Node Positions", fontsize=12)
    plt.tight_layout()  # Adjust subplot
    plt.show()


def visualize_combined_graph_edges(original_data, predicted_data):
    G_combined = nx.Graph()
    node_colors = []  # List to store node colors for all nodes

    for i in range(original_data.x.shape[0]):
        x, y = original_data.x[i][:2].tolist()
        G_combined.add_node(i, pos=(x, y))
        node_colors.append("skyblue")  # Blue color for original graph nodes

    edge_index_original = original_data.edge_index.cpu().numpy()
    for edge in edge_index_original.T:
        G_combined.add_edge(edge[0], edge[1], weight=1)
        # Set edge color to the same color as the original nodes
        node_colors[edge[0]] = "skyblue"
        node_colors[edge[1]] = "skyblue"

    offset = original_data.x.shape[0]

    for i in range(predicted_data.x.shape[0]):
        x, y = predicted_data.x[i][:2].tolist()
        G_combined.add_node(i + offset, pos=(x, y))
        node_colors.append("lightcoral")  # Red color for predicted graph nodes

    edge_index_predicted = predicted_data.edge_index.cpu().numpy()
    for edge in edge_index_predicted.T:
        G_combined.add_edge(edge[0] + offset, edge[1] + offset, weight=2)
        # Set edge color to the same color as the predicted nodes
        node_colors[edge[0] + offset] = "lightcoral"
        node_colors[edge[1] + offset] = "lightcoral"

    edge_weights = [data["weight"] for _, _, data in G_combined.edges(data=True)]
    pos = {node: pos for node, pos in nx.get_node_attributes(G_combined, "pos").items()}

    plt.figure(figsize=(8, 6))

    nx.draw(
        G_combined,
        pos,
        with_labels=True,
        node_color=node_colors,
        node_size=200,
        font_size=10,
        width=edge_weights,
    )

    plt.text(
        0.05,
        1.05,
        "Original Graph: Blue\nPredicted Graph: Red",
        transform=plt.gca().transAxes,
        fontsize=12,
    )

    plt.title("Combined Graph with Node Positions", fontsize=12)
    plt.tight_layout()
    plt.show()


def visualize_combined_graph_with_difference(original_data, predicted_data):
    G_combined = nx.Graph()
    node_colors = []  # List to store node colors for all nodes

    # Extract node positions for both graphs
    original_positions = nx.get_node_attributes(original_data.pos)
    predicted_positions = nx.get_node_attributes(predicted_data.pos)

    # Calculate the most middle node for both graphs
    middle_node_original = max(
        original_positions, key=lambda k: original_positions[k][0]
    )
    middle_node_predicted = max(
        predicted_positions, key=lambda k: predicted_positions[k][0]
    )

    # Calculate the difference in x and y coordinates
    x_diff = (
        original_positions[middle_node_original][0]
        - predicted_positions[middle_node_predicted][0]
    )
    y_diff = (
        original_positions[middle_node_original][1]
        - predicted_positions[middle_node_predicted][1]
    )

    for i in range(original_data.x.shape[0]):
        x, y = original_data.x[i][:2].tolist()
        G_combined.add_node(i, pos=(x, y))
        node_colors.append("skyblue")  # Blue color for original graph nodes

    edge_index_original = original_data.edge_index.cpu().numpy()
    for edge in edge_index_original.T:
        G_combined.add_edge(edge[0], edge[1], weight=1)
        # Set edge color to the same color as the original nodes
        node_colors[edge[0]] = "skyblue"
        node_colors[edge[1]] = "skyblue"

    offset = original_data.x.shape[0]

    for i in range(predicted_data.x.shape[0]):
        x, y = predicted_data.x[i][:2].tolist()
        # Add the difference to the predicted node positions
        x += x_diff
        y += y_diff
        G_combined.add_node(i + offset, pos=(x, y))
        node_colors.append("lightcoral")  # Red color for predicted graph nodes

    edge_index_predicted = predicted_data.edge_index.cpu().numpy()
    for edge in edge_index_predicted.T:
        G_combined.add_edge(edge[0] + offset, edge[1] + offset, weight=2)
        # Set edge color to the same color as the predicted nodes
        node_colors[edge[0] + offset] = "lightcoral"
        node_colors[edge[1] + offset] = "lightcoral"

    edge_weights = [data["weight"] for _, _, data in G_combined.edges(data=True)]
    pos = {node: pos for node, pos in nx.get_node_attributes(G_combined, "pos").items()}

    plt.figure(figsize=(8, 6))

    nx.draw(
        G_combined,
        pos,
        with_labels=True,
        node_color=node_colors,
        node_size=200,
        font_size=10,
        width=edge_weights,
    )

    plt.text(
        0.05,
        1.05,
        "Original Graph: Blue\nPredicted Graph: Red",
        transform=plt.gca().transAxes,
        fontsize=12,
    )

    plt.title("Combined Graph with Node Positions and Difference", fontsize=12)
    plt.tight_layout()
    plt.show()
