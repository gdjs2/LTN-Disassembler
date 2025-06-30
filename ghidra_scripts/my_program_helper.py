import ltn

from datetime import datetime
from graph_helper import *

class MyProgram:
    def __init__(self: Self, flat_api: FlatProgramAPI) -> None:
        """
        Initialize the Program instance with a FlatProgramAPI instance.
        Args:
            flat_api (FlatProgramAPI): Flat API instance to interact with the Ghidra program.
        """
        self.flat_api = flat_api
        
        program = flat_api.getCurrentProgram()
        listing = program.getListing()
        memory = program.getMemory()
        ref_manager = program.getReferenceManager()
        
        # Create relational graph
        self.graph = create_graph(flat_api)

        # Instance variable `self.blocks` will hold the blocks in sorted order by their start address
        self.blocks: list[Block] = list(self.graph.nodes)
        self.blocks.sort(key=lambda b: b.start_address)

        # Get feature vectors for the blocks
        get_feature_vector(self.blocks, ref_manager, listing, memory)
        check_compare_branch(self.blocks, program)
        check_very_short(self.blocks)

        # Generate embeddings from the feature vectors
        self.embeddings = generate_embeddings_from_feature_vector(self.blocks)
        self.block2idx = {block: idx for idx, block in enumerate(self.blocks)}

    def get_rel_vars(self: Self, edge_type: str) -> tuple[ltn.Variable, ltn.Variable] | tuple[None, None]:
        """
        Get the relational variables for the specified edge type in the graph.
        Args:
            edge_type (str): The type of edge to filter by (e.g., "call", "fallthrough").
        Returns:
            tuple[ltn.Variable, ltn.Variable]: Left and right relational variables for the specified edge type.
        """
        edges_idx = torch.tensor([(self.block2idx[u], self.block2idx[v]) for u, v in self.graph.edges() if self.graph[u][v]['type'] == edge_type], dtype=torch.long)
        if edges_idx.numel() == 0:
            return (None, None)
        left_idx = edges_idx[:, 0]
        right_idx = edges_idx[:, 1]
        left_embeddings = self.embeddings[left_idx]
        right_embeddings = self.embeddings[right_idx]
        return ltn.Variable(f"{edge_type}_rel_left", left_embeddings), ltn.Variable(f"{edge_type}_rel_right", right_embeddings)
        
    def get_identity_vars(self: Self, field: str, val: Literal["Code", "Data"] | bool) -> ltn.Variable | None:
        """
        Get the identity variables for a specific field and value in the blocks.
        Args:
            field (str): The field to filter by (e.g., "cond_branch_flg", "type").
            val (str | bool): The value to filter by (e.g., "Code", "Data", True, False).
        Returns:
            ltn.Variable | None: The identity variable for the specified field and value, or None if no blocks match.
        """
        matched_blocks_idx = torch.tensor([self.block2idx[b] for b in self.blocks if getattr(b, field) == val], dtype=torch.long)
        if matched_blocks_idx.numel() == 0:
            return None
        block_embeddings = self.embeddings[matched_blocks_idx]
        return ltn.Variable(f"{field}_{val}_id", block_embeddings)


