import networkx as nx

from blocks_helper import *
from ghidra.program.flatapi import FlatProgramAPI
from ghidra.program.model.address import AddressSet

def _get_fallthrough_edges(blocks: list[Block]):
    """
    Get fallthrough edges from the blocks.
    Args:
        blocks (list[Block]): List of code blocks.
    Returns:
        list[tuple[Block, Block]]: List of tuples representing fallthrough edges.
    """
    fall_through_edges = []
    blocks.sort(key=lambda b: b.start_address)

    for i in range(len(blocks) - 1):
        current_block = blocks[i]
        if current_block.pseudo_instrs is None:
            continue
        last_instr = current_block.pseudo_instrs[-1]
        if last_instr is None: continue
        if last_instr.hasFallthrough():
            next_block = blocks[i + 1]
            if next_block is not None:
                fall_through_edges.append((current_block, next_block))
    return fall_through_edges

def _bisearch_addr_in_blocks(blocks: list[Block], addr: Address) -> Block|None:
    """
    Binary search to find the block containing the address.
    Make sure blocks are sorted by (section_name, start_address).
    Args:
        blocks (list[Block]): List of blocks to search.
        addr (Address): Address to find.
    Returns:
        Block | None: The block containing the address, or None if not found.
    """
    left, right = 0, len(blocks)
    while left < right:
        mid = (left + right) >> 1
        if blocks[mid].start_address <= addr <= blocks[mid].end_address:
            return blocks[mid]
        elif addr < blocks[mid].start_address:
            right = mid
        elif addr > blocks[mid].end_address:
            left = mid + 1
    # logger.error(f"Address {addr} not found in blocks. May be a external reference.")
    return None

def _get_call_edges(blocks: list[Block], listing: Listing) -> list[tuple[Block, Block]]:
    """
    Get call edges between blocks.
    Args:
        blocks (list[Block]): List of code blocks.
        listing (Listing): Ghidra listing object to get instructions and references.
    Returns:
        list[tuple[Block, Block]]: List of tuples representing call edges.
    """
    call_edges = []
    blocks.sort(key=lambda b: b.start_address)

    for block in blocks:
        if block.type == "Data": continue
        addr_set = AddressSet(block.start_address, block.end_address)
        instructions = listing.getInstructions(addr_set, True)
        for instr in instructions:
            refs = instr.getReferencesFrom()
            for ref in refs:
                if ref.getReferenceType().isFlow():
                    target_block = _bisearch_addr_in_blocks(blocks, ref.getToAddress())
                    if target_block is not None:
                        call_edges.append((block, target_block))
    return call_edges

def create_graph(flat_api: FlatProgramAPI) -> nx.DiGraph:
    """
    Create a directed graph from the functions in the program.
    Args:
        flat_api (FlatProgramAPI): Flat API instance to interact with the Ghidra program.
    Returns:
        nx.DiGraph: Relational graph of the program.
    """
    program = flat_api.getCurrentProgram()
    listing = program.getListing()
    memory = program.getMemory()

    blocks = extract_all_blocks(listing, memory)
    blocks.sort(key=lambda b: b.start_address)
    
    pseudo_disassemble_blocks(blocks, program)

    fall_through_edges = _get_fallthrough_edges(blocks)
    call_edges = _get_call_edges(blocks, listing)
            
    graph = nx.DiGraph()
    graph.add_nodes_from(blocks)
    graph.add_edges_from(fall_through_edges, type="fallthrough")
    graph.add_edges_from(call_edges, type="call")

    return graph
    
    
if __name__ == "__main__":
    with pyghidra.open_program('/home/zhaoqi.xiao/Projects/Loadstar/Dataset/NS_1/bins/108.58.252.74.PRG', language='ARM:LE:32:Cortex') as flat_api:
        graph = create_graph(flat_api)
        blocks = list(graph.nodes)
        blocks.sort(key=lambda b: b.start_address)

    if argv[1] == "debug":
        with open("./debug/graph_helper_blocks.txt", "w") as f:
            for block in blocks:
                f.write(f"{block}\n")

        with open("./debug/graph_helper_edges.txt", "w") as f:
            for u, v in graph.edges():
                f.write(f"{u} -> {v} ({graph[u][v]['type']})\n")
    
        logger.info(f"Debug information saved to ./debug/graph_helper_blocks.txt and ./debug/graph_helper_edges.txt")