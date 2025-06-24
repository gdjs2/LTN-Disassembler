import networkx as nx
import pyghidra
from sys import argv
from typing import List
from loguru import logger

from blocks_helper import *
from pyvis.network import Network

if not pyghidra.started():
    pyghidra.start()

from ghidra.program.model.address import AddressSet

def _get_fallthrough_edges(blocks: List):
    """
    Get fallthrough edges from the blocks.
    """
    fall_through_edges = []
    blocks.sort(key=lambda b: b.start_address)

    for i in range(len(blocks) - 1):
        current_block = blocks[i]
        last_instr = current_block.pseudo_instrs[-1]
        if last_instr is None: continue
        if last_instr.hasFallthrough():
            next_block = blocks[i + 1]
            if next_block is not None:
                fall_through_edges.append((current_block, next_block))
    return fall_through_edges

def _bisearch_addr_in_blocks(blocks: List, addr):
    """
    Binary search to find the block containing the address.
    Make sure blocks are sorted by (section_name, start_address).
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

def _get_call_edges(blocks: List, listing):
    """
    Get call edges between blocks.
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

def create_graph(flat_api):
    """
    Create a directed graph from the functions in the program.
    """
    program = flat_api.getCurrentProgram()
    function_manager = program.getFunctionManager()
    listing = program.getListing()
    memory = program.getMemory()

    code_blocks = extract_code_blocks(listing, memory)
    data_blocks = extract_data_blocks(listing, memory)
    # unknown_blocks = extract_unknown_blocks(memory, code_blocks + data_blocks)
    blocks = [*code_blocks, *data_blocks]
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