import numpy as np
import loguru
import torch
import pyghidra
import ltn
import ltn.fuzzy_ops
import networkx as nx
from blocks_helper import check_compare_branch, check_def_use
from models import ByteEmbedder, ByteGRU, MLPClassifier
from graph_helper import extract_code_blocks, extract_data_blocks, extract_unknown_blocks, create_graph
from loguru import logger
from torch.nn.utils.rnn import pad_sequence

if not pyghidra.started():
    pyghidra.start()

from ghidra.program.model.address import AddressSet

def generate_embeddings(blocks, flat_apiFlatProgramAPI):
    """
    Sample: Generate embeddings for the blocks.
    """
    listing = flat_apiFlatProgramAPI.getCurrentProgram().getListing()
    memory = flat_apiFlatProgramAPI.getCurrentProgram().getMemory()
    embeddings = []
    for block in blocks:
        block_emb = []
        addr_set = AddressSet(block.start_address, block.end_address)
        instr_iter = listing.getInstructions(addr_set, False)
        if instr_iter is None or not instr_iter.hasNext():
            logger.warning(f"Cannot get instructions for block {block}.")
            continue
        while instr_iter.hasNext():
            vec = torch.zeros(16, dtype=torch.float32)
            instr = instr_iter.next()
            start_addr = instr.getAddress()
            length = instr.getLength()
            bytes_array = bytearray(length)
            memory.getBytes(start_addr, bytes_array)
            byte_len = min(len(bytes_array), 16)
            for i in range(byte_len):
                vec[i] = bytes_array[i] / 255.0
            vec[15] = byte_len / 16.0
            block_emb.append(vec)
        embeddings.append(torch.tensor(block_emb))
    return embeddings

def generate_random_embeddings(blocks, dim=16):
    n = len(blocks)
    embeddings = torch.randn(n, dim)
    return embeddings


def get_rel_vars(graph: nx.DiGraph, block2idx: dict, embeddings: torch.Tensor, edge_type):
    """
    Sample: Get the first-order logic variables of [edge_type] for the graph.
    """
    edges = torch.tensor([(block2idx[u], block2idx[v]) for u, v in graph.edges() if graph[u][v]['type'] == edge_type], dtype=torch.long)
    left_idx = edges[:, 0]
    right_idx = edges[:, 1]
    left_embeddings = embeddings[left_idx] 
    right_embeddings = embeddings[right_idx]
    return  ltn.Variable(f"{edge_type}_rel_left", left_embeddings), ltn.Variable(f"{edge_type}_rel_right", right_embeddings)

def get_identity_vars(blocks, block2idx, embeddings, field, val):
    """
    Sample: Get the first-order logic variables of field == val for the graph.
    """
    blocks = torch.tensor([block2idx[b] for b in blocks if getattr(b, field) == val], dtype=torch.long)
    block_embeddings = embeddings[blocks]
    return ltn.Variable(f"{field}_{val}_id", block_embeddings)
    
if __name__ == '__main__':

    with pyghidra.open_program('/home/zhaoqi.xiao/Projects/Loadstar/Dataset/NS_1/bins/108.58.252.74.PRG', language='ARM:LE:32:Cortex') as flat_api:
        program = flat_api.getCurrentProgram()
        blocks, graph = create_graph(flat_api)
        check_compare_branch(blocks, program.getListing())
        check_def_use(blocks, program.getListing())
        # embeddings = generate_embeddings(blocks, flat_api)
        embeddings = generate_random_embeddings(blocks, dim=16)
        # with open('embeddings.txt', 'w') as f:
        #     for block, embedding in zip(blocks, embeddings):
        #         f.write(f"{block}: {embedding.tolist()}\n")
    
    block2idx = {block: i for i, block in enumerate(blocks)}
    idx2block = {i: block for i, block in enumerate(blocks)}

    CodeBlock = ltn.Predicate(MLPClassifier(input_dim=16, hidden_dim1=64, hidden_dim2=32).to(ltn.device))
    # DataBlock = ltn.Predicate(MLPClassifier(input_dim=16, hidden_dim1=64, hidden_dim2=32).to(ltn.device))
    # UnknownBlock = ltn.Predicate(func=lambda x: 1 - (torch.abs(CodeBlock(x) - 0.5) + torch.abs(DataBlock(x) - 0.5)))

    SatAgg = ltn.fuzzy_ops.SatAgg(ltn.fuzzy_ops.AggregPMeanError(p=6))
    Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=6), quantifier='f')
    Exists = ltn.Quantifier(ltn.fuzzy_ops.AggregPMean(p=3), quantifier='e')

    Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach())
    Equiv = ltn.Connective(ltn.fuzzy_ops.Equiv(ltn.fuzzy_ops.AndProd(), ltn.fuzzy_ops.ImpliesReichenbach()))
    Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
    And = ltn.Connective(ltn.fuzzy_ops.AndProd())

    # GRU = ltn.Function(ByteGRU(emb_dim=16, hidden_dim=32, out_dim=16).to(ltn.device))
    x_ft, y_ft = get_rel_vars(graph, block2idx, embeddings, "fallthrough")
    x_call, y_call= get_rel_vars(graph, block2idx, embeddings, "call")

    cond_brch_t = get_identity_vars(blocks, block2idx, embeddings, "cond_branch_flg", True)
    cond_brch_f = get_identity_vars(blocks, block2idx, embeddings, "cond_branch_flg", False)
    def_use = get_identity_vars(blocks, block2idx, embeddings, "def_use_flg", True)
    

    disasm_gt_cb = get_identity_vars(blocks, block2idx, embeddings, "type", "Code")
    disasm_gt_db = get_identity_vars(blocks, block2idx, embeddings, "type", "Data")
    disasm_gt_ub = get_identity_vars(blocks, block2idx, embeddings, "type", "Unknown")

    optimizer = torch.optim.Adam(list(CodeBlock.parameters()), lr=0.001)
    epochs = 5000

    for epoch in range(epochs):
        
        ltn.diag(x_ft, y_ft)
        ltn.diag(x_call, y_call)

        optimizer.zero_grad()
        sat_agg = SatAgg(
            Forall([disasm_gt_cb], CodeBlock(disasm_gt_cb)),
            Forall([disasm_gt_db], Not(CodeBlock(disasm_gt_db))),
            # Forall([disasm_gt_ub], UnknownBlock(disasm_gt_ub)),
            # Forall([disasm_gt_ub], Not(CodeBlock(disasm_gt_ub))),
            # Forall([disasm_gt_ub], Not(DataBlock(disasm_gt_ub))),

            Forall([cond_brch_t], CodeBlock(cond_brch_t)),
            # Forall([cond_brch_f], Not(CodeBlock(cond_brch_f))),
            Forall([def_use], CodeBlock(def_use)),

            Forall([x_ft, y_ft], Equiv(CodeBlock(x_ft), CodeBlock(y_ft))),
            Forall([x_call, y_call], Equiv(CodeBlock(x_call), CodeBlock(y_call))),
            
        )
        loss = 1. - sat_agg
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            logger.info(f"Epoch {epoch}, Loss: {loss.item()}")
        if loss.item() < 0.01:
            logger.info(f"Early stopping at epoch {epoch}, Loss: {loss.item()}")
            break
    logger.info("Training complete.")
    with open('result.txt', 'w') as f:
        for i, block in enumerate(blocks):
            f.write(f"{block}: CodeBlock() = {CodeBlock(ltn.Constant(embeddings[i])).value}, DataBlock() = {1 - CodeBlock(ltn.Constant(embeddings[i])).value}\n")
    logger.info("Results saved to result.txt.")
