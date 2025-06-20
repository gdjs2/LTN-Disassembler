import numpy as np
import loguru
import torch
import pyghidra
import ltn
import ltn.fuzzy_ops
import networkx as nx
from blocks_helper import check_compare_branch, check_def_use, fix_undisassembled_data_blocks, get_feature_vector, check_very_short, pseudo_disassemble_blocks
from models import ByteEmbedder, ByteGRU, MLPClassifier
from graph_helper import extract_code_blocks, extract_data_blocks, extract_unknown_blocks, create_graph, _bisearch_addr_in_blocks
from loguru import logger
from torch.nn.utils.rnn import pad_sequence

if not pyghidra.started():
    pyghidra.start()

from ghidra.program.model.address import AddressSet
from ghidra.app.util import PseudoDisassembler
from ghidra.program.model.lang import RegisterValue

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

def embeddings_from_feature_vector(blocks, avg=None):
    """
    Sample: Generate embeddings from the feature vector of the blocks.
    """
    embeddings = []
    if avg is None:
        avg = [1, 1, 1, 1, 1, 1, 1, 1]  
    for block in blocks:
        # emb = torch.tensor(block.feature_vector, dtype=torch.float32)
        # emb_norm = (emb - emb.min()) / (emb.max() - emb.min()) if emb.max() != emb.min() else emb
        # emb_norm = emb_norm * torch.tensor(avg, dtype=torch.float32)
        embeddings.append(torch.tensor(block.feature_vector, dtype=torch.float32))
    return torch.stack(embeddings, dim=0)


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

        pseudo_disassemble_blocks(blocks, program)
        check_compare_branch(blocks, PseudoDisassembler(program), program)
        check_def_use(blocks, PseudoDisassembler(program))
        check_very_short(blocks)
        get_feature_vector(blocks, PseudoDisassembler(program), program.getReferenceManager(), program.getListing(), program.getMemory())
        # fix_undisassembled_data_blocks(blocks)

        # embeddings = generate_embeddings(blocks, flat_api)
        # embeddings = generate_random_embeddings(blocks, dim=8)
        embeddings = embeddings_from_feature_vector(blocks)
        with open('blocks.txt', 'w') as f:
            for i, block in enumerate(blocks):
                f.write(f"{block} {embeddings[i].tolist()}\n")
    
    block2idx = {block: i for i, block in enumerate(blocks)}
    idx2block = {i: block for i, block in enumerate(blocks)}

    CodeBlock = ltn.Predicate(MLPClassifier(input_dim=8, hidden_dim1=16, hidden_dim2=32).to(ltn.device))
    # DataBlock = ltn.Predicate(MLPClassifier(input_dim=16, hidden_dim1=64, hidden_dim2=32).to(ltn.device))
    # UnknownBlock = ltn.Predicate(func=lambda x: 1 - (torch.abs(CodeBlock(x) - 0.5) + torch.abs(DataBlock(x) - 0.5)))

    SatAgg = ltn.fuzzy_ops.SatAgg(ltn.fuzzy_ops.AggregPMeanError(p=4))
    Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=4), quantifier='f')
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
    very_short = get_identity_vars(blocks, block2idx, embeddings, "very_short_flg", True)
    high_zero_rate = get_identity_vars(blocks, block2idx, embeddings, "high_zero_rate_flg", True)
    high_def_use_rate = get_identity_vars(blocks, block2idx, embeddings, "high_def_use_rate_flg", True)

    disasm_gt_cb = get_identity_vars(blocks, block2idx, embeddings, "type", "Code")
    disasm_gt_db = get_identity_vars(blocks, block2idx, embeddings, "type", "Data")
    # disasm_gt_ub = get_identity_vars(blocks, block2idx, embeddings, "type", "Unknown")

    optimizer = torch.optim.Adam(list(CodeBlock.parameters()), lr=0.001)
    epochs = 3000

    for epoch in range(epochs):
        
        ltn.diag(x_ft, y_ft)
        ltn.diag(x_call, y_call)

        optimizer.zero_grad()
        sat_agg = SatAgg(
            Forall([disasm_gt_cb], CodeBlock(disasm_gt_cb)),
            Forall([disasm_gt_db], Not(CodeBlock(disasm_gt_db))),

            Forall([cond_brch_t], CodeBlock(cond_brch_t)),
            Forall([cond_brch_f], Not(CodeBlock(cond_brch_f))),
            # Forall([def_use], CodeBlock(def_use)),
            Forall([high_zero_rate], Not(CodeBlock(high_zero_rate))),
            # Forall([high_def_use_rate], CodeBlock(high_def_use_rate)),

            # Forall([x_ft, y_ft], Equiv(CodeBlock(x_ft), CodeBlock(y_ft))),
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

    with open("/home/zhaoqi.xiao/Projects/Loadstar/Dataset/NS_1/labeled/108.58.252.74.txt", "r") as f:
        lines = f.readlines()
    lines = [line.strip()[-1] for line in lines]

    tp_code = 0
    tp_data = 0
    fp_code = 0
    fp_data = 0
    fn_code = 0
    fn_data = 0
    cnt = 0
    with open("detail.txt", "w") as f:
        for i, block in enumerate(blocks):
            if block.start_address.getOffset() > 0x16f4c:
                break
            code_flg = 1
            if CodeBlock(ltn.Constant(embeddings[i])).value > 0.40 and block.type != "FixedData":
                code_flg = 0

            addr = block.start_address
            while addr <= block.end_address:
                if cnt // 4 >= len(lines):
                    logger.error(f"Not enough labels in lines for address {addr}.")
                    break
                if code_flg == int(lines[cnt//4]):
                    if code_flg == 0:
                        tp_code += 1
                    else:
                        tp_data += 1
                else:
                    if code_flg == 0:
                        fp_code += 1
                        fn_data += 1
                    else:
                        fp_data += 1
                        fn_code += 1
                    
                f.write(f"{addr} expected: {lines[cnt//4]}, predicted: {code_flg}\n")
                addr = addr.add(1)
                cnt += 1

    # logger.info(f"True Positives: {tp}, False Positives: {fp}, Precision: {tp / (tp + fp) if (tp + fp) > 0 else 0.0}")
    logger.info(f"Code Precision: {tp_code / (tp_code + fp_code) if (tp_code + fp_code) > 0 else 0.0}, Code Recall: {tp_code / (tp_code + fn_code) if (tp_code + fn_code) > 0 else 0.0}, Data Precision: {tp_data / (tp_data + fp_data) if (tp_data + fp_data) > 0 else 0.0}, Data Recall: {tp_data / (tp_data + fn_data) if (tp_data + fn_data) > 0 else 0.0}")

    with open("result.txt", "w") as f:
        for i, block in enumerate(blocks):
            f.write(f"{block} {embeddings[i]} CodeBlock={CodeBlock(ltn.Constant(embeddings[i])).value.item()}\n")
