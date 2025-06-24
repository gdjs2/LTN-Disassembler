import torch
import pyghidra
import ltn
import ltn.fuzzy_ops
import networkx as nx
from blocks_helper import *
from models import MLPClassifier
from graph_helper import *
from loguru import logger
from sys import argv

if not pyghidra.started():
    pyghidra.start()

from ghidra.program.model.address import AddressSet
from ghidra.app.util import PseudoDisassembler
from ghidra.program.model.lang import RegisterValue

def generate_random_embeddings(blocks, dim=16):
    n = len(blocks)
    embeddings = torch.randn(n, dim)
    return embeddings

def embeddings_from_feature_vector(blocks):
    """
    Generate embeddings from the feature vector of the blocks.
    """
    embeddings = []
    for block in blocks:
        embeddings.append(torch.tensor(block.feature_vector, dtype=torch.float32))
    return torch.stack(embeddings, dim=0)


def get_rel_vars(graph: nx.DiGraph, block2idx: dict, embeddings: torch.Tensor, edge_type):
    """
    Get the first-order logic variables of [edge_type] for the graph.
    """
    edges = torch.tensor([(block2idx[u], block2idx[v]) for u, v in graph.edges() if graph[u][v]['type'] == edge_type], dtype=torch.long)
    if edges.numel() == 0:
        return None, None
    left_idx = edges[:, 0]
    right_idx = edges[:, 1]
    left_embeddings = embeddings[left_idx] 
    right_embeddings = embeddings[right_idx]
    return  ltn.Variable(f"{edge_type}_rel_left", left_embeddings), ltn.Variable(f"{edge_type}_rel_right", right_embeddings)

def get_identity_vars(blocks, block2idx, embeddings, field, val):
    """
    Get the first-order logic variables of field == val for the graph.
    """
    blocks = torch.tensor([block2idx[b] for b in blocks if getattr(b, field) == val], dtype=torch.long)
    if blocks.numel() == 0:
        return None
    block_embeddings = embeddings[blocks]
    return ltn.Variable(f"{field}_{val}_id", block_embeddings)

def train(flat_api, CodeBlock=None):
    """
    Single iteration of training
    """
    program = flat_api.getCurrentProgram()
    graph = create_graph(flat_api)
    blocks = list(graph.nodes())
    ref_manager = program.getReferenceManager()
    listing = program.getListing()
    memory = program.getMemory()
    blocks.sort(key=lambda b: b.start_address)

    get_feature_vector(blocks, PseudoDisassembler(program), ref_manager, listing, memory)
    check_compare_branch(blocks, program)
    check_very_short(blocks)
    
    embeddings = embeddings_from_feature_vector(blocks)

    block2idx = {block: i for i, block in enumerate(blocks)}
    idx2block = {i: block for i, block in enumerate(blocks)}
    if not CodeBlock: CodeBlock = ltn.Predicate(MLPClassifier(input_dim=embeddings.size(1), hidden_dim1=16, hidden_dim2=32).to(ltn.device))

    SatAgg = ltn.fuzzy_ops.SatAgg(ltn.fuzzy_ops.AggregPMeanError(p=4))
    Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=4), quantifier='f')
    Exists = ltn.Quantifier(ltn.fuzzy_ops.AggregPMean(p=3), quantifier='e')

    Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach())
    Equiv = ltn.Connective(ltn.fuzzy_ops.Equiv(ltn.fuzzy_ops.AndProd(), ltn.fuzzy_ops.ImpliesReichenbach()))
    Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
    And = ltn.Connective(ltn.fuzzy_ops.AndProd())

    x_call, y_call= get_rel_vars(graph, block2idx, embeddings, "call")
    x_ft, y_ft = get_rel_vars(graph, block2idx, embeddings, "fallthrough")

    cond_brch_t = get_identity_vars(blocks, block2idx, embeddings, "cond_branch_flg", True)
    cond_brch_f = get_identity_vars(blocks, block2idx, embeddings, "cond_branch_flg", False)
    high_zero_rate = get_identity_vars(blocks, block2idx, embeddings, "high_zero_rate_flg", True)

    disasm_gt_cb = get_identity_vars(blocks, block2idx, embeddings, "type", "Code")
    disasm_gt_db = get_identity_vars(blocks, block2idx, embeddings, "type", "Data")

    optimizer = torch.optim.Adam(list(CodeBlock.parameters()), lr=0.001)
    epochs = 2000

    for epoch in range(epochs):
        
        ltn.diag(x_ft, y_ft)
        ltn.diag(x_call, y_call)

        optimizer.zero_grad()

        sat_agg_list = []
        if disasm_gt_cb: 
            sat_agg_list.append(Forall([disasm_gt_cb], CodeBlock(disasm_gt_cb)))
        if disasm_gt_db:
            sat_agg_list.append(Forall([disasm_gt_db], Not(CodeBlock(disasm_gt_db))))
        if cond_brch_t:
            sat_agg_list.append(Forall([cond_brch_t], CodeBlock(cond_brch_t)))
        if cond_brch_f:
            sat_agg_list.append(Forall([cond_brch_f], Not(CodeBlock(cond_brch_f))))
        if high_zero_rate:
            sat_agg_list.append(Forall([high_zero_rate], Not(CodeBlock(high_zero_rate))))
        if x_ft and y_ft:
            sat_agg_list.append(Forall([x_ft, y_ft], Equiv(CodeBlock(x_ft), CodeBlock(y_ft))))
        if x_call and y_call:
            sat_agg_list.append(Forall([x_call, y_call], Equiv(CodeBlock(x_call), CodeBlock(y_call))))

        sat_agg = SatAgg(*sat_agg_list)

        loss = 1. - sat_agg
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            logger.info(f"Epoch {epoch}, Loss: {loss.item()}")
        if loss.item() < 0.01:
            logger.info(f"Early stopping at epoch {epoch}, Loss: {loss.item()}")
            break
    logger.info("Training complete.")
    return (blocks, embeddings, CodeBlock)

def evaluate(blocks, embeddings, CodeBlock, label_file):
    """
    Evaluate the model on the labeled data.
    """
    with open(label_file, "r") as f:
        lines = f.readlines()
    lines = [line.strip()[-1] for line in lines]

    tp_code, tp_data, fp_code, fp_data, fn_code, fn_data, cnt = 0, 0, 0, 0, 0, 0, 0

    with open("debug/detail.txt", "w") as f:
        for i, block in enumerate(blocks):
            block_opt_flg = True
            if block.start_address.getOffset() > 0x16f4c:
                break
            code_flg = 1
            if CodeBlock(ltn.Constant(embeddings[i])).value > 0.40:
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
                    block_opt_flg = False
                addr = addr.add(1)
                cnt += 1
            if not block_opt_flg:
                f.write("\n")

    code_prec = tp_code / (tp_code + fp_code) if (tp_code + fp_code) > 0 else 0.0
    code_rec  = tp_code / (tp_code + fn_code) if (tp_code + fn_code) > 0 else 0.0
    code_f1   = 2 * code_prec * code_rec / (code_prec + code_rec) if (code_prec + code_rec) > 0 else 0.0

    data_prec = tp_data / (tp_data + fp_data) if (tp_data + fp_data) > 0 else 0.0
    data_rec  = tp_data / (tp_data + fn_data) if (tp_data + fn_data) > 0 else 0.0
    data_f1   = 2 * data_prec * data_rec / (data_prec + data_rec) if (data_prec + data_rec) > 0 else 0.0

    logger.info(
        f"Code  P/R/F1: {code_prec:.5f}/{code_rec:.5f}/{code_f1:.5f} | "
        f"Data  P/R/F1: {data_prec:.5f}/{data_rec:.5f}/{data_f1:.5f}"
    )

    with open("debug/result.txt", "w") as f:
        for i, block in enumerate(blocks):
            f.write(
                f"{block}\n"
                f"  Embedding : {embeddings[i]}\n"
                f"  CodeBlock : {CodeBlock(ltn.Constant(embeddings[i])).value.item()}\n\n"
            )
    return {
        "code_precision": code_prec,
        "code_recall": code_rec,
        "code_f1": code_f1,
        "data_precision": data_prec,
        "data_recall": data_rec,
        "data_f1": data_f1
    }


if __name__ == '__main__':
    with pyghidra.open_program('/home/zhaoqi.xiao/Projects/Loadstar/Dataset/NS_1/bins/108.58.252.74.PRG', language='ARM:LE:32:Cortex') as flat_api:
        blocks, embeddings, CodeBlock = train(flat_api)
        evaluate(blocks, embeddings, CodeBlock, "/home/zhaoqi.xiao/Projects/Loadstar/Dataset/NS_1/labeled/108.58.252.74.txt")