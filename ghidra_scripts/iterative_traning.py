# This file is deprecated and will be updated in the future.

import pyghidra
import ltn
import ltn.fuzzy_ops
import torch

from loguru import logger
from my_models import MLPClassifier
from graph_helper import create_graph
from blocks_helper import check_compare_branch, check_very_short, get_feature_vector, pseudo_disassemble_blocks
from ltn_helper import generate_random_embeddings, embeddings_from_feature_vector

if not pyghidra.started():
    pyghidra.start()

from ghidra.program.model.address import AddressSet
from ghidra.app.util import PseudoDisassembler
from ghidra.program.disassemble import Disassembler
from java.math import BigInteger

def redisasemble(CodeBlock, embeddings, blocks, flat_api):
    """
    Sample: Re-disassemble the blocks using the trained CodeBlock model.
    """
    listing = flat_api.getCurrentProgram().getListing()
    memory = flat_api.getCurrentProgram().getMemory()
    program = flat_api.getCurrentProgram()
    flg = False
    
    for block, emb in zip(blocks, embeddings):
        if CodeBlock(ltn.Constant(emb)).value >= 0.5 and block.type == "Data":
            ctx = program.getProgramContext()
            flat_api.clearListing(block.start_address, block.end_address)
            tmode_reg = ctx.getRegister('TMode')
            ctx.setValue(tmode_reg, block.start_address, block.end_address, BigInteger.ZERO)
            flat_api.disassemble(block.start_address)
            flg = True
        elif (CodeBlock(ltn.Constant(emb)).value < 0.5 and block.type == "Code") or block.type == "FixedData":
            flat_api.clearListing(block.start_address, block.end_address)
            flg = True
    return flg
        

if __name__ == '__main__':

    with pyghidra.open_program('/home/zhaoqi.xiao/Projects/Loadstar/Dataset/NS_1/bins/108.58.252.74.PRG', language='ARM:LE:32:Cortex') as flat_api:
        program = flat_api.getCurrentProgram()
        flg = True
        while flg:
            blocks, graph = create_graph(flat_api)

            pseudo_disassemble_blocks(blocks, program)
            check_compare_branch(blocks, PseudoDisassembler(program), program)
            check_def_use(blocks, PseudoDisassembler(program))
            check_very_short(blocks)
            get_feature_vector(blocks, PseudoDisassembler(program), program.getReferenceManager(), program.getListing(), program.getMemory())
            fix_undisassembled_data_blocks(blocks)

            # embeddings = generate_embeddings(blocks, flat_api)
            embeddings = generate_random_embeddings(blocks, dim=16)
            # embeddings = embeddings_from_feature_vector(blocks)
            with open('blocks.txt', 'w') as f:
                for i, block in enumerate(blocks):
                    f.write(f"{block} {embeddings[i].tolist()}\n")
        
            block2idx = {block: i for i, block in enumerate(blocks)}
            idx2block = {i: block for i, block in enumerate(blocks)}

            CodeBlock = ltn.Predicate(MLPClassifier(input_dim=16, hidden_dim1=32, hidden_dim2=32).to(ltn.device))
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
            epochs = 2000

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
                    Forall([high_def_use_rate], CodeBlock(high_def_use_rate)),
                    # Forall([very_short], )

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
                    if CodeBlock(ltn.Constant(embeddings[i])).value > 0.50 and block.type != "FixedData":
                        code_flg = 0

                    addr = block.start_address
                    while addr <= block.end_address:
                        if cnt // 4 >= len(lines):
                            # logger.error(f"Not enough labels in lines for address {addr}.")
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

            flg = redisasemble(CodeBlock, embeddings, blocks, flat_api)
