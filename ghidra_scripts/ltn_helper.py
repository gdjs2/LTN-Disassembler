from my_program_helper import *
from my_models import MLPClassifier
from ltn import fuzzy_ops
from pathlib import Path
from datetime import datetime

def train(
        my_program: MyProgram,
        CodeBlock: ltn.Predicate|None = None, 
        epochs: int = 1000
    ) -> tuple[ltn.Predicate, float]:
    """
    Single iteration of training
    Args:
        my_program (MyProgram): The program containing the embeddings and blocks.
        CodeBlock (ltn.Predicate | None): Optional predicate for code blocks, if None, a new one will be created.
        epochs (int): Number of training epochs.
        progress (Progress | None): Optional progress bar for training, if None, no progress bar will be shown.
    Returns:
        tuple[ltn.Predicate, float]: The trained program, the CodeBlock predicate, and the final loss value.
    """
    if not CodeBlock: 
        CodeBlock = ltn.Predicate(MLPClassifier(input_dim=my_program.embeddings.size(1), hidden_dim1=32, hidden_dim2=64).to(ltn.device))

    SatAgg = fuzzy_ops.SatAgg(fuzzy_ops.AggregPMeanError(p=4))
    Forall = ltn.Quantifier(fuzzy_ops.AggregPMeanError(p=4), quantifier='f')

    Equiv = ltn.Connective(fuzzy_ops.Equiv(fuzzy_ops.AndProd(), fuzzy_ops.ImpliesReichenbach()))
    Not = ltn.Connective(fuzzy_ops.NotStandard())

    x_call, y_call= my_program.get_rel_vars("call")
    x_ft, y_ft = my_program.get_rel_vars("fallthrough")

    cond_brch_t = my_program.get_identity_vars("cond_branch_flg", True)
    cond_brch_f = my_program.get_identity_vars("cond_branch_flg", False)
    high_zero_rate = my_program.get_identity_vars("high_zero_rate_flg", True)
    high_cont_printable_char_rate = my_program.get_identity_vars("high_cont_printable_char_rate_flg", True)
    failed_disasm = my_program.get_identity_vars("failed_disasm_flg", True)

    disasm_gt_cb = my_program.get_identity_vars("type", "Code")
    disasm_gt_db = my_program.get_identity_vars("type", "Data")

    optimizer = torch.optim.Adam(list(CodeBlock.parameters()), lr=0.001)

    start = datetime.now()

    for epoch in range(epochs):
        
        if x_ft and y_ft:
            ltn.diag(x_ft, y_ft)
        if x_call and y_call:
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
        if high_cont_printable_char_rate:
            sat_agg_list.append(Forall([high_cont_printable_char_rate], Not(CodeBlock(high_cont_printable_char_rate))))
        if failed_disasm:
            sat_agg_list.append(Forall([failed_disasm], Not(CodeBlock(failed_disasm))))

        sat_agg = SatAgg(*sat_agg_list)

        loss = 1. - sat_agg
        loss.backward()
        optimizer.step()
        # if epoch % 100 == 0:
        #     logger.info(f"Epoch {epoch}, Loss: {loss.item():.5f}")
        if loss.item() < 0.01:
            logger.info(f"Early stopping at epoch {epoch}, Loss: {loss.item():.5f}")
            break

    logger.info(f"Training completed in {(datetime.now() - start).total_seconds():.2f}s, final loss: {loss.item():.5f}")

    return (CodeBlock, loss.item())

def evaluate(
        my_program: MyProgram,
        CodeBlock: ltn.Predicate,
        loss: float,
        label_file,
        result_path: Path,
        debug_flg: bool = False
    ) -> dict:
    start = datetime.now()

    with open(label_file, "r") as f:
        lines = f.readlines()
    lines = [line.strip()[-1] for line in lines]

    tp_code = tp_data = fp_code = fp_data = fn_code = fn_data = cnt = 0

    if not result_path.exists():
        result_path.mkdir(parents=True, exist_ok=True)

    f = open(result_path / "debug.txt", "w") if debug_flg else None

    for i, block in enumerate(my_program.blocks):
        block_opt_flg = True
        code_flg = 0
        if CodeBlock(ltn.Constant(my_program.embeddings[i])).value < loss:
            code_flg = 1

        addr = block.start_address
        while addr <= block.end_address:
            if cnt // 4 >= len(lines):
                break
            if code_flg == int(lines[cnt // 4]):
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
                if f:
                    f.write(f"{addr} expected: {lines[cnt // 4]}, predicted: {code_flg}\n")
                block_opt_flg = False
            addr = addr.add(1)
            cnt += 1
        if not block_opt_flg and f:
            f.write("\n")

    if f:
        f.close()

    code_prec = tp_code / (tp_code + fp_code) if (tp_code + fp_code) > 0 else 0.0
    code_rec  = tp_code / (tp_code + fn_code) if (tp_code + fn_code) > 0 else 0.0
    code_f1   = 2 * code_prec * code_rec / (code_prec + code_rec) if (code_prec + code_rec) > 0 else 0.0

    data_prec = tp_data / (tp_data + fp_data) if (tp_data + fp_data) > 0 else 0.0
    data_rec  = tp_data / (tp_data + fn_data) if (tp_data + fn_data) > 0 else 0.0
    data_f1   = 2 * data_prec * data_rec / (data_prec + data_rec) if (data_prec + data_rec) > 0 else 0.0

    if debug_flg:
        with open(result_path / "result.txt", "w") as f:
            for i, block in enumerate(my_program.blocks):
                f.write(
                    f"{repr(block)}\n"
                    f"  Embedding : {my_program.embeddings[i]}\n"
                    f"  CodeBlock : {CodeBlock(ltn.Constant(my_program.embeddings[i])).value.item()}\n\n"
                )

    elapsed = (datetime.now() - start).total_seconds()
    logger.info(
        f"Code  P/R/F1: {code_prec:.5f}/{code_rec:.5f}/{code_f1:.5f} (tp:{tp_code}) | "
        f"Data  P/R/F1: {data_prec:.5f}/{data_rec:.5f}/{data_f1:.5f} (tp:{tp_data}) | "
        f"Time: {elapsed:.2f}s"
    )

    return {
        "code_precision": code_prec,
        "code_recall": code_rec,
        "code_f1": code_f1,
        "data_precision": data_prec,
        "data_recall": data_rec,
        "data_f1": data_f1
    }

# You can train single binary using this script
if __name__ == '__main__':
    # Command line argument to enable debug mode
    debug_flg = (argv[1] == "debug")
    # Set your binary file path here
    with pyghidra.open_program('/home/zhaoqi.xiao/Projects/Loadstar/Dataset/NS_3/bins/xor_st.app', language='ARM:LE:32:v4') as flat_api:

        time = datetime.now()
        my_program = MyProgram(flat_api)
        logger.info(f"Program preprocessed in {(datetime.now() - time).total_seconds():.2f}s")
        
        CodeBlock, loss = train(my_program, None, 1000)
        time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # gt file and debug directory should be set here
        evaluate(
            my_program, 
            CodeBlock, 
            0.5, 
            "/home/zhaoqi.xiao/Projects/Loadstar/Dataset/NS_3/labeled/ton_ld.txt", 
            Path(f"./debug/{time_stamp}/ton_ld.app"), 
            debug_flg
        )
        if debug_flg:
            torch.save(CodeBlock.state_dict(), Path(f"./debug/{time_stamp}/CodeBlock.pt"))