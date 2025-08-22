import math
from ltn_helper import *
from pathlib import Path
from datetime import datetime
from functools import reduce
import argparse
import shutil
import os

def binary_input(binary_folder):
    # global binaries, gt
    binary_folder = Path(binary_folder)
    files = [f for f in binary_folder.iterdir() if f.is_file()]
    files = sorted(files, key=lambda x: x.stat().st_size)

    binaries = files
    # binaries = [Path("/app/Loadstar/Dataset/NS_1/bins/5.185.84.3.PRG")]
    gt = [Path(p).with_name(Path(p).name).with_suffix('.txt').as_posix().replace("/bins/", "/fixed_labeled/") for p in binaries]

    return binaries, gt

results = []

geomean = lambda x: math.exp(sum(map(math.log, x)) / len(x))

def redisasemble(
        CodeBlock: ltn.Predicate, 
        my_programs: dict[Path, MyProgram]
    ) -> bool:
    """
    Sample: Re-disassemble the blocks using the trained CodeBlock model.
    """
    flg = True

    for prg_file, my_program in my_programs.items():
        
            # logger.info(f"Deleted existing file {prg_file}.")
        with pyghidra.open_program(prg_file, language='ARM:LE:32:v4') as flat_api:
            for block, emb in zip(my_program.blocks, my_program.embeddings):
                if CodeBlock(ltn.Constant(emb)).value >= 0.5 and block.type == "Data" and not block.failed_disasm_flg:
                    flat_api.clearListing(block.start_address, block.end_address)
                    flat_api.disassemble(block.start_address)
                    flg = False
                    logger.info(f"Re-disassembled block {block.start_address} in {prg_file.name}")

    return flg

def main(binaries, gt):
    # Check and delete any "{binary}_ghidra" folders if they exist
    for prg_file in binaries:
        ghidra_folder = f"{prg_file}_ghidra"
        if Path(ghidra_folder).exists() and Path(ghidra_folder).is_dir():
            shutil.rmtree(ghidra_folder)
            logger.info(f"Deleted existing folder {ghidra_folder}")

    whole_start_time = datetime.now()
    code_f1s, data_f1s = [], []
    my_programs: dict[Path, MyProgram] = {}

    process_times: list[tuple[str, float]] = []

    # Load Ghidra results
    for prg_file in binaries:
        start_time = datetime.now()
        with pyghidra.open_program(prg_file, language='ARM:LE:32:v4') as flat_api:
            my_program = MyProgram(flat_api)
        process_times.append((prg_file.name, (datetime.now() - start_time).total_seconds()))
        logger.info(f"Program {prg_file.name} preprocessed in {process_times[-1][1]:.2f}s")
        my_programs[prg_file] = my_program
    logger.info(f"All programs preprocessed in {sum(t[1] for t in process_times):.2f}s, total blocks: {sum(len(prg.blocks) for prg in my_programs.values())}")

    logger.info("Evaluating plain Ghidra results (no training)")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Evaluate plain Ghidra results
    for prg_file, gt_file in zip(my_programs, gt):
        try:
            results = evaluate(my_programs[prg_file], gt_file, Path(f"debug/{timestamp}/{prg_file.name}"), True)
            code_f1s.append(results["code_f1"])
            data_f1s.append(results["data_f1"])
        except Exception as e:
            logger.info(f"Error evaluating {gt_file}: {e}")
    
    with open(f"debug/{timestamp}/batch_results.txt", "w") as f:
        f.write(f"Binaries: {binaries}\n")
        f.write(f"Code F1 scores: {code_f1s}\n")
        f.write(f"Data F1 scores: {data_f1s}\n")
    
    if len(code_f1s) > 0:
        logger.info(f"F1 Score Geomean: {geomean(code_f1s):.5f} for code, {geomean(data_f1s):.5f} for data")

    return {
        "bin_name": prg_file.name if 'prg_file' in locals() else "batch",
        "code_f1": code_f1s,
        "data_f1": data_f1s,
        "total_time": (datetime.now() - whole_start_time).total_seconds()
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch training and evaluation script.")
    parser.add_argument('--single_training', type=str)
    parser.add_argument('--batch_training', type=str)
    parser.add_argument('--binary_folder', type=str, help='Path to the binary folder')
    parser.add_argument('--test_binary', type=str, help='Path to the test binary')
    args = parser.parse_args()

    if args.test_binary:
        bin_file = Path(args.test_binary)
        gt = [Path(p).with_name(Path(p).name).with_suffix('.txt').as_posix().replace("/bins/", "/fixed_labeled/") for p in [bin_file]]
        if "ghidra" in bin_file.name or not os.path.exists(gt[0]):
            logger.error(f"Test binary {bin_file} or its ground truth {gt[0]} does not exist.")
        else:
            result = main([bin_file], gt)
            print(f"{result}\n")
            

    dataset = "ns_1" if "NS_1" in args.binary_folder else "ns_3" if "NS_3" in args.binary_folder else "ns_2"
    result_file = open(f"{dataset}_results.txt", "w") 
    if args.batch_training:
        binaries, gt = binary_input(args.binary_folder)
        result = main(binaries, gt)
        result_file.write(f"{result}\n")
        result_file.close()
    elif args.single_training:
        result_list = []
        for bin in os.listdir(args.binary_folder):
            bin_name = Path(bin).stem
            # if "ton_ld" not in bin_name:
            #     continue
            bin_file = Path(f"{args.binary_folder}/{bin}")
            gt = [Path(p).with_name(Path(p).name).with_suffix('.txt').as_posix().replace("/bins/", "/fixed_labeled/") for p in [bin_file]]
            if "ghidra" in bin or not os.path.exists(gt[0]):
                continue
        
            try:
                result = main([bin_file], gt)
                print(result)
                result_list.append(result)
            except Exception as e:
                logger.error(f"Error processing {bin_file}: {e}")

        for i in result_list:
            result_file.write(f"{i}\n")
        result_file.close()
