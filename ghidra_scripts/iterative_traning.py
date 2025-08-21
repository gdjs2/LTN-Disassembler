import math
from ltn_helper import *
from pathlib import Path
from datetime import datetime
from functools import reduce
import argparse
import shutil

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
    finish = False
    iteration = 0
    # Check and delete any "{binary}_ghidra" folders if they exist
    for prg_file in binaries:
        ghidra_folder = f"{prg_file}_ghidra"
        if Path(ghidra_folder).exists() and Path(ghidra_folder).is_dir():
            shutil.rmtree(ghidra_folder)
            logger.info(f"Deleted existing folder {ghidra_folder}")

    while not finish and iteration < 10:
        CodeBlock = None
        code_f1s, data_f1s = [], []
        iteration += 1
        my_programs: dict[Path, MyProgram] = {}

        process_times: list[tuple[str, float]] = []

        for prg_file in binaries:
            start_time = datetime.now()
            with pyghidra.open_program(prg_file, language='ARM:LE:32:v4') as flat_api:
                my_program = MyProgram(flat_api)
            process_times.append((prg_file.name, (datetime.now() - start_time).total_seconds()))
            logger.info(f"Program {prg_file.name} preprocessed in {process_times[-1][1]:.2f}s")
            my_programs[prg_file] = my_program
        logger.info(f"All programs preprocessed in {sum(t[1] for t in process_times):.2f}s, total blocks: {sum(len(prg.blocks) for prg in my_programs.values())}")

        # Small epoches is the epochs for training on each program
        # We will train on all binaries for a few epochs, which is large_epochs
        small_epochs = 300
        large_epochs = 3
        logger.info(f"Start training with {len(my_programs)} programs, small epochs: {small_epochs}, large epochs: {large_epochs}")

        for i in range(large_epochs):
            for prg_file in my_programs:
                logger.info(f"Training epoch {i + 1}/{large_epochs} on {prg_file.name}")
                CodeBlock, loss = train(my_programs[prg_file], CodeBlock, small_epochs)

        logger.info("Training finished")

        if CodeBlock is None:
            logger.error("CodeBlock is None, training failed.")
            raise RuntimeError("CodeBlock is None, training failed.")

        finish = redisasemble(CodeBlock, my_programs)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if CodeBlock is None:
            logger.error("CodeBlock is None, training failed.")
            exit(1)

        for prg_file, gt_file in zip(my_programs, gt):
            try:
                results = evaluate(my_programs[prg_file], CodeBlock, 0.5, gt_file, Path(f"debug/{timestamp}/{prg_file.name}"), True)
                code_f1s.append(results["code_f1"])
                data_f1s.append(results["data_f1"])
            except Exception as e:
                logger.info(f"Error evaluating {gt_file}: {e}")
        
        with open(f"debug/{timestamp}/batch_results.txt", "w") as f:
            f.write(f"Binaires: {binaries}\n")
            f.write(f"Code F1 scores: {code_f1s}\n")
            f.write(f"Data F1 scores: {data_f1s}\n")
        logger.info(f"F1 Score Geomean: {geomean(code_f1s):.5f} for code, {geomean(data_f1s):.5f} for data")

        if CodeBlock: torch.save(CodeBlock.state_dict(), f"debug/{timestamp}/CodeBlock.pth")
        logger.info(f"Saved CodeBlock model to debug/{timestamp}/CodeBlock.pth")

        return binaries[0], code_f1s, data_f1s

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch training and evaluation script.")
    parser.add_argument('--single_training', type=str)
    parser.add_argument('--batch_training', type=str)
    parser.add_argument('--binary_folder', type=str, help='Path to the binary folder')
    args = parser.parse_args()

    if args.batch_training:
        binaries, gt = binary_input(args.binary_folder)
        main(binaries, gt)
    elif args.single_training:
        dataset = "ns_1" if "ns_1" in args.binary_folder else "ns_3" if "ns_3" in args.binary_folder else "ns_2"
        result_file = open(f"{dataset}_results.txt", "w") 
        for bin in os.listdir(args.binary_folder):
            bin_name = Path(bin).stem
            # if "ton_ld" not in bin_name:
            #     continue
            bin_file = Path(f"{args.binary_folder}/{bin}")
            gt = [Path(p).with_name(Path(p).name).with_suffix('.txt').as_posix().replace("/bins/", "/fixed_labeled/") for p in [bin_file]]
            if "ghidra" in bin or not os.path.exists(gt[0]):
                continue

            main([bin_file], gt)
