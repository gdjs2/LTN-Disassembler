import math
from ltn_helper import *
from pathlib import Path
from datetime import datetime
from functools import reduce

binary_folder = Path("/home/zhaoqi.xiao/Projects/Loadstar/Dataset/NS_3/bins")
files = [f for f in binary_folder.iterdir() if f.is_file()]
files = sorted(files, key=lambda x: x.stat().st_size)

binaries = files[:5]
gt = [Path(p).with_name(Path(p).name).with_suffix('.txt').as_posix().replace("/bins/", "/fixed_labeled/") for p in binaries]

results = []

geomean = lambda x: math.exp(sum(map(math.log, x)) / len(x))

if __name__ == "__main__":
    CodeBlock = None
    code_f1s, data_f1s = [], []
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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if CodeBlock is None:
        logger.error("CodeBlock is None, training failed.")
        exit(1)

    for prg_file, gt_file in zip(my_programs, gt):
        try:
            results = evaluate(my_programs[prg_file], CodeBlock, 0.4, gt_file, Path(f"debug/{timestamp}/{prg_file.name}"), True)
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
