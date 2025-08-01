import math
from ltn_helper import *
from pathlib import Path
from datetime import datetime
from functools import reduce

# Choose the binary folder
binary_folder = Path("/home/zhaoqi.xiao/Projects/Loadstar/Dataset/NS_1/bins")
files = [f for f in binary_folder.iterdir() if f.is_file()]
# files = [Path("/home/zhaoqi.xiao/Projects/Loadstar/Dataset/NS_1/bins/5.226.124.166.PRG")]
files = sorted(files, key=lambda x: x.stat().st_size)

binaries = files
# Ground truth files are expected to be in the same folder with a different suffix
gt = [Path(p).with_name(Path(p).name).with_suffix('.txt').as_posix().replace("/bins/", "/labeled/") for p in binaries]

results = []

geomean = lambda x: math.exp(sum(map(math.log, x)) / len(x))

# This function fixes the ground truth labels for code blocks that failed disassembly.
# It reads the ground truth file, checks each block, and writes the corrected labels to a
# new file.
# If a block is marked as code in the ground truth but failed disassembly, it will
# correct the labels to indicate that it is indeed code.
def fix(
        my_program: MyProgram,
        label_file,
        fixed_file: Path,
    ) -> None:
    """
    Fix the ground truth labels for code blocks that failed disassembly.
    Args:
        my_program (MyProgram): The program instance containing blocks and disassembly info.
        label_file (str): Path to the ground truth label file.
        fixed_file (Path): Path to save the fixed labels.
    """
    with open(label_file, "r") as f:
        lines = f.readlines()
    gt = [line.strip()[-1] for line in lines]

    cnt = 0

    f = open(fixed_file, "w")
    for block in my_program.blocks:
        addr = block.start_address
        start_idx = cnt // 4
        code_block_flg = True
        while addr <= block.end_address:
            if cnt // 4 >= len(gt):
                break
            if int(gt[cnt // 4]) == 1:
                code_block_flg = False
                # break
            addr = addr.add(1)
            cnt += 1
        end_idx = (cnt-1) // 4

        if code_block_flg and block.failed_disasm_flg:
            for i in range(start_idx, end_idx + 1):
                f.write(lines[i].strip()[:-1] + "1\n")
            logger.info(f"Fixed block {block}, which was Code in GT but failed disassembly.")
        else:
            for i in range(start_idx, end_idx + 1):
                f.write(lines[i])
        

    if f:
        f.close()

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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for prg_file, gt_file in zip(my_programs, gt):
        try:
            if not Path(f"fixed_gt/{timestamp}").exists():
                Path(f"fixed_gt/{timestamp}").mkdir(parents=True, exist_ok=True)
            fix(my_programs[prg_file], gt_file, Path(f"fixed_gt/{timestamp}/{prg_file.stem}.txt"))
            logger.info(f"Fix completed for {prg_file}")
        except Exception as e:
            logger.info(f"Error when fixing {prg_file}: {e}")
