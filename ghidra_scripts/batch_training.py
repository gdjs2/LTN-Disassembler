import os
from ltn_helper import *
from pathlib import Path
from datetime import datetime

binary_folder = Path("/home/zhaoqi.xiao/Projects/Loadstar/Dataset/NS_1/bins")
files = [f for f in binary_folder.iterdir() if f.is_file()]
files = sorted(files, key=lambda x: x.stat().st_size)

binaries = files[:20]

gt = [Path(p).with_name(Path(p).name).with_suffix('.txt').as_posix().replace("/bins/", "/labeled/") for p in binaries]

results = []

if __name__ == "__main__":
    CodeBlock = None
    code_f1s, data_f1s = [], []
    for prg_file in binaries:
        with pyghidra.open_program(prg_file, language='ARM:LE:32:Cortex') as flat_api:
            blocks, embeddings, CodeBlock = train(flat_api, CodeBlock)
            results.append((blocks, embeddings))
    for (blocks, embeddings), gt_file in zip(results, gt):
        try:
            results = evaluate(blocks, embeddings, CodeBlock, gt_file)
            code_f1s.append(results["code_f1"])
            data_f1s.append(results["data_f1"])
        except Exception as e:
            print(f"Error evaluating {gt_file}: {e}")
            continue
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"debug/f1_results_{timestamp}.txt", "w") as f:
        f.write("Code F1 scores: {code_f1s}\n")
        f.write("Data F1 scores: {data_f1s}\n")
