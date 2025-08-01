# Ignore this file
from my_program_helper import *
from pathlib import Path
from datetime import datetime

if __name__ == '__main__':
    binary_folder = Path("/home/zhaoqi.xiao/Projects/Loadstar/Dataset/NS_1/bins")
    files = [f for f in binary_folder.iterdir() if f.is_file()]
    files = sorted(files, key=lambda x: x.stat().st_size)
    for file in files[12:]:
        start_time = datetime.now()
        logger.info(f"Processing file: {file.name}")
        with pyghidra.open_program(file, language='ARM:LE:32:Cortex') as flat_api:
            my_program = MyProgram(flat_api)
        logger.info(f"Processed {file.name}({file.stat().st_size} Bytes) in {(datetime.now() - start_time).total_seconds():.2f}s")
        