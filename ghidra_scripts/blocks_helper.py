import pyghidra

if not pyghidra.started():
    pyghidra.start()

from loguru import logger

from ghidra.program.model.address import AddressRangeImpl, AddressSet
from ghidra.program.model.pcode import PcodeOp
from ghidra.program.model.listing import Instruction, Data

COMPARISON_OPCODES = [
    PcodeOp.INT_EQUAL,
    PcodeOp.INT_NOTEQUAL,
    PcodeOp.INT_LESS,
    PcodeOp.INT_LESSEQUAL,
    PcodeOp.INT_SLESS,
    PcodeOp.INT_SLESSEQUAL,
]


class Block:
    def __init__(self, start_address, end_address, type, section_name):
        self.start_address = start_address
        self.end_address = end_address
        self.type = type
        self.section_name = section_name
        self.cond_branch_flg = None
        self.def_use_flg = None
    
    def __repr__(self):
        return f"{self.type}Block [{self.start_address} - {self.end_address}] [{self.section_name}] [{self.cond_branch_flg}] [{self.def_use_flg}]"

    def __str__(self):
        return self.__repr__()
    
def extract_code_blocks(function_manager, listing, memory):
    blocks = []

    functions = function_manager.getFunctions(True)
    while functions.hasNext():
        function = functions.next()
        block_start = None
        new_block_flag = True

        instructions = listing.getInstructions(function.getBody(), True)
        while instructions.hasNext():
            instruction = instructions.next()
            if new_block_flag:
                block_start = instruction.getAddress()
                new_block_flag = False

            mnemonic = instruction.getMnemonicString().lower()
            if mnemonic == "call" or mnemonic == "ret" or mnemonic.startswith("j"):
                block_end = instruction.getMaxAddress()
                block = Block(block_start, block_end, "Code", memory.getBlock(block_start).getName())
                blocks.append(block)
                new_block_flag = True

        if not new_block_flag:
            block_end = function.getBody().getMaxAddress()
            block = Block(block_start, block_end, "Code", memory.getBlock(block_start).getName())
            blocks.append(block)

    return blocks

def extract_data_blocks(listing, memory):
    blocks = []

    data_list = []
    data_iter = listing.getDefinedData(True)
    while data_iter.hasNext():
        data = data_iter.next()
        data_list.append(data)

    data_list.sort(key=lambda d: d.getAddress())

    current_range = None
    current_block_data = []

    for data in data_list:
        start_address = data.getAddress()
        end_address = start_address.add(data.getLength() - 1)

        if current_range is None:
            current_range = AddressRangeImpl(start_address, end_address)
            current_block_data.append(data)
        else:
            expected_next = current_range.getMaxAddress().next()
            if expected_next is not None and expected_next.equals(start_address) and memory.getBlock(start_address).getName() == memory.getBlock(current_range.getMinAddress()).getName():
                current_range = AddressRangeImpl(current_range.getMinAddress(), end_address)
                current_block_data.append(data)
            else:
                block = Block(current_range.getMinAddress(), current_range.getMaxAddress(), "Data", memory.getBlock(current_range.getMinAddress()).getName())
                blocks.append(block)
                current_range = AddressRangeImpl(start_address, end_address)
                current_block_data = [data]

    return blocks

def extract_unknown_blocks(memory, blocks):
    unknown_blocks = []
    blocks.sort(key=lambda b: b.start_address)

    for i in range(len(blocks) - 1):
        current_block = blocks[i]
        next_block = blocks[i + 1]

        if current_block.section_name == next_block.section_name: 
            if current_block.end_address.add(1) == next_block.start_address:
                continue
            elif current_block.end_address.add(1).compareTo(next_block.start_address) < 0:
                unknown_start = current_block.end_address.add(1)
                unknown_end = next_block.start_address.subtract(1)
                block = Block(unknown_start, unknown_end, "Unknown", memory.getBlock(unknown_start).getName())
                unknown_blocks.append(block)
            else:
                logger.warning(f"Invalid block order: {current_block} -> {next_block}")
        else:
            pass
    
    return unknown_blocks

def check_compare_branch(blocks, listing):
    """
    Check conditional branches in the blocks following a comparison instructions.
    """
    for block in blocks:
        addr_set = AddressSet(block.start_address, block.end_address)
        instr_iter = listing.getInstructions(addr_set, False)
        if instr_iter is None or not instr_iter.hasNext():
            logger.warning(f"Cannot get instructions for block {block}.")
            continue
        first_instr = instr_iter.next()
        if first_instr.getFlowType().isConditional():
            logger.debug(f"Block {block} has a conditional branch.")
            # Check if the second instruction is a comparison
            detect_comp_flg = False
            while instr_iter.hasNext():
                instr = instr_iter.next()
                pcode_ops = instr.getPcode()
                if any(op.getOpcode() in COMPARISON_OPCODES for op in pcode_ops):
                    detect_comp_flg = True
                    break
            block.cond_branch_flg = detect_comp_flg

        else:
            pass
    return 

def check_def_use(blocks, listing):
    for block in blocks:
        if block.type == "Code":
            continue
        addr_set = AddressSet(block.start_address, block.end_address)
        instr_iter = listing.getInstructions(addr_set, True)
        defs = []
        if instr_iter is None or not instr_iter.hasNext():
            logger.warning(f"Cannot get instructions for block {block}.")
            continue
        while instr_iter.hasNext():
            instr = instr_iter.next()
            pcode_ops = instr.getPcode()
            instr_def = []
            for op in pcode_ops:
                uses = op.getInputs()
                for use in uses:
                    if use in defs:
                        block.def_use_flg = True
                        break
                instr_def.append(op.getOutput())
            defs.extend(instr_def)
    return 
        
    

if __name__ == "__main__":
    with pyghidra.open_program('/home/zhaoqi.xiao/Projects/Loadstar/Dataset/NS_1/bins/108.58.252.74.PRG', language='ARM:LE:32:Cortex') as flat_api:
        program = flat_api.getCurrentProgram()
        function_manager = program.getFunctionManager()
        listing = program.getListing()
        memory = program.getMemory()

        for block in memory.getBlocks():
            addr = block.getStart()
            end = block.getEnd()

            while addr <= end:
                code_unit = listing.getCodeUnitAt(addr)
                print(f'{addr}: {code_unit}, IsInstr [{isinstance(code_unit, Instruction)}], IsData [{isinstance(code_unit, Data)}]')
                # if code_unit.isInstruction():
                addr = addr.add(code_unit.getLength())
                # else:
                #     addr = addr.add(1)

        # code_blocks = extract_code_blocks(function_manager, listing, memory)
        # data_blocks = extract_data_blocks(listing, memory)
        # unknown_blocks = extract_unknown_blocks(memory, code_blocks + data_blocks)

        # blocks = [*code_blocks, *data_blocks, *unknown_blocks]

        # check_compare_branch(blocks, listing)
        # check_def_use(blocks, listing)

        # with open('blocks.txt', 'w') as f:
        #     for block in blocks:
        #         f.write(f"{block}\n")