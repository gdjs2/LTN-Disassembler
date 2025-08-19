import torch
import pyghidra
from datetime import datetime

if not pyghidra.started():
    pyghidra.start()

from sys import argv
from loguru import logger
from typing import Literal, Self
from ghidra.program.model.address import Address
from ghidra.program.model.pcode import PcodeOp
from ghidra.program.model.listing import Instruction, Listing, Program
from ghidra.program.model.scalar import Scalar
from ghidra.app.util import PseudoDisassembler, PseudoDisassemblerContext, PseudoInstruction
from ghidra.program.model.mem import Memory
from ghidra.program.model.symbol import ReferenceManager
from java.math import BigInteger

COMPARISON_OPCODES = [
    PcodeOp.INT_EQUAL,
    PcodeOp.INT_NOTEQUAL,
    PcodeOp.INT_LESS,
    PcodeOp.INT_LESSEQUAL,
    PcodeOp.INT_SLESS,
    PcodeOp.INT_SLESSEQUAL,
]

ARITHMETIC_OPCODES = [
    PcodeOp.INT_ADD, 
    PcodeOp.INT_SUB, 
    PcodeOp.INT_MULT, 
    PcodeOp.INT_DIV
]

class Block:
    """
    Represents a block of code or data in the program.
    Attributes:
        start_address (Address): The starting address of the block.
        end_address (Address): The ending address of the block.
        type (str): The type of the block, either "Code" or "Data".
        section_name (str): The name of the section the block belongs to.
        cond_branch_flg (bool|None): Flag indicating if the block contains conditional branches.
        def_use_flg (bool|None): Flag indicating if the block has def-use relationships.
        very_short_flg (bool|None): Flag indicating if the block is very short.
        high_zero_rate_flg (bool|None): Flag indicating if the block has a high rate of zero bytes.
        high_def_use_rate_flg (bool|None): Flag indicating if the block has a high def-use rate.
        high_cont_printable_char_rate_flg (bool|None): Flag indicating if the block has a high rate of continuous printable characters.
        feature_vector (list[float]|None): Feature vector representing various characteristics of the block.
        pseudo_instrs (list[PseudoInstruction]|None): List of pseudo instructions in the block.
    """
    def __init__(
            self: Self, 
            start_address: Address, 
            end_address: Address, 
            type: Literal["Code", "Data"], 
            section_name: str
        ) -> None:
        """
        Initializes a Block instance.
        Args:
            start_address (Address): The starting address of the block.
            end_address (Address): The ending address of the block.
            type (str): The type of the block, either "Code" or "Data".
            section_name (str): The name of the section the block belongs to.
        """
        self.start_address: Address = start_address
        self.end_address: Address = end_address
        self.type: str = type
        self.section_name: str = section_name
        self.cond_branch_flg: bool|None = None
        self.def_use_flg: bool|None = None
        self.very_short_flg: bool|None = None
        self.high_zero_rate_flg: bool|None = None
        self.high_def_use_rate_flg: bool|None = None
        self.high_cont_printable_char_rate_flg: bool|None = None
        self.failed_disasm_flg: bool|None = None

        self.feature_vector: list[float]|None = None
        self.pseudo_instrs: list[PseudoInstruction]|None = None

    def __repr__(self: Self) -> str:
        return (
            f"{self.type}Block:\n"
            f"  Address Range : [{self.start_address} - {self.end_address}] (l{self.start_address.getOffset()//4 + 1} - l{self.end_address.getOffset()//4 + 1})\n"
            f"  Section       : [{self.section_name}]\n"
            f"  Flags:\n"
            f"    cond_branch        : {self.cond_branch_flg}\n"
            f"    def_use            : {self.def_use_flg}\n"
            f"    very_short         : {self.very_short_flg}\n"
            f"    high_zero_rate     : {self.high_zero_rate_flg}\n"
            f"    high_def_use_rate  : {self.high_def_use_rate_flg}\n"
            f"    high_printable_char: {self.high_cont_printable_char_rate_flg}\n"
            f"    failed_disasm      : {self.failed_disasm_flg}\n"
            f"  Feature Vector: {self.feature_vector}"
        )

    def __str__(self: Self) -> str:
        return (
            f"{self.type}Block @ [{self.start_address} - {self.end_address}]"
        )

def extract_all_blocks(listing: Listing, memory: Memory) -> list[Block]:
    """
    Extract all code and data blocks from the program's listing and memory.
    Args:
        listing (Listing): The program's listing containing code units.
        memory (Memory): The program's memory containing blocks.
    Returns:
        list[Block]: A combined list of code and data Block instances.
    """
    blocks: list[Block] = []

    for mmry_blk in memory.getBlocks():
        addr = mmry_blk.getStart()
        blk_end_addr = mmry_blk.getEnd()

        in_code_block = False
        code_start = None

        in_data_block = False
        data_start = None

        while addr <= blk_end_addr:
            code_unit = listing.getCodeUnitAt(addr)

            if isinstance(code_unit, Instruction):
                code_unit: Instruction = code_unit
                # Close any ongoing data block
                if in_data_block:
                    block_end = code_unit.getMinAddress().subtract(1)
                    blocks.append(Block(data_start, block_end, "Data", mmry_blk.getName()))
                    in_data_block = False
                    data_start = None

                if not in_code_block:
                    code_start = addr
                    in_code_block = True

                flow_type = code_unit.getFlowType()

                if flow_type is not None and (flow_type.isCall() or flow_type.isJump() or flow_type.isTerminal()):
                    block_end = code_unit.getMaxAddress()
                    blocks.append(Block(code_start, block_end, "Code", mmry_blk.getName()))
                    in_code_block = False
                    code_start = None

            else:
                # Close any ongoing code block
                if in_code_block:
                    block_end = code_unit.getMinAddress().subtract(1)
                    blocks.append(Block(code_start, block_end, "Code", mmry_blk.getName()))
                    in_code_block = False
                    code_start = None

                if not in_data_block:
                    data_start = addr
                    in_data_block = True

            addr = addr.add(code_unit.getLength())

        # Handle block at the very end
        if in_code_block:
            blocks.append(Block(code_start, blk_end_addr, "Code", mmry_blk.getName()))
        if in_data_block:
            blocks.append(Block(data_start, blk_end_addr, "Data", mmry_blk.getName()))

    return blocks

def pseudo_disassemble_blocks(blocks: list[Block], program: Program) -> None:
    """
    Pseudo disassemble the blocks using the PseudoDisassembler, default in ARM mode (SEE TODO).
    Args:
        blocks (list[Block]): The list of blocks to pseudo disassemble.
        program (Program): The program containing the blocks.
    Returns:
        None: The blocks will be updated in place with their pseudo instructions.
    """
    pseudo_disassembler = PseudoDisassembler(program)
    for block in blocks:
        ctx = PseudoDisassemblerContext(program.getProgramContext())
        # tmode_reg = program.getRegister("TMode")
        # If you don't care about thumb mode, just comment the next line
        # ctx.setValue(tmode_reg, block.start_address, BigInteger.ZERO)
        ctx.flowStart(block.start_address)

        instrs: list[PseudoInstruction] = []
        addr = block.start_address
        while addr <= block.end_address:
            instr = pseudo_disassembler.disassemble(addr, ctx, False)
            instrs.append(instr)
            if instr is not None:
                addr = instr.getMaxAddress().next()
            else:
                block.failed_disasm_flg = True
                addr = addr.add(4) # TODO: double check here 
        if block.failed_disasm_flg is None:
            block.failed_disasm_flg = False
        block.pseudo_instrs = instrs

def split_data_blocks(blocks: list[Block]) -> list[Block]:
    splited_blocks = []
    for block in blocks:
        if block.type != "Data" or block.pseudo_instrs is None:
            splited_blocks.append(block)
            continue
        last_instr_idx = 0
        last_instr_address = block.start_address
        for idx, instr in enumerate(block.pseudo_instrs):
            if instr is None: continue
            flow_type = instr.getFlowType()

            if flow_type is not None and (flow_type.isCall() or flow_type.isJump() or flow_type.isTerminal()):
                # logger.debug(f"Split {block} @ {instr.getMaxAddress()} by {instr}, flow type: {instr.getFlowType()}")
                new_block = Block(last_instr_address, instr.getMaxAddress(), "Data", block.section_name)
                new_block.pseudo_instrs = block.pseudo_instrs[last_instr_idx:idx + 1]
                new_block.failed_disasm_flg = None in new_block.pseudo_instrs
                splited_blocks.append(new_block)
                last_instr_idx = idx + 1
                last_instr_address = instr.getMaxAddress().add(1)
        # Append any remaining instructions as a new block
        if last_instr_idx < len(block.pseudo_instrs):
            new_block = Block(last_instr_address, block.end_address, "Data", block.section_name)
            new_block.pseudo_instrs = block.pseudo_instrs[last_instr_idx:]
            splited_blocks.append(new_block)

    return splited_blocks

def get_string_number(block: Block, refs: ReferenceManager, listing: Listing) -> int:
    """
    Get the string number of the block.
    Args:
        block (Block): The block to analyze.
        refs (ReferenceManager): The reference manager to get references from.
        listing (Listing): The program's listing to get data from.
    Returns:
        int: The number of strings in the block.
    """
    string_number = 0
    if not block.pseudo_instrs: 
        return string_number
    for instr in block.pseudo_instrs:
        if instr is None: continue
        addr = instr.getAddress()
        references = refs.getReferencesFrom(addr)
        for ref in references:
            to_addr = ref.getToAddress()
            data = listing.getDataAt(to_addr)
            if data and data.hasStringValue():
                string_number += 1
    return string_number

def get_num_constant(block: Block) -> int:
    """
    Get the number of constant values in the block.
    Args:
        block (Block): The block to analyze.
    Returns:
        int: The number of constant values in the block.
    """
    constant_count = 0
    if not block.pseudo_instrs:
        return constant_count
    for instr in block.pseudo_instrs:
        if instr is None: continue
        for i in range(instr.getNumOperands()):
            objs = instr.getOpObjects(i)
            for obj in objs:
                if isinstance(obj, Scalar) or isinstance(obj, Address):
                    constant_count += 1
    return constant_count

def get_transfer_number(block: Block) -> int:
    """
    Get the number of transfer instructions in the block.
    Args:
        block (Block): The block to analyze.
    Returns:
        int: The number of transfer instructions in the block.
    """
    transfer_count = 0
    if not block.pseudo_instrs:
        return transfer_count
    for instr in block.pseudo_instrs:
        if instr is None: continue
        if instr.getFlowType().isCall() or instr.getFlowType().isJump() or instr.getFlowType().isTerminal():
            transfer_count += 1
    return transfer_count

def get_call_number(block: Block) -> int:
    """
    Get the number of call instructions in the block.
    Args:
        block (Block): The block to analyze.
    Returns:
        int: The number of call instructions in the block.
    """
    call_count = 0
    if not block.pseudo_instrs:
        return call_count
    for instr in block.pseudo_instrs:
        if instr is None: continue
        if instr.getFlowType().isCall():
            call_count += 1
    return call_count

def get_instr_number(block: Block) -> int:
    """
    Get the number of instructions in the block.
    Args:
        block (Block): The block to analyze.
    Returns:
        int: The number of instructions in the block.
    """
    return sum(i is not None for i in block.pseudo_instrs) if block.pseudo_instrs else 0

def get_arithmetic_number(block: Block) -> int:
    """
    Get the number of arithmetic instructions in the block.
    """
    arithmetic_count = 0
    if not block.pseudo_instrs:
        return arithmetic_count
    for instr in block.pseudo_instrs:
        if instr is None: continue
        pcode_ops = instr.getPcode()
        for op in pcode_ops:
            if op.getOpcode() in ARITHMETIC_OPCODES:
                arithmetic_count += 1
    return arithmetic_count

def get_zero_bytes_number(block: Block, memory: Memory) -> int:
    """
    Get the number of zero bytes in the block. This function will also set the `high_zero_rate_flg` attribute of the block.
    Args:
        block (Block): The block to analyze.
        memory (Memory): The program's memory to read bytes from.
    Returns:
        int: The number of zero bytes in the block.
    """
    zero_bytes_cnt = 0
    addr = block.start_address
    while addr <= block.end_address:
        try:
            data = memory.getByte(addr) & 0xFF
            if data == 0:
                zero_bytes_cnt += 1
        except:
            pass
        addr = addr.add(1)
    if zero_bytes_cnt * 2 >= block.end_address.subtract(block.start_address):
        block.high_zero_rate_flg = True
    else:
        block.high_zero_rate_flg = False
    return zero_bytes_cnt

def get_def_use_number(block: Block) -> int:
    """
    Get the number of def-use relationships in the block. This function will also set the `high_def_use_rate_flg` attribute of the block.
    Args:
        block (Block): The block to analyze.
        memory (Memory): The program's memory to read bytes from.
    Returns:
        int: The number of def-use relationships in the block.
    """
    def_use_cnt = 0
    defs = {}
    if block.pseudo_instrs is None:
        block.high_def_use_rate_flg = False
        return def_use_cnt
    for i, instr in enumerate(block.pseudo_instrs):
        for d in defs:
            if defs[d] - i > 16:
                del defs[d]
        if instr is None: continue
        pcode_ops = instr.getPcode()
        instr_def = {}
        for op in pcode_ops:
            uses = op.getInputs()
            for use in uses:
                if use in defs:
                    def_use_cnt += 1
            instr_def[op.getOutput()] = i
        defs.update(instr_def)
    
    if def_use_cnt * 3 >= block.end_address.subtract(block.start_address):
    # if def_use_cnt >= 1:
        block.high_def_use_rate_flg = True
    else:
        block.high_def_use_rate_flg = False
    return def_use_cnt

def get_printable_char_number(block: Block, memory: Memory) -> int:
    """
    Get the number of printable characters in the block. This function will also set the `high_cont_printable_char_rate_flg` attribute of the block.
    Args:
        block (Block): The block to analyze.
        memory (Memory): The program's memory to read bytes from.
    Returns:
        int: The number of printable characters in the block.
    """
    printable_count = 0
    continous_printable_count = 0
    addr = block.start_address
    while addr <= block.end_address:
        try:
            data = memory.getByte(addr) & 0xFF
            if data >= 32 and data <= 126:  # ASCII printable range
                printable_count += 1
                continous_printable_count += 1
            else:
                continous_printable_count = 0
        except: # Reading memory may fail if the address is not valid
            pass
        addr = addr.add(1)
    if continous_printable_count * 2 >= block.end_address.subtract(block.start_address):
        block.high_cont_printable_char_rate_flg = True
    else:
        block.high_cont_printable_char_rate_flg = False
    return printable_count
    
def get_feature_vector(
        blocks: list[Block],
        refs: ReferenceManager, 
        listing: Listing, 
        memory: Memory
    ) -> None:
    """
    Get the feature vector for each block.
    """
    for block in blocks:
        block_size = block.end_address.subtract(block.start_address)
        feature_vec = [
            get_string_number(block, refs, listing) / block_size,
            get_num_constant(block) / block_size,
            get_transfer_number(block) / block_size,
            get_call_number(block) / block_size,
            get_instr_number(block) / block_size,
            get_arithmetic_number(block) / block_size,
            get_zero_bytes_number(block, memory) / block_size,
            get_def_use_number(block) / block_size,
            get_printable_char_number(block, memory) / block_size,
        ]
        block.feature_vector = feature_vec

def check_compare_branch(blocks: list[Block], program: Program) -> None:
    """
    Check conditional branches in the blocks following a comparison instructions. 
    This function will set the `cond_branch_flg` attribute of the block.
    Args:
        blocks (list[Block]): The list of blocks to analyze.
        program (Program): The program containing the blocks.
    Returns:
        None: The blocks will be updated in place with their conditional branch flags.
    """
    for block in blocks:
        if block.pseudo_instrs is None: 
            block.cond_branch_flg = None
            continue
        instrs = block.pseudo_instrs[::-1]  # Reverse the order to check from the end
        first_instr = instrs[0]
        if first_instr is None:
            block.cond_branch_flg = None
        elif first_instr.getFlowType().isConditional():
            # Check if the second instruction is a comparison
            detect_comp_flg = False
            for instr in instrs[1:]:
                if instr is None: continue
                pcode_ops = instr.getPcode()
                if any(op.getOpcode() in COMPARISON_OPCODES for op in pcode_ops):
                    detect_comp_flg = True
                    break
            block.cond_branch_flg = detect_comp_flg
    return

# def check_def_use(blocks, psuedo_disassembler: PseudoDisassembler):
#     for block in blocks:
#         if block.type == "Code": continue
#         instrs = block.pseudo_instrs
#         defs = {}
#         for i, instr in enumerate(instrs):
#             pcode_ops = instr.getPcode()
#             instr_def = []
#             for op in pcode_ops:
#                 if op.getOpcode() == PcodeOp.STORE:
#                     uses = op.getInputs()
#                     for use in uses:
#                         if use in defs and i - defs[use] <= 16:
#                             block.def_use_flg = True
#                             break
#                 if op.getOpcode() in [PcodeOp.COPY, PcodeOp.LOAD]:
#                     instr_def.append(op.getOutput())
#             for d in instr_def:
#                 defs[d] = i
#     return

def check_very_short(blocks: list[Block]) -> None:
    """
    Check if the block is very short, i.e., less than 3 instruction (12 bytes).
    This function will set the `very_short_flg` attribute of the block.
    Args:
        blocks (list[Block]): The list of blocks to analyze.
    Returns:
        None: The blocks will be updated in place with their very short flags.
    """
    for block in blocks:
        if block.end_address.subtract(block.start_address) <= 12:
            block.very_short_flg = True
            block.type = "Code" if block.type == "Data" else "Data"
        else:
            block.very_short_flg = False
    return

def generate_embeddings_from_feature_vector(blocks: list[Block]) -> torch.Tensor:
    """
    Generate embeddings from the feature vector of the blocks.
    Args:
        blocks (list[Block]): The list of blocks to generate embeddings from.
    Returns:
        torch.Tensor: A tensor containing the embeddings for each block.
    """
    embeddings = []
    for block in blocks:
        embeddings.append(torch.tensor(block.feature_vector, dtype=torch.float32))
    return torch.stack(embeddings, dim=0)

def generate_random_embeddings(blocks, dim=16):
    n = len(blocks)
    embeddings = torch.randn(n, dim)
    return embeddings

# def fix_undisassembled_data_blocks(blocks):
#     for block in blocks:
#         if block.pseudo_instrs is None:
#             block.type = "FixedData"

if __name__ == "__main__":
    with pyghidra.open_program('/home/zhaoqi.xiao/Projects/Loadstar/Dataset/NS_1/bins/108.58.252.74.PRG', language='ARM:LE:32:Cortex') as flat_api:
    # with pyghidra.open_program("/home/zhaoqi.xiao/Projects/ghidra-ic/Binaries/xmltest") as flat_api:
        program = flat_api.getCurrentProgram()
        function_manager = program.getFunctionManager()
        listing = program.getListing()
        memory = program.getMemory()

        blocks = extract_all_blocks(listing, memory)
        blocks.sort(key=lambda b: b.start_address)

        pseudo_disassemble_blocks(blocks, program)

        get_feature_vector(blocks, program.getReferenceManager(), listing, memory)
        check_compare_branch(blocks, program)
        # check_very_short(blocks)

        # check_def_use(blocks, PseudoDisassembler(program))
        if argv[1] == "debug":
            with open('blocks.txt', 'w') as f:
                for block in blocks:
                    f.write(f"{repr(block)}\n\n")
        
        logger.info(f"Extracted {len(blocks)} blocks from the program, exported to blocks.txt")
