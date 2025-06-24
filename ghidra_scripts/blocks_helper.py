import pyghidra

if not pyghidra.started():
    pyghidra.start()

from loguru import logger
from ghidra.program.model.address import Address
from ghidra.program.model.pcode import PcodeOp
from ghidra.program.model.listing import Instruction, Data
from ghidra.program.model.scalar import Scalar
from ghidra.app.util import PseudoDisassembler, PseudoDisassemblerContext
from java.math import BigInteger

COMPARISON_OPCODES = [
    PcodeOp.INT_EQUAL,
    PcodeOp.INT_NOTEQUAL,
    PcodeOp.INT_LESS,
    PcodeOp.INT_LESSEQUAL,
    PcodeOp.INT_SLESS,
    PcodeOp.INT_SLESSEQUAL,
]

ARITHMETIC_OPCODES = [PcodeOp.INT_ADD, PcodeOp.INT_SUB, PcodeOp.INT_MULT, PcodeOp.INT_DIV]

class Block:
    def __init__(self, start_address, end_address, type, section_name):
        self.start_address = start_address
        self.end_address = end_address
        self.type = type
        self.section_name = section_name
        self.cond_branch_flg = None
        self.def_use_flg = None
        self.very_short_flg = None
        self.high_zero_rate_flg = None
        self.high_def_use_rate_flg = None

        self.feature_vector = None
        self.pseudo_instrs = None

    def __repr__(self):
        return (
            f"{self.type}Block:\n"
            f"  Address Range : [{self.start_address} - {self.end_address}]\n"
            f"  Section       : [{self.section_name}]\n"
            f"  Flags:\n"
            f"    cond_branch        : {self.cond_branch_flg}\n"
            f"    def_use            : {self.def_use_flg}\n"
            f"    very_short         : {self.very_short_flg}\n"
            f"    high_zero_rate     : {self.high_zero_rate_flg}\n"
            f"    high_def_use_rate  : {self.high_def_use_rate_flg}\n"
            f"  Feature Vector: {self.feature_vector}"
        )


    def __str__(self):
        return (
            f"{self.type}Block @ [{self.start_address} - {self.end_address}]"
        )

def extract_code_blocks(listing, memory):
    code_blocks = [] 

    for mmry_blk in memory.getBlocks():
        addr = mmry_blk.getStart()
        blk_end_addr = mmry_blk.getEnd()
        new_blk = True
        blk_start_addr = None

        while addr <= blk_end_addr:
            code_unit = listing.getCodeUnitAt(addr)
            if isinstance(code_unit, Instruction):
                if new_blk:
                    blk_start_addr = addr
                    new_blk = False
                mnemonic = code_unit.getMnemonicString().lower()
                if mnemonic == "call" or mnemonic == "ret" or mnemonic.startswith("j") or mnemonic.startswith("b"):
                    block_end_addr = code_unit.getMaxAddress()
                    block = Block(blk_start_addr, block_end_addr, "Code", memory.getBlock(blk_start_addr).getName())
                    code_blocks.append(block)
                    new_blk = True
                    blk_start_addr = None
            elif blk_start_addr is not None:
                block_end_addr = code_unit.getMinAddress().subtract(1)
                block = Block(blk_start_addr, block_end_addr, "Code", memory.getBlock(blk_start_addr).getName())
                code_blocks.append(block)
                new_blk = True
                blk_start_addr = None
            addr = addr.add(code_unit.getLength())

        if blk_start_addr is not None:
            block_end = blk_end_addr
            block = Block(blk_start_addr, block_end, "Code", memory.getBlock(blk_start_addr).getName())
            code_blocks.append(block)
            blk_start_addr = None

    return code_blocks

def extract_data_blocks(listing, memory):
    data_blocks = []

    for mmry_blk in memory.getBlocks():
        addr = mmry_blk.getStart()
        blk_end_addr = mmry_blk.getEnd()
        new_blk = True
        blk_start_addr = None

        while addr <= blk_end_addr:
            code_unit = listing.getCodeUnitAt(addr)
            if isinstance(code_unit, Data):
                if new_blk:
                    blk_start_addr = addr
                    new_blk = False
            elif blk_start_addr is not None:
                block_end_addr = code_unit.getMinAddress().subtract(1)
                block = Block(blk_start_addr, block_end_addr, "Data", memory.getBlock(blk_start_addr).getName())
                data_blocks.append(block)
                new_blk = True
                blk_start_addr = None
            addr = addr.add(code_unit.getLength())
        if blk_start_addr is not None:
            block_end = blk_end_addr
            block = Block(blk_start_addr, block_end, "Data", memory.getBlock(blk_start_addr).getName())
            data_blocks.append(block)
            blk_start_addr = None
    return data_blocks

def pseudo_disassemble_blocks(blocks, program):
    """
    Pseudo disassemble the blocks using the PseudoDisassembler, default in ARM mode (SEE TODO).
    """
    pseudo_disassembler = PseudoDisassembler(program)
    for block in blocks:
        ctx = PseudoDisassemblerContext(program.getProgramContext())
        tmode_reg = program.getRegister("TMode")
        ctx.setValue(tmode_reg, block.start_address, BigInteger.ZERO)
        ctx.flowStart(block.start_address)

        instrs = []
        addr = block.start_address
        while addr <= block.end_address:
            instr = pseudo_disassembler.disassemble(addr, ctx, False)
            instrs.append(instr)
            if instr is not None:
                addr = instr.getMaxAddress().next()
            else:
                addr = addr.add(4) # TODO: double check here
        block.pseudo_instrs = instrs

def get_string_number(block, refs, listing):
    """
    Get the string number of the block.
    """
    string_number = 0
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

def get_num_constant(block):
    """
    Get the number of constant values in the block.
    """
    constant_count = 0
    for instr in block.pseudo_instrs:
        if instr is None: continue
        for i in range(instr.getNumOperands()):
            objs = instr.getOpObjects(i)
            for obj in objs:
                if isinstance(obj, Scalar) or isinstance(obj, Address):
                    constant_count += 1
    return constant_count

def get_transfer_number(block):
    """
    Get the number of transfer instructions in the block.
    """
    transfer_count = 0
    for instr in block.pseudo_instrs:
        if instr is None: continue
        if instr.getFlowType().isCall() or instr.getFlowType().isJump() or instr.getFlowType().isTerminal():
            transfer_count += 1
    return transfer_count

def get_call_number(block):
    """
    Get the number of call instructions in the block.
    """
    call_count = 0
    for instr in block.pseudo_instrs:
        if instr is None: continue
        if instr.getFlowType().isCall():
            call_count += 1
    return call_count

def get_instr_number(block):
    """
    Get the number of instructions in the block.
    """
    return sum(i is not None for i in block.pseudo_instrs)

def get_arithmetic_number(block):
    """
    Get the number of arithmetic instructions in the block.
    """
    arithmetic_count = 0
    for instr in block.pseudo_instrs:
        if instr is None: continue
        pcode_ops = instr.getPcode()
        for op in pcode_ops:
            if op.getOpcode() in ARITHMETIC_OPCODES:
                arithmetic_count += 1
    return arithmetic_count

def get_zero_bytes_number(block, memory):
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

def get_def_use_number(block):
    def_use_cnt = 0
    defs = []
    for instr in block.pseudo_instrs:
        if instr is None: continue
        pcode_ops = instr.getPcode()
        instr_def = []
        for op in pcode_ops:
            uses = op.getInputs()
            for use in uses:
                if use in defs:
                    def_use_cnt += 1
            instr_def.append(op.getOutput())
        defs.extend(instr_def)
    if def_use_cnt * 3 >= block.end_address.subtract(block.start_address):
    # if def_use_cnt >= 1:
        block.high_def_use_rate_flg = True
    else:
        block.high_def_use_rate_flg = False
    return def_use_cnt

def get_feature_vector(blocks, psuedo_disassembler: PseudoDisassembler, refs, listing, memory):
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
            get_def_use_number(block) / block_size
        ]
        block.feature_vector = feature_vec

def check_compare_branch(blocks, program):
    """
    Check conditional branches in the blocks following a comparison instructions.
    """
    for block in blocks:
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

def check_very_short(blocks):
    """
    Check if the block is very short, i.e., less than 3 instruction (12 bytes).
    """
    for block in blocks:
        if block.end_address.subtract(block.start_address) <= 12:
            block.very_short_flg = True
            block.type = "Code" if block.type == "Data" else "Data"
        else:
            block.very_short_flg = False
    return

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

        code_blocks = extract_code_blocks(listing, memory)
        data_blocks = extract_data_blocks(listing, memory)

        blocks = [*code_blocks, *data_blocks]
        blocks.sort(key=lambda b: b.start_address)

        pseudo_disassemble_blocks(blocks, program)

        get_feature_vector(blocks, PseudoDisassembler(program), program.getReferenceManager(), listing, memory)
        check_compare_branch(blocks, program)
        check_very_short(blocks)

        # check_def_use(blocks, PseudoDisassembler(program))
        with open('blocks.txt', 'w') as f:
            for block in blocks:
                f.write(f"{block}\n")
        logger.info(f"Extracted {len(blocks)} blocks from the program, exported to blocks.txt")
