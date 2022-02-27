#!/bin/bash
export DESIGN_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

export RISCV_TOOLTAG=20220210-1
export RISCV_TOOLCHAIN=$DESIGN_DIR/toolchains/lowrisc-toolchain-gcc-rv32imcb-$RISCV_TOOLTAG

export RISCV_GCC=$RISCV_TOOLCHAIN/bin/riscv32-unknown-elf-gcc
export RISCV_OBJCOPY=$RISCV_TOOLCHAIN/bin/riscv32-unknown-elf-objcopy

export SPIKE_DIR=$DESIGN_DIR/spike
export SPIKE_PATH=$SPIKE_DIR/bin

PATH=$RISCV_TOOLCHAIN/bin:$PATH
