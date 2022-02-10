#!/bin/bash
export REPO=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Maybe download and extract
export RISCV_TOOLTAG=20210412-1
export RISCV_TOOLCHAIN=$REPO/toolchains/lowrisc-toolchain-gcc-rv32imc-${RISCV_TOOLTAG}

if [ ! -d $RISCV_TOOLCHAIN ]
then
  mkdir -p $REPO/toolchains
  cd $REPO/toolchains
  wget https://github.com/lowRISC/lowrisc-toolchains/releases/download/${RISCV_TOOLTAG}/lowrisc-toolchain-gcc-rv32imc-${RISCV_TOOLTAG}.tar.xz
  tar -xvf lowrisc-toolchain-gcc-rv32imc-${RISCV_TOOLTAG}.tar.xz
  cd -
fi

export RISCV_GCC=$RISCV_TOOLCHAIN/bin/riscv32-unknown-elf-gcc

# export SPIKE_PATH=/home/grads/f/fzl1029/RISCV/IbexSpikeCosim/bin
source /opt/coe/synopsys/vcs/Q-2020.03-SP1-1/setup.vcs.sh # VCS directory in ECE hera server

PATH=${RISCV_TOOLCHAIN}/bin:${PATH}
