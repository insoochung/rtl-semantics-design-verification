#!/bin/bash
set -e

if [ -z ${RISCV_TOOLCHAIN+x} ];
then
  echo "Environment variable RISCV_TOOLCHAIN not set. Maybe you haven't sourced 'set_env_vars.sh'?"
  exit 0
fi

if [ ! -d $RISCV_TOOLCHAIN ]
then
  mkdir -p $REPO/toolchains
  cd $REPO/toolchains
  wget https://github.com/lowRISC/lowrisc-toolchains/releases/download/$RISCV_TOOLTAG/lowrisc-toolchain-gcc-rv32imc-$RISCV_TOOLTAG.tar.xz
  tar -xvf lowrisc-toolchain-gcc-rv32imc-$RISCV_TOOLTAG.tar.xz
  cd -
  echo "RISCV toolchain downloaded and extracted at {$RISCV_TOOLCHAIN}"
else
  echo "$RISCV_TOOLCHAIN already exists. Maybe remove the directory and retry?"
fi
