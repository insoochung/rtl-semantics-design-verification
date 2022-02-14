#!/bin/bash
set -e

git submodule update --init --recursive

if [ -z ${SPIKE_DIR+x} ];
then
  echo "Environment variable SPIKE_DIR not set. Maybe you haven't sourced 'set_env_vars.sh'?"
  exit 0
fi

# Maybe build spike binaries
if [ ! -d $SPIKE_PATH ]
then
  mkdir -p $REPO/riscv-isa-sim/build
  cd $REPO/riscv-isa-sim/build
  ../configure --enable-commitlog --enable-misaligned --prefix=$SPIKE_DIR
  make -j16
  make install
  cd -
  echo "Spike built. Binary can be found at '$SPIKE_PATH'."
else
  echo "'$SPIKE_PATH' already exists. Maybe remove the directory and retry?"
fi
