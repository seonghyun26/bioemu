PDB_PATH=./res/debug/topology.pdb
XTC_PATH=./res/debug/samples.xtc

python -m bioemu.sidechain_relax \
  --pdb-path ${PDB_PATH} \
  --xtc-path ${XTC_PATH} \
