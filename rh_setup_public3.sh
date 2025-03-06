pip install git+https://github.com/HabanaAI/DeepSpeed.git@1.19.0
#pip install git+https://github.com/huggingface/optimum-habana.git
pip install git+https://github.com/huggingface/optimum-habana@refs/pull/1777/head

\rm -rf test3
mkdir test3
cd test3
git clone https://github.com/instructlab/training
cd training
git apply /software/users/jojimon/habana_instructlab.patch
pip uninstall instructlab-training
pip install .

pip install git+https://github.com/huggingface/transformers.git@482cb28
#pip install accelerate==0.34.1

export HF_HOME="/software/users/jojimon/HF_HOME"
#cp /software/users/jojimon/MySandbox/upstream/public_upstream/orig/training/scripts/run_ds.py scripts
cd src/instructlab/training
cp /software/users/jojimon/rh_train_simple_fsdp.py .
ln -s ../../../sample-data .

