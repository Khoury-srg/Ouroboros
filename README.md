Install miniconda, julia and git clone this repo.
"""
wget https://repo.anaconda.com/miniconda/Miniconda3-py37_23.1.0-1-Linux-x86_64.sh
bash Miniconda3-py37_23.1.0-1-Linux-x86_64.sh
# yes yes
wget https://julialang-s3.julialang.org/bin/linux/x64/1.8/julia-1.8.5-linux-x86_64.tar.gz
tar zxvf julia-1.8.5-linux-x86_64.tar.gz
git clone https://github.com/Khoury-srg/Ouroboros.git
"""

Add the following line to "~/.bashrc"
"""
export PATH="$PATH:/path/to/<Julia directory>/bin" 
"""
then in terminal
"""
source ~/.bashrc
julia
"""
checkout NeuralVerification.jl to the "ouroboros" branch.
"""
using Pkg
Pkg.develop(path="/path/to/NeuralVerification.jl")
Pkg.add("LazySets")
"""

Reopen terminal to activate miniconda and set up python environment
"""
conda create --name ouroboros python==3.7
conda activate ouroboros
pip3 install torch torchvision torchaudio pandas onnx onnxruntime matplotlib annoy julia ipykernel
python
"""
Inside python
"""
import julia
julia.install()
"""

The following commands run all experiments, draw all figures and generate the model accuracy table in our paper correspondingly.
"""
cd /path/to/ouroboros/src
python run_exp.py
python plot_figures.py
python generate_table.py
"""