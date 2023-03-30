# Instruction

Install miniconda, julia and git clone this repo.
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-py37_23.1.0-1-Linux-x86_64.sh
bash Miniconda3-py37_23.1.0-1-Linux-x86_64.sh
wget https://julialang-s3.julialang.org/bin/linux/x64/1.8/julia-1.8.5-linux-x86_64.tar.gz
tar zxvf julia-1.8.5-linux-x86_64.tar.gz
git clone https://github.com/Khoury-srg/Ouroboros.git
```

Add the following line to "~/.bashrc"
```bash
export PATH="$PATH:/path/to/<Julia directory>/bin" 
```
then in terminal
```bash
source ~/.bashrc
julia
```
checkout NeuralVerification.jl to the "ouroboros" branch.
```julia
using Pkg
Pkg.develop(path="/path/to/NeuralVerification.jl")
Pkg.add("LazySets")
```

Reopen terminal to activate miniconda and set up python environment
```bash
conda create --name ouroboros python==3.7
conda activate ouroboros
pip3 install torch torchvision torchaudio pandas onnx onnxruntime matplotlib annoy julia ipykernel
python
```
Inside python
```python
import julia
julia.install()
```

The following commands reproduce all the figures and the model accuracy table in our paper.
```bash
cd /path/to/ouroboros/src
python plot_figures.py
python generate_table.py
```
To re-run all the experiments:
```
python run_exp.py
```
All the models and training curves will be stored in `results` and figures will be sotred in `imgs`.

A copy of our results are stored in the `paper_results` folder for your reference.