echo "Einsum python"
python bench.py
echo "Opt-Einsum python"
python bench_opt_einsum.py
echo "Julia"
julia bench.jl
