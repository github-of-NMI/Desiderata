source env/bin/activate

export HF_TOKEN=$(cat closedAI_bench/hf_token)


time python main.py
python plot.py
