sudo apt update;
sudo apt install -y build-essential
sudo /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install llama.cpp

python -m venv env
source env/bin/activate

pip install -U pandas matplotlib requests pyright ruff tqdm huggingface-hub

