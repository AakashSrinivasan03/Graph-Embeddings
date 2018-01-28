
export PATH="/usr/local/nvidia/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib:$LD_LIBRARY_PATH"
# Tools config for CUDA, Anaconda installed in the common /tools directory
source /tools/config.sh
# Activate your environment
source activate py35
# Change to the directory in which your code is present
cd /storage/home/priyesh/lap/graph_0018
# Run the code. The -u option is used here to use unbuffered writes
# so that output is piped to the file as and when it is produced.
# Here, the code is the MNIST Tensorflow example.
python -u __main__.py &> out
python -u lin_class.py &> out_class
python -u embed_visualisation.py 
