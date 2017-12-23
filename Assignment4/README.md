Setup for running TELBO/JMVAE/BiVCCA on Google cloud with GPU

1. Setup instance on Google cloud
- Go to Google cloud compute engine, create VM instance
	Select us-east-1-c 
	CPU doesn't seem to be a bottleneck, mine is 4 cores and max RAM (I think it was 26GB)
	Select 1 NVidia P100 GPU (you will need to request it)
	Select Ubuntu 16.04
	Allow HTTP/HTTPS (don't know why, I followed a tutorial)

- To request GPU, go to Quota, then upgrade account. It will upgrade you to a paid account. You still have the $300 credits, and it will use it first.
- Search for Compute Engine P100 GPU for us-east-1, select and request. This might take a few minutes.

- Once you're in the VM
2. Install packages
sudo apt-get update
sudo apt-get install python-pip
sudo pip install --upgrade pip
sudo pip install numpy
sudo pip install scipy
sudo pip install sklearn
sudo apt-get install python-pip python-dev python-virtualenv

3. Clone repo
git clone https://github.com/google/joint_vae.git

4. Install tensorflow and GPU stuff.
4.1 NVidia cuda 8 toolkit
- On PC:
	Go to https://developer.nvidia.com/cuda-80-ga2-download-archive
	Select Linux, x86_64, Ubuntu, 16.04, deb (network)
	Download the installer
	Upload (top right corner button of terminal)
- On terminal:
	sudo dpkg -i cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
	sudo apt-get update
	sudo apt-get install cuda-8-0
export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64\${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
sudo pip install tensorflow

4.2 cuDNN
CUDNN_TAR_FILE="cudnn-8.0-linux-x64-v6.0.tgz"
wget http://developer.download.nvidia.com/compute/redist/cudnn/v6.0/${CUDNN_TAR_FILE}
tar -xzvf ${CUDNN_TAR_FILE}
sudo cp -P cuda/include/cudnn.h /usr/local/cuda-8.0/include
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda-8.0/lib64/
sudo chmod a+r /usr/local/cuda-8.0/lib64/libcudnn*
export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64\${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
sudo apt-get install libcupti-dev
sudo pip install tensorflow-gpu

4.3 Check correctness
nvidia-smi (if no error, this command will show you GPU usage (0 at this moment))
python (now you're running python console)
	import tensorflow as tf
	hello = tf.constant('Hello, TensorFlow!')
	sess = tf.Session()
	print(sess.run(hello))
- If this runs and outputs "Hello, TensorFlow!", then you're good!

5. Run code
cd joint_vae directory (project root)
- From now on, don't change directory, read line 77 first
source install/install_deps.sh
- Notice that now there is a (imag) in front of each line, it indicates that you're in the virtual environment

cd datasets
export PYTHONPATH=
cd ..
cd experiments
export PYTHONPATH=
cd .. (now you're back in project root)

- Go to preamble.sh, change path value, from imagination to imag (Rama made a mistake, alternatively, you can change install file and create a venv called imagination)
- Upload iclr_minista_fresh_jmvae_noslurm.sh and put it in scripts folder
source scripts/iclr_minista_fresh_jmvae_noslurm.sh

- If it doesn't run, there's a problem in setup. 
- If it runs, AWESOME! now cancel it. 
source script/iclr_minista_fresh_bivcca_noslurm.sh

- After successfully running the bash file
cd runs/imagination/$PLACE_OF_METHOD
cp process.py .
python process.py > results.txt

- The final results can be seen in results.txt 






