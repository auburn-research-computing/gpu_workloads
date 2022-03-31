.. _gpu_lab:

*************
GPU Workloads
*************

.. _Slides: https://hpc.auburn.edu/hpc/docs/hpcdocs/build/html/easley/hpc_training_gpu.pdf
   .. _Repository: https://github.com/auburn-research-computing/gpu_lab.git

| Presentation Slides_
| Github Repository_

CUDA Toolkit & GPU Programming Constructs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: console
        
   $ cd ~
   $ mkdir hpc_gpu
   $ cd hpc_gpu
   $ cp -R /tools/gpu/tutorials/* .
   $ ls
   $ cd hello
   $ module load cuda11.0/toolkit
   $ nvcc -o hello hello.cu 
   $ srun -N1 -n1 --partition=gpu4 --gres=gpu:tesla:1 ./hello
   $ nvcc -o threads hello_threads.cu 
   $ srun -N1 -n1 --partition=gpu4 --gres=gpu:tesla:1 ./threads 

Introduction
^^^^^^^^^^^^

#TODO: add basic gpu info e.g. partitions, devices, gres

Listing Available packages
^^^^^^^^^^^^^^^^^^^^^^^^^^

#TODO: add gpu related modules, tools, paths, etc.

Using Cuda with Pytorch
^^^^^^^^^^^^^^^^^^^^^^^^
Pytorch is one of the many popular deep learning frameworks used among data scientist. In order to set up and run CUDA operations, Pytorch provides the torch.cuda package. 
This package adds support for CUDA tensor types, that impliment the same function as CPU tensors but utilizes GPUS for computation.

Installing Pytorch
^^^^^^^^^^^^^^^^^^
Before we begin we will need to install Pytorch. We will create a virtual environment within our home directory exclusively for Pytorch.
For more information about virtual environments please visit the following:

https://hpc.auburn.edu/hpc/docs/hpcdocs/build/html/easley/python.html#python-virtual-environments

.. code-block:: console

   cd ~
   module load python
   mkdir virtualenvs
   cd virtualenvs
   python3 -m virtualenv pytorch   
   source pytorch/bin/activate

At this point, the virtual environment is created and activated using the source command. Now that the pytorch virtual environment is activated, install pytorch using the following

.. code-block:: console
   
   cd pytorch
   pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
   pip list installed
   deactivate


In this first exercise we will use the torch.cuda package to check the availability of the cuda device and to gather information. 

torch.cuda
^^^^^^^^^^

.. code-block:: console
   
   srun -N1 -n1 --partition=gpu2 --gres=gpu:tesla:1 --pty /bin/bash
   module load python
   module load cuda11.0/toolkit
   cd virtualenvs/pytorch
   source bin/activate
   python3
   import torch
   

To check if your system supports cuda, use the following command. is_available() will return a bool value either true if your system supports cuda or false.

.. code-block:: console
   
   torch.cuda.is_available()
   true   

The current_device() command will provide information about the id of the cuda device 

.. code-block:: console

   torch.cuda.current_device()
   0

Taking the following id value provided above, you can also retrieve the name of the device using the following command

.. code-block:: console
  
   torch.cuda.get_device_name(0)
   'Tesla T4'

To provide even further information using the id of the cuda device you can do the following

.. code-block:: console
 
   torch.cuda.get_device_properties(0)
   _CudaDeviceProperties(name='Tesla T4', major=7, minor=5, total_memory=15109MB, multi_processor_count=40)

Lets finish the interactive job by doing the following. Exit the python program and the interactive job as such

.. code-block:: console
 
   exit()
   

Copy the following exercise in the virtualenvs/pytorch directory

.. code-block:: console

   cp /tools/gpu/tutorials/pytorch/pytorch_example.py .
   chown username:username pytorch_example.py
   ./pytorch_example.py
   exit

Once placed back onto the login node, deactivate the virtual environment

.. code-block:: console

   deactivate

Batch Job Submission
^^^^^^^^^^^^^^^^^^^^   
Create a bash script named pytorch_lab.sh and place the following 

.. code-block:: console

   nano pytorch_lab.sh

.. code-block:: console

        
   #!/bin/bash

   #SBATCH --partition=gpu4
   #SBATCH --time=5:00
   #SBATCH --nodes=1
   #SBATCH --ntasks-per-node=1
   #SBATCH --gres=gpu:tesla:1
   #SBATCH --job-name=pytorch_lab

   module load python
   module load cuda11.0/toolkit/11.0.3

   source /home/username/virtualenvs/pytorch/bin/activate

   python3 pytorch_example.py  > results.out



.. code-block:: console

   sbatch pytorch_lab.sh
