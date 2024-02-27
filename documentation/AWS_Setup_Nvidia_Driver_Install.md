# Capstone AWS Setup 
- OS: Ubuntu
- Instance type: g5.2xlarge

#Nvidia Driver Install

## Pre-install Instructions

### Verify You Have a CUDA-Capable GPU

 Use the command : `lspci | grep -i nvidia`  
 For g5 instance should output: `00:1e.0 3D controller: NVIDIA Corporation GA102GL [A10G] (rev a1)`   
 
 
 
### Verify OS
Use the command: `uname -m && cat /etc/*release`
You should expect an output like 

	x86_64  
	DISTRIB_ID=Ubuntu
	DISTRIB_RELEASE=22.04
	DISTRIB_CODENAME=jammy
	DISTRIB_DESCRIPTION="Ubuntu 22.04.3 LTS"
	PRETTY_NAME="Ubuntu 22.04.3 LTS"
	NAME="Ubuntu"
	VERSION_ID="22.04"
	VERSION="22.04.3 LTS (Jammy Jellyfish)"
	VERSION_CODENAME=jammy
	ID=ubuntu
	ID_LIKE=debian
	HOME_URL="https://www.ubuntu.com/"
	SUPPORT_URL="https://help.ubuntu.com/"
	BUG_REPORT_URL="https://bugs.launchpad.net/ubuntu/"
	PRIVACY_POLICY_URL="https://www.ubuntu.com/legal/terms-and-policies/privacy-policy"
	UBUNTU_CODENAME=jammy
 
 
### Update all packages

Use the commands : `sudo apt-get update` followed by `sudo apt-get upgrade` and then reboot the system using `sudo reboot`

### Install GCC

To check if gcc is installed use : `gcc --version` expect `gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0`

To install gcc use: `sudo apt install gcc` optionally also install `sudo apt install build-essential`

### Install kernel headers and development packages
To find your kernal version: `uname -r`	   
expect: `6.2.0-1017-aws`

to install: `sudo apt-get install linux-headers-$(uname -r)`.   
remove outdated key, (if it exists): `sudo apt-key del 7fa2af80`


### Install the CUDA repository public GPG key.

Run: 

	wget https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch/cuda-keyring_1.1-1_all.deb
	sudo dpkg -i cuda-keyring_1.1-1_all.deb


Replace `$distro/$arch` with respective Linux distribution and architecture. Here it is `ubuntu2204/x86_64` (refer section [Verify OS](#Verify-OS))



## Install CUDA SDK

Update repo cache: `sudo apt-get update`

Install drivers: `sudo apt-get -y install cuda-drivers`    
Install CUDA toolkit: `sudo apt-get install cuda-toolkit`	  
Install GDS packages: `sudo apt-get install nvidia-gds`

Reboot System: `sudo reboot`

## Post-install instructions

### Export PATH variable (MANDATORY)

Run the following command: `export PATH=/usr/local/cuda-12.3/bin${PATH:+:${PATH}}` _Note that the version installed here is cuda-12.3 change this if necessary_

### Install Persistence Daemon

Run: `/usr/bin/nvidia-persistenced --verbose`

### Verify the installation
To verify that the drivers and related tools have been installed properly. It is useful to run a few tests.  

Clone the cuda-samples repo: `git clone https://github.com/NVIDIA/cuda-samples.git`. This contains a variety of sample files. 
I recommend running atleast the ones in the Utilities folder especaily the device Query. Refer the repo for more details.

Also try running `nvidia-smi`




