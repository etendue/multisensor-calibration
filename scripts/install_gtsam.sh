#!/bin/bash
# Script to install GTSAM

# Don't exit on error - we want to try multiple installation methods
set +e

echo "Installing GTSAM dependencies..."
sudo apt-get update
sudo apt-get install -y build-essential cmake libboost-all-dev libtbb-dev python3-dev python3-pip

echo "Checking if GTSAM is already installed..."
if python3 -c "import gtsam" &> /dev/null; then
    echo "GTSAM is already installed."
    exit 0
fi

echo "Attempting to install GTSAM from pip..."
pip install gtsam

# Check if pip installation was successful
if python3 -c "import gtsam" &> /dev/null; then
    echo "GTSAM installed successfully from pip."
    exit 0
fi

echo "Pip installation failed. Attempting to install GTSAM using conda..."

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "Conda found. Installing GTSAM from conda-forge..."
    conda install -y -c conda-forge gtsam

    # Check if conda installation was successful
    if python3 -c "import gtsam" &> /dev/null; then
        echo "GTSAM installed successfully from conda-forge."
        exit 0
    else
        echo "Conda installation failed."
    fi
else
    echo "Conda not found. Skipping conda installation."
fi

echo "Package manager installations failed. Building GTSAM from source..."

# Create temporary directory
TEMP_DIR=$(mktemp -d)
cd $TEMP_DIR

# Clone GTSAM repository
git clone https://github.com/borglab/gtsam.git
cd gtsam

# Create build directory
mkdir build
cd build

# Configure and build
cmake -DGTSAM_BUILD_PYTHON=ON -DGTSAM_PYTHON_VERSION=3 -DGTSAM_USE_SYSTEM_EIGEN=OFF ..
make -j$(nproc)

# Install
sudo make install

# Add library path to ldconfig
sudo bash -c 'echo "/usr/local/lib" > /etc/ld.so.conf.d/gtsam.conf'
sudo ldconfig

# Install Python wrapper
cd ../python
pip install -e .

# Clean up
cd $TEMP_DIR
rm -rf gtsam

echo "Checking installation..."
if python3 -c "import gtsam; print('GTSAM installed successfully')"; then
    echo "GTSAM installation successful."
else
    echo "GTSAM installation failed. Please check the error messages above."
    exit 1
fi

echo "GTSAM installation complete."
