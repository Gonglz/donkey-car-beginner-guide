# 🚗 Donkey Car Reinforcement Learning - Beginner Guide

- A simple guide for beginners to train an AI that can drive around a track using reinforcement learning!
- All of them are based on WSL2 Ubuntu20.04.

## 🎯 What You'll Learn

- **Reinforcement Learning**: How AI learns through trial and error
- **Neural Networks**: How AI processes visual information  
- **Training Process**: How to train AI models
- **Real Results**: Train an AI that can actually drive!

## 📁 What's Included

- **README.md** - Complete setup and training instructions
- **HOW_TO_USE.md** - Quick start guide
- **train_ai.py** - AI training script
- **test_ai.py** - AI testing script

## 🚀 Quick Start

### Step 1: Setup Environment
```bash
# Create project directory
cd ~
mkdir smartcar
cd smartcar

# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
# Follow prompts, type "yes" to initialize conda

# Reload environment
source ~/.bashrc

# Create Python environment
conda create -n donkey python=3.11 -y
conda activate donkey

# Install packages
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install opencv-python pillow tensorflow
```

### Step 2: Install Donkey Car
```bash
# Download Donkey Car
git clone https://github.com/autorope/donkeycar.git
cd donkeycar
pip install -e .
cd ..

# Download Gym environment
git clone https://github.com/tawnkramer/gym-donkeycar.git
cd gym-donkeycar
pip install -e .
cd ..

# Create application
donkey createcar --path ./mysim
```

### Step 3: Configure Application
```bash
# Edit configuration
nano mysim/myconfig.py
```

Add these lines to `myconfig.py`:
```python
DONKEY_GYM = True
DONKEY_SIM_PATH = "/home/$(whoami)/smartcar/DonkeySimLinux/donkey_sim.x86_64"
DONKEY_GYM_ENV_NAME = "donkey-generated-track-v0"
```

### Step 4: Download Simulator
```bash
# Download simulator
wget https://github.com/tawnkramer/gym-donkeycar/releases/download/v22.11.06/DonkeySimLinux.zip
unzip DonkeySimLinux.zip
chmod +x DonkeySimLinux/donkey_sim.x86_64
```

### Step 5: Copy Training Scripts
```bash
# Copy scripts from this repository
cp donkey_car_beginner_guide/train_ai.py .
cp donkey_car_beginner_guide/test_ai.py .
```

### Step 6: Start Training
```bash
# Terminal 1: Start simulator
cd DonkeySimLinux
./donkey_sim.x86_64

# Terminal 2: Start training
conda activate donkey
cd ~/smartcar
python train_ai.py
```

### Step 7: Test Your AI
```bash
python test_ai.py
```

## 📊 Expected Results

### Training Progress
- **Episodes 1-20**: AI explores randomly (0-10% success)
- **Episodes 21-50**: AI starts learning (10-40% success)
- **Episodes 51-80**: AI improves (40-70% success)
- **Episodes 81-100**: AI masters driving (70-90% success)

### Success Indicators
- **Success Rate**: Percentage of completed laps
- **Average Reward**: Higher is better
- **Average Steps**: Lower means faster completion

## ❓ Common Issues

### Simulator won't start
```bash
# Kill existing processes
pkill -f donkey_sim.x86_64

# Start simulator manually
cd DonkeySimLinux
./donkey_sim.x86_64
```

### conda command not found
```bash
source ~/.bashrc
```

### Port occupied
```bash
pkill -f donkey
# Then restart simulator
```

### Training stuck on "waiting for sim"
- Make sure simulator is running in a separate terminal
- Wait 10-15 seconds after starting simulator
- Check if simulator window appears

## 🎉 Success!

When you see:
```
🎉 Training completed!
Success rate: 85.0%
🎉 Your AI can drive!
```

Congratulations! You've trained an AI that can drive around the track!

## 🚗 What You Learned

1. **Reinforcement Learning**: AI learns through trial and error
2. **Neural Networks**: How AI processes visual information
3. **Reward Functions**: How to guide AI learning
4. **Training Process**: How to train AI models

**You're using the same techniques as Tesla and Waymo!** 🚗💨

## 📚 Learning Resources

- [Donkey Car Official Documentation](https://docs.donkeycar.com/)
- [Reinforcement Learning Tutorial](https://www.gymlibrary.dev/)
- [Python Tutorial](https://docs.python.org/3/tutorial/)

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

