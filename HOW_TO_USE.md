# 🚗 How to Use This Guide

## 📁 What's in this folder?

- **README.md** - Complete setup and training instructions
- **train_ai.py** - AI training script
- **test_ai.py** - AI testing script

## 🚀 Quick Start (3 steps)

### Step 1: Setup Environment
Follow the instructions in `README.md` to install everything.

### Step 2: Start Training
```bash
# Terminal 1: Start simulator
cd DonkeySimLinux
./donkey_sim.x86_64

# Terminal 2: Start training
conda activate donkey
python train_ai.py
```

### Step 3: Test Your AI
```bash
python test_ai.py
```

## 🎯 Expected Results

### Training Progress
- **Episodes 1-20**: AI explores (0-10% success)
- **Episodes 21-50**: AI learns (10-40% success)  
- **Episodes 51-80**: AI improves (40-70% success)
- **Episodes 81-100**: AI masters driving (70-90% success)

### Success Indicators
- Success rate > 60% = Good AI
- Success rate > 80% = Excellent AI

## ❓ Need Help?

1. Make sure simulator is running
2. Check README.md for troubleshooting
3. Try running again

## 🎉 Success!

When you see "Your AI can drive!", congratulations! You've trained an AI that can drive around the track!

**You're using the same techniques as Tesla and Waymo!** 🚗💨 