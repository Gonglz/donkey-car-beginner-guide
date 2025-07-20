#!/usr/bin/env python3
"""
Test your trained AI model
"""

import os
import sys
import numpy as np
import tensorflow as tf
import gym
import gym_donkeycar

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'donkeycar'))
sys.path.append(os.path.join(current_dir, 'gym-donkeycar'))

def preprocess_image(image):
    """Preprocess camera image"""
    image = tf.image.resize(image, (80, 120))
    image = image / 255.0
    return image

def test_model(model_path, num_episodes=5):
    """Test the trained model"""
    print(f"🧪 Testing AI model: {model_path}")
    
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    # Action mapping: convert discrete action to [steering, throttle]
    def map_action(action):
        if action == 0:  # straight
            return [0.0, 0.5]
        elif action == 1:  # left
            return [-0.5, 0.5]
        elif action == 2:  # right
            return [0.5, 0.5]
        elif action == 3:  # accelerate
            return [0.0, 1.0]
        elif action == 4:  # brake
            return [0.0, 0.0]
        else:
            return [0.0, 0.5]
    
    # Create environment
    env = gym.make("donkey-generated-track-v0", conf={
        "body_style": "donkey",
        "body_rgb": (128, 128, 128),
        "car_name": "Test_Car",
        "font_size": 100,
        "racer_name": "Test_Agent",
        "country": "USA",
        "bio": "AI Testing"
    })
    
    successful_runs = 0
    total_rewards = []
    total_steps = []
    
    for episode in range(num_episodes):
        state = env.reset()
        state = preprocess_image(state)
        state = np.expand_dims(state, axis=0)
        
        total_reward = 0
        step_count = 0
        
        print(f"Test Episode {episode + 1}/{num_episodes} starting...")
        
        for step in range(1000):
            # Get action from model
            action_values = model.predict(state, verbose=0)
            action = np.argmax(action_values[0])
            
            # Map discrete action to continuous [steering, throttle]
            continuous_action = map_action(action)
            
            # Take action
            next_state, reward, done, info = env.step(continuous_action)
            next_state = preprocess_image(next_state)
            next_state = np.expand_dims(next_state, axis=0)
            
            state = next_state
            total_reward += reward
            step_count += 1
            
            # Print progress
            if step % 100 == 0:
                speed = info.get('speed', 0)
                cte = info.get('cte', 0)
                print(f"  Step {step}: Action={action}, Speed={speed:.2f}, CTE={cte:.2f}")
            
            if done:
                if info.get('finished', False):
                    successful_runs += 1
                    print(f"  🎉 Test episode {episode + 1} completed successfully!")
                else:
                    print(f"  💥 Test episode {episode + 1} failed!")
                break
        
        total_rewards.append(total_reward)
        total_steps.append(step_count)
        
        print(f"  Steps: {step_count}, Total Reward: {total_reward:.2f}")
        print("-" * 40)
    
    env.close()
    
    # Calculate statistics
    success_rate = (successful_runs / num_episodes) * 100
    avg_reward = np.mean(total_rewards)
    avg_steps = np.mean(total_steps)
    
    print(f"\n📊 Test Results:")
    print(f"Successful runs: {successful_runs}/{num_episodes}")
    print(f"Success rate: {success_rate:.1f}%")
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Average steps: {avg_steps:.1f}")
    
    if success_rate >= 80:
        print("🎉 Excellent! Your AI is working perfectly!")
    elif success_rate >= 60:
        print("👍 Good! Your AI is performing well.")
    elif success_rate >= 40:
        print("⚠️  Your AI needs more training.")
    else:
        print("❌ Your AI needs significant improvement.")
    
    return success_rate, avg_reward, avg_steps

def main():
    # Test the best model
    model_path = "mysim/models/ai_best_model.h5"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Please run training first:")
        print("python train_ai.py")
        return
    
    # Test the model
    success_rate, avg_reward, avg_steps = test_model(model_path)
    
    # Save test results
    results = {
        'model_path': model_path,
        'success_rate': success_rate,
        'avg_reward': avg_reward,
        'avg_steps': avg_steps
    }
    
    print(f"\n💾 Test results saved to test_results.txt")
    with open("test_results.txt", "w") as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")

if __name__ == "__main__":
    main() 