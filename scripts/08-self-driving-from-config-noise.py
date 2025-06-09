# CNN self-driving car from config, without fiducial markers
# Mods: 
# 1. Add road segmentation
# 2. Remove fiducial markers
# 3. Add dynamic steering bins based on --bins argument
# 4. Add GPU-optimized noise type and intensity as arguments

#########
# MODEL #
#########

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import numpy as np
import cv2
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import carla_helpers as helpers
import image_noise_utils as noise_utils  # For reference, will override noise function

class NVIDIANet(nn.Module):
    def __init__(self, num_outputs=15, dropout_rate=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(24, 32, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(32, 48, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1152, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, num_outputs)
    
    def forward(self, x):
        x = x / 255.0
        x = F.elu(self.conv1(x))
        x = self.dropout(x)
        x = F.elu(self.conv2(x))
        x = self.dropout(x)
        x = F.elu(self.conv3(x))
        x = self.dropout(x)
        x = F.elu(self.conv4(x))
        x = self.dropout(x)
        x = F.elu(self.conv5(x))
        x = self.dropout(x)
        x = self.flatten(x)
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        x = self.fc4(x)
        return F.softmax(x, dim=1)

def load_model(model, model_path, device='cuda'):
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

#############
# SPECTATOR #
#############

import math
import carla

def set_spectator_camera_following_car(world, vehicle):
    spectator = world.get_spectator()
    transform = vehicle.get_transform()
    location = transform.location
    rotation = transform.rotation
    offset_location = location - carla.Location(x=35 * math.cos(math.radians(rotation.yaw)),
                                              y=35 * math.sin(math.radians(rotation.yaw)))
    offset_location.z += 20
    spectator.set_transform(carla.Transform(offset_location, carla.Rotation(pitch=-15, yaw=rotation.yaw, roll=rotation.roll)))
    return spectator

def draw_permanent_waypoint_lines(world, waypoints, color=carla.Color(0, 255, 0), thickness=2, life_time=0):
    for i in range(len(waypoints) - 1):
        wp1, wp2 = waypoints[i], waypoints[i + 1]
        world.debug.draw_line(wp1.transform.location, wp2.transform.location, thickness=thickness, color=color, life_time=life_time)

##############
# SELF-DRIVE #
##############

import carla
import numpy as np
import torch
import cv2
import queue
import threading
import time
from config_utils import load_config
import argparse

class CarlaSteering:
    def __init__(self, config, model_path='model.pth', distances_file='self_driving_distances_05.txt', bins=15, noise_type=None, intensity=None):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.distances_file = distances_file
        self.bins = bins
        self.noise_type = noise_type
        self.intensity = intensity

        self.steering_values = {
            3: [-0.065, 0.0, 0.065],
            5: [-0.065, -0.015, 0.0, 0.015, 0.065],
            15: [-0.065, -0.055, -0.045, -0.035, -0.025, -0.015, -0.005, 0.0,
                 0.005, 0.015, 0.025, 0.035, 0.045, 0.055, 0.065]
        }
        if self.bins not in self.steering_values:
            raise ValueError(f"Invalid bins value. Must be 3, 5, or 15. Got {self.bins}")

        sim_config = config['simulation']
        self.client = carla.Client(sim_config['server_host'], sim_config['server_port'])
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()

        settings = self.world.get_settings()
        settings.synchronous_mode = sim_config['synchronous_mode']
        settings.fixed_delta_seconds = sim_config['fixed_delta_seconds_self_drive']
        self.world.apply_settings(settings)
        
        self.model = NVIDIANet(num_outputs=len(self.steering_values[self.bins]))
        self.model = load_model(self.model, model_path, self.device)
        
        control_config = config.get('control', {})
        self.max_steering_angle = control_config.get('max_steering_angle', 1.0)
        self.steering_smoothing = control_config.get('steering_smoothing', 0.5)
        self.last_steering = 0.0
        self.target_speed = sim_config.get('target_speed', 10)
        
        camera_config = config['camera']
        self.image_width = camera_config['image_width']
        self.image_height = camera_config['image_height']
        
        img_proc_config = config['image_processing']
        self.crop_top = img_proc_config['crop_top']
        self.crop_bottom = img_proc_config['crop_bottom']
        self.resize_width = img_proc_config['resize_width']
        self.resize_height = img_proc_config['resize_height']
        
        self.image_queue = queue.Queue(maxsize=1)
        self.current_image = None
        self.display_counter = 0
        
    def setup_vehicle(self):
        sim_config = self.config['simulation']
        self.client.load_world(sim_config['town'])
        self.world = self.client.get_world()
        
        vehicle_config = self.config['vehicle']
        self.waypoints = helpers.get_town04_figure8_waypoints(self.world, lane_id=-2, waypoint_distance=vehicle_config['waypoint_distance'])
        print(f"Loaded {len(self.waypoints)} waypoints.")
        
        first_waypoint = self.waypoints[0]
        spawn_location = first_waypoint.transform.location
        spawn_location.z += vehicle_config['spawn_height_offset']
        spawn_point = carla.Transform(spawn_location, first_waypoint.transform.rotation)
        
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter(vehicle_config['model'].split('.')[-1])[0]
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        
        draw_permanent_waypoint_lines(self.world, self.waypoints, color=carla.Color(35, 35, 0), thickness=1.5, life_time=0)

        camera_config = self.config['camera']
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(self.image_width))
        camera_bp.set_attribute('image_size_y', str(self.image_height))
        camera_bp.set_attribute('fov', str(camera_config['fov']))
        
        pos = camera_config['position']
        rot = camera_config['rotation']
        camera_spawn_point = carla.Transform(
            carla.Location(x=pos['x'], y=pos['y'], z=pos['z']),
            carla.Rotation(pitch=rot['pitch'], yaw=rot['yaw'], roll=rot['roll'])
        )
        self.camera = self.world.spawn_actor(camera_bp, camera_spawn_point, attach_to=self.vehicle)
        self.camera.listen(self.process_image)
        
    def process_image(self, image):
        img = np.array(image.raw_data).reshape(self.image_height, self.image_width, 4)[:, :, :3]
        img = img[:, :, [2, 1, 0]]  # BGR to RGB
        tensor_img = torch.from_numpy(img).permute(2, 0, 1).float().to(self.device) / 255.0
        
        if self.noise_type and self.intensity is not None:
            if self.noise_type == 'gaussian_noise':
                sigma = 10 * (2 ** (self.intensity - 1))  # Exaggerated Gaussian
                noise = torch.normal(mean=0, std=sigma, size=tensor_img.shape, device=self.device)
                tensor_img = torch.clamp(tensor_img + noise, 0, 1)
            elif self.noise_type == 'brightness':
                brightness_factor = 1 + (self.intensity - 5) * 0.1  # Range: 0.5 to 1.5
                tensor_img = torch.clamp(tensor_img * brightness_factor, 0, 1)
        
        self.image_queue.put(tensor_img.cpu().numpy())  # Convert back to NumPy for queue (temporary)
        
    def preprocess_image(self, img):
        self.original_img = img.copy()
        
        # Convert back to tensor for GPU processing
        tensor_img = torch.from_numpy(img).permute(2, 0, 1).float().to(self.device) / 255.0
        
        cropped = tensor_img[:, self.crop_top:self.crop_bottom, :]
        resized = F.interpolate(cropped.unsqueeze(0), size=(self.resize_height, self.resize_width), mode='bilinear', align_corners=False).squeeze(0)
        
        # YUV conversion on CPU for simplicity (can be optimized further)
        yuv = cv2.cvtColor((resized.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8), cv2.COLOR_RGB2YUV)
        self.preprocessed_img = yuv.copy()
        
        yuv = torch.from_numpy(yuv.transpose(2, 0, 1)).float().to(self.device) / 255.0
        return yuv.unsqueeze(0)
        
    def display_images(self):
        if not (hasattr(self, 'original_img') and hasattr(self, 'preprocessed_img')):
            return
        
        display_height = 264
        aspect_ratio = self.original_img.shape[1] / self.original_img.shape[0]
        display_width = int(display_height * aspect_ratio)
        
        if not hasattr(self, 'canvas') or self.canvas.shape[1] != display_width + self.image_width + 20:
            self.canvas = np.zeros((display_height, display_width + self.image_width + 20, 3), dtype=np.uint8)
        
        original_resized = cv2.resize(self.original_img, (display_width, display_height))
        preprocessed_display = cv2.resize(self.preprocessed_img, (self.image_width, 264))
        
        self.canvas[:, :display_width] = original_resized
        self.canvas[:, display_width+20:] = preprocessed_display
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.canvas, 'Original Camera Feed', (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(self.canvas, 'Neural Network Input (YUV)', (display_width+30, 30), font, 1, (255, 255, 255), 2)
        
        cv2.imshow('Camera Views', self.canvas)
        cv2.waitKey(1)
        
    def predict_steering(self, image):
        with torch.no_grad():
            class_probs = self.model(image)
            predicted_class = torch.argmax(class_probs, dim=1).item()
            steering_angle = self.steering_values[self.bins][predicted_class]
        
        steering_angle = np.clip(steering_angle, -self.max_steering_angle, self.max_steering_angle)
        self.last_steering = steering_angle
        return steering_angle

    def apply_control(self, steering):
        control = carla.VehicleControl()
        control.steer = steering
        current_velocity = self.vehicle.get_velocity()
        speed = 3.6 * math.sqrt(current_velocity.x**2 + current_velocity.y**2)
        speed_error = self.target_speed - speed
        if speed_error > 0:
            control.throttle = min(0.3, speed_error / self.target_speed)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = min(0.3, -speed_error / self.target_speed)
        self.vehicle.apply_control(control)

    def find_nearest_waypoint(self, vehicle_location, waypoints):
        min_dist = float('inf')
        nearest_idx = 0
        for i, wp in enumerate(waypoints):
            dist = vehicle_location.distance(wp.transform.location)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
        return nearest_idx, min_dist
    
    def get_perpendicular_distance(self, vehicle_location, wp1, wp2):
        p = np.array([vehicle_location.x, vehicle_location.y])
        a = np.array([wp1.transform.location.x, wp1.transform.location.y])
        b = np.array([wp2.transform.location.x, wp2.transform.location.y])
        ab = b - a
        ap = p - a
        ab_norm = np.dot(ab, ab)
        if ab_norm == 0:
            return np.linalg.norm(p - a)
        t = np.dot(ap, ab) / ab_norm
        t = max(0, min(1, t))
        closest = a + t * ab
        return np.linalg.norm(p - closest)

    def run(self):
        try:
            output_dir = self.config['output']['directory']
            os.makedirs(output_dir, exist_ok=True)
        except (KeyError, OSError) as e:
            print(f"Error setting up output directory: {e}")
            output_dir = None

        self_driving_distances = [0.0]
        current_wp_idx = 1

        try:
            self.setup_vehicle()
            print("Vehicle and sensors initialized. Starting control loop...")
            
            while current_wp_idx < len(self.waypoints):
                while not self.image_queue.empty():
                    _ = self.image_queue.get()
                
                self.world.tick()
                
                try:
                    img = self.image_queue.get(timeout=0.1)
                    processed_img = self.preprocess_image(img)
                    steering = self.predict_steering(processed_img)
                    self.apply_control(steering)
                    set_spectator_camera_following_car(self.world, self.vehicle)
                    
                    self.display_counter += 1
                    if self.display_counter % 5 == 0:
                        self.display_images()
                    
                except queue.Empty:
                    print("Warning: Frame missed!")
                    
        except KeyboardInterrupt:
            print("Stopping...")
        finally:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
            
            if hasattr(self, 'camera'):
                self.camera.destroy()
            if hasattr(self, 'vehicle'):
                self.vehicle.destroy()
            
            if output_dir and self_driving_distances:
                with open(os.path.join(output_dir, self.distances_file), 'w') as f:
                    for dist in self_driving_distances:
                        f.write(f"{dist:.4f}\n")
                print(f"Self-driving ended. Recorded {len(self_driving_distances)} distances.")
            else:
                print("Self-driving ended. No distances saved due to invalid output directory or no distances recorded.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run CARLA self-driving simulation.')
    parser.add_argument('--config', type=str, default='config.json', help='Path to the configuration JSON file')
    parser.add_argument('--model', type=str, default='model.pth', help='Path to the trained model file')
    parser.add_argument('--distance_filename', type=str, default='self_driving_distances_05.txt', help='Path to the distances file')
    parser.add_argument('--bins', type=int, choices=[3, 5, 15], default=15, help='Number of steering bins')
    parser.add_argument('--noise_type', type=str, choices=['brightness', 'gaussian_noise'], default=None, help='Type of noise to apply')
    parser.add_argument('--intensity', type=int, choices=range(1, 11), default=None, help='Intensity level of noise')
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        controller = CarlaSteering(config, model_path=args.model, distances_file=args.distance_filename, 
                                 bins=args.bins, noise_type=args.noise_type, intensity=args.intensity)
        controller.run()
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage:
# For 5 bins with Gaussian noise at intensity 5
# python 08-self-driving-from-config-noise.py \
# --config /home/daniel/git/neurips-2025/scripts/config_640x480_segmented_06.json \
# --model /home/daniel/git/neurips-2025/scripts/best_quantized_steering_model_5_bins_balanced_20250529-211142.pth \
# --distance_filename ClsCNN5binBalanced_self_driving_gaussian_noise_5_distances.txt \
# --bins 5 \
# --noise_type gaussian_noise \
# --intensity 5

# Example usage:
# For 3 bins with no noise
# python 08-self-driving-from-config_05_15_bins.py \
# --config /home/daniel/git/neurips-2025/scripts/config_640x480_segmented_05.json \
# --model /home/daniel/git/neurips-2025/scripts/best_quantized_steering_model_3_bins.pth \
# --distance_filename ClsCNN3bin_self_driving_distances.txt \
# --bins 3

# ClsCNN5binBalanced
# For 5 bins with Gaussian noise at intensity 3
# python 08-self-driving-from-config-noise.py \
# --config /home/daniel/git/neurips-2025/scripts/config_640x480_segmented_06.json \
# --model /home/daniel/git/neurips-2025/scripts/best_quantized_steering_model_5_bins_balanced_20250529-211142.pth \
# --distance_filename ClsCNN5binBalanced_self_driving_gaussian_noise_3_distances.txt \
# --bins 5 \
# --noise_type gaussian_noise \
# --intensity 3

# ClsCNN5binBalanced
# For 5 bins with Gaussian noise at intensity 4
# python 08-self-driving-from-config-noise.py \
# --config /home/daniel/git/neurips-2025/scripts/config_640x480_segmented_06.json \
# --model /home/daniel/git/neurips-2025/scripts/best_quantized_steering_model_5_bins_balanced_20250529-211142.pth \
# --distance_filename ClsCNN5binBalanced_self_driving_gaussian_noise_4_distances.txt \
# --bins 5 \
# --noise_type gaussian_noise \
# --intensity 4

# ClsCNN5binBalanced
# For 5 bins with Gaussian noise at intensity 5
# python 08-self-driving-from-config-noise.py \
# --config /home/daniel/git/neurips-2025/scripts/config_640x480_segmented_06.json \
# --model /home/daniel/git/neurips-2025/scripts/best_quantized_steering_model_5_bins_balanced_20250529-211142.pth \
# --distance_filename ClsCNN5binBalanced_self_driving_gaussian_noise_4_distances.txt \
# --bins 5 \
# --noise_type gaussian_noise \
# --intensity 5

# ClsCNN5binBalanced - no distance logging
# For 5 bins with Gaussian noise at intensity 5
# python 08-self-driving-from-config-noise.py \
# --config /home/daniel/git/neurips-2025/scripts/config_640x480_segmented_06.json \
# --model /home/daniel/git/neurips-2025/scripts/best_quantized_steering_model_5_bins_balanced_20250529-211142.pth \
# --bins 5 \
# --noise_type gaussian_noise \
# --intensity 5