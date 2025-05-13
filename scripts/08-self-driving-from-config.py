# CNN self-driving car from config, with fiducial markers
# Mods: 
# 1. Add road segmentation
# 2. Add fiducial markers

#########
# MODEL #
#########

## Helper functions
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
# Add the src/ directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import carla_helpers as helpers

class NVIDIANet(nn.Module):
    def __init__(self, num_outputs=1, dropout_rate=0.1):
        super(NVIDIANet, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(24, 32, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(32, 48, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3)
        
        # Dropout layer
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # Dense layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1152, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, num_outputs)
        
    def forward(self, x):
        # Input normalization
        x = x / 255.0
        
        # Convolutional layers with ELU activation and dropout
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
        
        # Flatten and dense layers
        x = self.flatten(x)
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        x = self.fc4(x)
        
        return x

def load_model(model, model_path, device='cuda'):
    """Load a saved model"""
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

    spectator.set_transform(carla.Transform(offset_location,
                                          carla.Rotation(pitch=-15, yaw=rotation.yaw, roll=rotation.roll)))
    return spectator

def draw_permanent_waypoint_lines(world, waypoints, color=carla.Color(0, 255, 0), thickness=2, life_time=0):
    """
    Draw permanent lines linking every waypoint on the road.

    Parameters:
    - world: CARLA world object.
    - waypoints: List of waypoints to link.
    - color: Color of the lines (default is neon green).
    - thickness: Thickness of the lines (default is 0.1 meters).
    """
    for i in range(len(waypoints) - 1):
        # Get the current and next waypoint
        wp1 = waypoints[i]
        wp2 = waypoints[i + 1]

        # Draw a line between the two waypoints
        world.debug.draw_line(
            wp1.transform.location,  # Start point
            wp2.transform.location,  # End point
            thickness=thickness,     # Line thickness
            color=color,             # Line color
            life_time=life_time      # Permanent line (life_time=0 means infinite)
        )

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
    def __init__(self, config, model_path='model.pth'):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load simulation settings
        sim_config = config['simulation']
        self.client = carla.Client(sim_config['server_host'], sim_config['server_port'])
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        
        # Set synchronous mode with fixed time step
        settings = self.world.get_settings()
        settings.synchronous_mode = sim_config['synchronous_mode']
        settings.fixed_delta_seconds = sim_config['fixed_delta_seconds_self_drive']
        self.world.apply_settings(settings)
        
        # Initialize model
        self.model = NVIDIANet()
        self.model = load_model(self.model, model_path, self.device)
        
        # Load control parameters
        control_config = config.get('control', {})
        self.max_steering_angle = control_config.get('max_steering_angle', 1.0)
        self.steering_smoothing = control_config.get('steering_smoothing', 0.5)
        self.last_steering = 0.0
        self.target_speed = sim_config.get('target_speed', 10)
        
        # Load camera settings
        camera_config = config['camera']
        self.image_width = camera_config['image_width']
        self.image_height = camera_config['image_height']
        
        # Load image processing parameters
        img_proc_config = config['image_processing']
        self.crop_top = img_proc_config['crop_top']
        self.crop_bottom = img_proc_config['crop_bottom']
        self.resize_width = img_proc_config['resize_width']
        self.resize_height = img_proc_config['resize_height']
        
        # Image queue for processing
        self.image_queue = queue.Queue()
        self.current_image = None
        
    def setup_vehicle(self):
        """Spawn and setup the ego vehicle with sensors"""
        # Load town from config
        sim_config = self.config['simulation']
        self.client.load_world(sim_config['town'])
        self.world = self.client.get_world()
        
        # Load vehicle settings
        vehicle_config = self.config['vehicle']
        # waypoints = self.world.get_map().generate_waypoints(1.0)
        waypoints = helpers.get_town04_figure8_waypoints(self.world, lane_id=-2) 
        route_waypoints = [w for w in waypoints if w.road_id == vehicle_config['spawn_road_id'] 
                         and w.lane_id == vehicle_config['spawn_lane_id']]
        if not route_waypoints:
            raise ValueError(f"Could not find waypoints for road {vehicle_config['spawn_road_id']}, "
                           f"lane {vehicle_config['spawn_lane_id']}")
        
        # Get spawn point
        first_waypoint = route_waypoints[0]
        spawn_location = first_waypoint.transform.location
        spawn_location.z += vehicle_config['spawn_height_offset']
        spawn_point = carla.Transform(spawn_location, first_waypoint.transform.rotation)
        
        # Spawn vehicle
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter(vehicle_config['model'].split('.')[-1])[0]
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        
        # lane segmentation
        draw_permanent_waypoint_lines(self.world, waypoints, color=carla.Color(35, 35, 0), thickness=1.5, life_time=0)

        # Spawn camera
        camera_config = self.config['camera']
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(self.image_width))
        camera_bp.set_attribute('image_size_y', str(self.image_height))
        camera_bp.set_attribute('fov', str(camera_config['fov']))
        
        # Camera position and rotation
        pos = camera_config['position']
        rot = camera_config['rotation']
        camera_spawn_point = carla.Transform(
            carla.Location(x=pos['x'], y=pos['y'], z=pos['z']),
            carla.Rotation(pitch=rot['pitch'], yaw=rot['yaw'], roll=rot['roll'])
        )
        self.camera = self.world.spawn_actor(camera_bp, camera_spawn_point, attach_to=self.vehicle)
        self.camera.listen(self.process_image)
        
    def process_image(self, image):
        """
        Callback to process images from CARLA camera
        Convert image from BRG to RGB and add fiducial markers
        """
        # Convert CARLA image to numpy array
        img = np.array(image.raw_data).reshape(self.image_height, self.image_width, 4)
        img = img[:, :, :3]  # Remove alpha channel
        img = img[:, :, [2, 1, 0]]  # Convert BGR to RGB
        
        # Add fiducial markers
        marker_size = 20
        marker_color = [255, 0, 0]  # Red in RGB
        offset = 90  # Pixels from left/right edges
        bottom_y = self.image_height - marker_size  # Start at bottom of image

        # Left marker
        img[bottom_y:bottom_y + marker_size, offset:offset + marker_size, :] = marker_color

        # Right marker
        right_x = self.image_width - offset - marker_size
        img[bottom_y:bottom_y + marker_size, right_x:right_x + marker_size, :] = marker_color
        
        # Store in queue
        self.image_queue.put(img)
        
    def preprocess_image(self, img):
        """Preprocess image for neural network"""
        self.original_img = img.copy()
        
        # Crop and resize using config values
        cropped = img[self.crop_top:self.crop_bottom, :]
        resized = cv2.resize(cropped, (self.resize_width, self.resize_height))
        
        # Convert to YUV
        yuv = cv2.cvtColor(resized, cv2.COLOR_RGB2YUV)
        self.preprocessed_img = yuv.copy()
        
        # Prepare for PyTorch
        yuv = yuv.transpose((2, 0, 1))
        yuv = np.ascontiguousarray(yuv)
        
        return torch.from_numpy(yuv).float().unsqueeze(0).to(self.device)
        
    def display_images(self):
        """Display original and preprocessed images side by side"""
        if hasattr(self, 'original_img') and hasattr(self, 'preprocessed_img'):
            display_height = 264
            aspect_ratio = self.original_img.shape[1] / self.original_img.shape[0]
            display_width = int(display_height * aspect_ratio)
            original_resized = cv2.resize(self.original_img, (display_width, display_height))
            
            preprocessed_display = cv2.resize(self.preprocessed_img, (self.image_width, 264))
            
            canvas_width = display_width + self.image_width + 20
            canvas = np.zeros((display_height, canvas_width, 3), dtype=np.uint8)
            
            canvas[:, :display_width] = original_resized
            canvas[:, display_width+20:] = preprocessed_display
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(canvas, 'Original Camera Feed', (10, 30), font, 1, (255, 255, 255), 2)
            cv2.putText(canvas, 'Neural Network Input (YUV)', (display_width+30, 30), font, 1, (255, 255, 255), 2)
            
            cv2.imshow('Camera Views', canvas)
            cv2.waitKey(1)
        
    def predict_steering(self, image):
        """Make steering prediction from image"""
        with torch.no_grad():
            steering_pred = self.model(image)
            
        steering_angle = float(steering_pred.cpu().numpy()[0, 0])
        steering_angle = np.clip(steering_angle, -self.max_steering_angle, self.max_steering_angle)
        self.last_steering = steering_angle
        
        return steering_angle

    def apply_control(self, steering):
        """Apply control to vehicle with neural network steering and proportional speed control"""
        control = carla.VehicleControl()
        control.steer = steering

        current_velocity = self.vehicle.get_velocity()
        speed = 3.6 * math.sqrt(current_velocity.x**2 + current_velocity.y**2)  # km/h
        speed_error = self.target_speed - speed
        
        if speed_error > 0:
            control.throttle = min(0.3, speed_error / self.target_speed)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = min(0.3, -speed_error / self.target_speed)

        self.vehicle.apply_control(control)        
        
    def run(self):
        """Main control loop"""
        try:
            self.setup_vehicle()
            print("Vehicle and sensors initialized. Starting control loop...")

            while True:
                while not self.image_queue.empty():
                    _ = self.image_queue.get()
                
                self.world.tick()
                
                try:
                    img = self.image_queue.get(timeout=0.1)
                    processed_img = self.preprocess_image(img)
                    steering = self.predict_steering(processed_img)
                    self.apply_control(steering)
                    set_spectator_camera_following_car(self.world, self.vehicle)
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run CARLA self-driving simulation.')
    parser.add_argument('--config', type=str, default='config.json', 
                        help='Path to the configuration JSON file (default: config.json)')
    parser.add_argument('--model', type=str, default='model.pth', 
                        help='Path to the trained model file (default: model.pth)')
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        controller = CarlaSteering(config, model_path=args.model)
        controller.run()
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage:
# python 08-self-driving-from-config.py \
# --config /home/daniel/git/neurips-2025/scripts/config_640x480_segmented.json \
# --model /home/daniel/git/neurips-2025/scripts/best_steering_model_20250513-093008.pth