# CNN self-driving car from config, without fiducial markers
# Mods: 
# 1. Add road segmentation
# 2. Remove fiducial markers
# 3. Add dynamic steering bins based on --bins argument
# 4. Add CPU-based pepper noise in preprocess_image

#########
# MODEL #
#########

## Helper functions
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import time  # Add this import at the top of the file with other imports

# Add the src/ directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import carla_helpers as helpers

class NVIDIANet(nn.Module):
    def __init__(self, num_outputs=15, dropout_rate=0.1):
        super(NVIDIANet, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(24, 32, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(32, 48, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3)
        
        # Dropout layer
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # Dense layers (adjustable final layer based on num_outputs)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1152, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, num_outputs)  # Dynamic output size
        
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
        x = self.fc4(x)  # Output logits for num_outputs classes
        
        # Apply softmax to get probabilities
        x = F.softmax(x, dim=1)
        
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
    """
    for i in range(len(waypoints) - 1):
        wp1 = waypoints[i]
        wp2 = waypoints[i + 1]
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
    def __init__(self, config, model_path='model.pth', distances_file='self_driving_distances_05.txt', bins=15, noise_type=None, intensity=None, network='cnn', dataset_balance='balanced'):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.distances_file = distances_file
        self.bins = bins
        self.noise_type = noise_type  # Added for noise support
        self.intensity = intensity    # Added for noise intensity
        self.network = network
        # self.distances_file = f"self_driving_distances_{self.bins}_bins_{'no_noise' if self.noise_type is None else self.noise_type}_{int(self.intensity) if self.intensity is not None else 0}_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        self.distances_file = f"self_driving_{bins}_{network}_{dataset_balance}_{'no' if noise_type is None else noise_type[:3]}_{int(intensity) if intensity is not None else 0}_{time.strftime('%y%m%d_%H%M')}.txt"
        # Define steering values based on bins
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
        
        # Initialize model with dynamic number of outputs
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
        
        self.image_queue = queue.Queue()
        self.current_image = None
        from distance_metrics import DistanceMetrics
        self.dm = DistanceMetrics(num_bins=256, val_range=(0, 255))  # Initialize DistanceMetrics
        self.pre_noise_rgb = None  # Store current pre-noise RGB image
        self.post_noise_rgb = None  # Store current post-noise RGB image
        self.prediction_data = []  # Store (softmax, class, bhattacharyya, hist_inter, kl_div)
        
        # Add paths for image and prediction files from config or defaults
        self.image_dir = Path(config['output'].get('image_dir', '/home/daniel/dev/claude-dr/transformer-regressor/input_dir'))
        self.prediction_file = Path(config['output'].get('prediction_file', '/home/daniel/dev/claude-dr/transformer-regressor/output_dir/prediction.npy'))
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.frame_count = 0
                
    def setup_vehicle(self):
        """Spawn and setup the ego vehicle with sensors"""
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
        """
        Callback to process images from CARLA camera
        Convert image from BGR to RGB
        """
        img = np.array(image.raw_data).reshape(self.image_height, self.image_width, 4)
        img = img[:, :, :3]  # Remove alpha channel
        img = img[:, :, [2, 1, 0]]  # Convert BGR to RGB
        
        self.image_queue.put(img)
        
    def preprocess_image(self, img):
        """Preprocess image and apply noise, store RGB images"""
        self.original_img = img.copy()
        # do not resize or crop
        # cropped = img[self.crop_top:self.crop_bottom, :]
        #resized = cv2.resize(cropped, (self.resize_width, self.resize_height))
        self.pre_noise_rgb = img.copy() # resized.copy()  # Store pre-noise RGB
        resized = img.copy()
        
        # Apply CPU-based pepper noise to RGB
        if hasattr(self, 'noise_type') and self.noise_type and hasattr(self, 'intensity') and self.intensity is not None:
            if self.noise_type == 'pepper_noise':
                noise_prob = self.intensity * 0.01  # 1%-100% of pixels
                mask = np.random.random(resized.shape) < noise_prob
                noisy_rgb = resized.copy()
                noisy_rgb[mask] = np.random.randint(0, 2, mask.sum()) * 255  # 0 or 255
                self.post_noise_rgb = noisy_rgb
        else:
            self.post_noise_rgb = self.pre_noise_rgb.copy()  # No noise case
        
        # Copy both RGB images to input_dir (optional: only post-noise if ViT handles pre-noise)
        self.frame_count += 1
        image_filename = f"frame_{self.frame_count:06d}.jpg"
        cv2.imwrite(str(self.image_dir / image_filename), cv2.cvtColor(self.post_noise_rgb, cv2.COLOR_RGB2BGR))
        print(f"Copied {image_filename} to {self.image_dir} for prediction")
        
        return self.post_noise_rgb  # Return for consistency, though not used for prediction
        
    def display_images(self):
        """Display pre-noise and post-noise RGB images side by side"""
        if hasattr(self, 'pre_noise_rgb') and hasattr(self, 'post_noise_rgb'):
            display_height = 264
            aspect_ratio = self.pre_noise_rgb.shape[1] / self.pre_noise_rgb.shape[0]
            display_width = int(display_height * aspect_ratio)
            pre_noise_resized = cv2.resize(self.pre_noise_rgb, (display_width, display_height))
            post_noise_resized = cv2.resize(self.post_noise_rgb, (display_width, display_height))
            
            canvas_width = display_width * 2 + 20
            canvas = np.zeros((display_height, canvas_width, 3), dtype=np.uint8)
            
            canvas[:, :display_width] = pre_noise_resized
            canvas[:, display_width + 20:] = post_noise_resized
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            label_height = int(display_height * 0.1)
            cv2.putText(canvas, 'Pre-Noise RGB', (10, label_height), font, 1, (255, 255, 255), 2)
            cv2.putText(canvas, 'Post-Noise RGB', (display_width + 30, label_height), font, 1, (255, 255, 255), 2)
            
            cv2.imshow(f'Camera Views - {self.network} {self.bins} bins', canvas)
            cv2.waitKey(1)
        else:
            print("Warning: Images not available for display")
        
    # def predict_steering(self, image):
    #     """Make steering prediction from image"""
    #     with torch.no_grad():
    #         class_probs = self.model(image)  # Shape: (1, num_outputs)
    #         predicted_class = torch.argmax(class_probs, dim=1).item()  # Get index of highest probability
    #         steering_angle = self.steering_values[self.bins][predicted_class]  # Map index to steering value
            
    #     steering_angle = np.clip(steering_angle, -self.max_steering_angle, self.max_steering_angle)
    #     self.last_steering = steering_angle
        
    #     # Compute distances if images are available
    #     if self.pre_noise_yuv is not None and self.post_noise_yuv is not None:
    #         distances = self.dm.get_distances(self.pre_noise_yuv, self.post_noise_yuv)
    #         bhattacharyya = distances['Bhattacharyya']
    #         hist_intersection = distances['HistogramIntersection']
    #         kl_divergence = distances['KLDivergence']
            
    #         # Store prediction data
    #         self.prediction_data.append((class_probs.cpu().numpy(), predicted_class, bhattacharyya, hist_intersection, kl_divergence))
        
    #     return steering_angle

    def predict_steering(self, image):
        """Monitor prediction.npy for softmax and steering angle from ViT, compute distances"""
        image_filename = f"frame_{self.frame_count:06d}.jpg"
        steering = self.last_steering  # Default to last steering if no new prediction
        softmax_output = None
        predicted_class = None
        start_time = time.time()
        while time.time() - start_time < 10.0:
            if self.prediction_file.exists():
                try:
                    # Load the .npy file containing softmax output (assuming shape [1, num_bins])
                    prediction_data = np.load(self.prediction_file, allow_pickle=True)
                    if prediction_data.size > 0:
                        softmax_output = prediction_data[0] if prediction_data.ndim > 1 else prediction_data  # Adjust based on ViT output
                        
                        if len(softmax_output) != self.bins:
                            raise ValueError(
                                f"Softmax output size mismatch. Expected {self.bins} bins, "
                                f"but received {len(softmax_output)} elements from {self.prediction_file}."
                            )
                                            
                        predicted_class = np.argmax(softmax_output).item()
                        steering = self.steering_values[self.bins][predicted_class]
                        self.prediction_file.unlink()  # Delete after reading
                        print(f"Received prediction for {image_filename}: softmax={softmax_output}, steering={steering:.4f}")
                        break
                except Exception as e:
                    print(f"Error loading {self.prediction_file}: {e}")
            time.sleep(0.1)
        
        # Compute distances if images are available
        bhattacharyya = hist_intersection = kl_divergence = 0.0
        if self.pre_noise_rgb is not None and self.post_noise_rgb is not None:
            distances = self.dm.get_distances(self.pre_noise_rgb, self.post_noise_rgb)
            bhattacharyya = distances['Bhattacharyya']
            hist_intersection = distances['HistogramIntersection']
            kl_divergence = distances['KLDivergence']
        
        # Store prediction data only if a valid prediction was received
        if softmax_output is not None and predicted_class is not None:
            self.prediction_data.append((softmax_output, predicted_class, bhattacharyya, hist_intersection, kl_divergence))
        
        # steering = np.clip(steering, -self.max_steering_angle, self.max_steering_angle)
        self.last_steering = steering
        return steering

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

    def find_nearest_waypoint(self, vehicle_location, waypoints):
        """Find the index of the nearest waypoint."""
        min_dist = float('inf')
        nearest_idx = 0
        for i, wp in enumerate(waypoints):
            dist = vehicle_location.distance(wp.transform.location)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
        return nearest_idx, min_dist
    
    # def get_perpendicular_distance(self, vehicle_location, wp1, wp2):
    #     """Calculate perpendicular distance from vehicle_location to line segment wp1-wp2."""
    #     p = np.array([vehicle_location.x, vehicle_location.y])
    #     a = np.array([wp1.transform.location.x, wp1.transform.location.y])
    #     b = np.array([wp2.transform.location.x, wp2.transform.location.y])
    #     ab = b - a
    #     ap = p - a
    #     ab_norm = np.dot(ab, ab)
    #     if ab_norm == 0:
    #         return np.linalg.norm(p - a)
    #     t = np.dot(ap, ab) / ab_norm
    #     t = max(0, min(1, t))
    #     closest = a + t * ab
    #     return np.linalg.norm(p - closest)

    def get_perpendicular_distance(self, vehicle_location, wp1, wp2):
        """Calculate perpendicular distance from vehicle_location to line segment wp1-wp2."""
        p = np.array([vehicle_location.x, vehicle_location.y])
        a = np.array([wp1.transform.location.x, wp1.transform.location.y])
        b = np.array([wp2.transform.location.x, wp2.transform.location.y])
        ab = b - a
        ap = p - a
        ab_norm = np.dot(ab, ab)
        
        # Debug print to track positions and segment length
        print(f"Vehicle: ({p[0]:.2f}, {p[1]:.2f}), WP1: ({a[0]:.2f}, {a[1]:.2f}), WP2: ({b[0]:.2f}, {b[1]:.2f}), ab_norm: {ab_norm:.2f}, "
            f"Segment Length: {np.sqrt(ab_norm):.2f}m")
        
        if ab_norm < 1.0:  # Skip if segment is too short
            print(f"Warning: Short segment detected (length {np.sqrt(ab_norm):.2f}m), using distance to WP1")
            return np.linalg.norm(p - a)
        
        t = np.dot(ap, ab) / ab_norm
        t = max(0, min(1, t))  # Clamp t to segment bounds
        closest = a + t * ab
        distance = np.linalg.norm(p - closest)
        
        return distance

    def run(self):
        """Main control loop with waypoint distance computation, exiting after all waypoints."""
        try:
            output_dir = self.config['output']['directory']
            os.makedirs(output_dir, exist_ok=True)
        except (KeyError, OSError) as e:
            print(f"Error setting up output directory: {e}")
            output_dir = None

        self_driving_distances = [0.0]  # W_1 distance
        current_wp_idx = 1  # Start at W_2

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
                    self.display_images()
                    
                    # vehicle_location = self.vehicle.get_transform().location
                    # distance_to_waypoint = vehicle_location.distance(self.waypoints[current_wp_idx].transform.location)
                    # if distance_to_waypoint < 2.0:
                    #     path_distance = self.get_perpendicular_distance(vehicle_location, self.waypoints[current_wp_idx - 1], self.waypoints[current_wp_idx])
                    #     self_driving_distances.append(path_distance)
                    #     print(f"Reached waypoint {current_wp_idx + 1}/{len(self.waypoints)}, path distance: {path_distance:.4f}")
                    #     current_wp_idx += 1

                    vehicle_location = self.vehicle.get_transform().location
                    distance_to_waypoint = vehicle_location.distance(self.waypoints[current_wp_idx].transform.location)
                    print(f"Distance to waypoint {current_wp_idx}: {distance_to_waypoint:.4f}")
                    if distance_to_waypoint < 1.5:
                        segment_length = np.sqrt((self.waypoints[current_wp_idx].transform.location.x - self.waypoints[current_wp_idx - 1].transform.location.x)**2 +
                                                (self.waypoints[current_wp_idx].transform.location.y - self.waypoints[current_wp_idx - 1].transform.location.y)**2)
                        if segment_length > 1.0:
                            path_distance = self.get_perpendicular_distance(vehicle_location, self.waypoints[current_wp_idx - 1], self.waypoints[current_wp_idx])
                        else:
                            path_distance = self_driving_distances[-1]
                            print(f"Warning: Segment length {segment_length:.2f}m <= 1.0m, using previous path distance")
                        self_driving_distances.append(path_distance)  # Append before invasion check
                        print(f"Reached waypoint {current_wp_idx + 1}/{len(self.waypoints)}, path distance: {path_distance:.4f}")
                        # Failsafe for lane invasion
                        if path_distance > 0.85:
                            log_filename = os.path.splitext(self.distances_file)[0] + '.log'
                            with open(log_filename, 'a') as log_file:
                                log_file.write(f"Lane invasion detected at waypoint {current_wp_idx + 1} with path distance {path_distance:.4f} at {time.strftime('%Y%m%d_%H%M%S')}\n")
                            print(f"Lane invasion detected at waypoint {current_wp_idx + 1}, stopping simulation...")
                            # Save distances and prediction data before terminating
                            if output_dir and self_driving_distances:
                                with open(os.path.join(output_dir, self.distances_file), 'w') as f:
                                    for dist in self_driving_distances:
                                        f.write(f"{dist:.4f}\n")
                                print(f"Saved {len(self_driving_distances)} distances to {os.path.join(output_dir, self.distances_file)}")
                            if self.prediction_data:
                                npy_filename = os.path.splitext(self.distances_file)[0] + '.npy'
                                np.save(os.path.join(output_dir, npy_filename), self.prediction_data)
                                print(f"Saved prediction data to {os.path.join(output_dir, npy_filename)}")
                            raise Exception("Lane invasion detected, simulation terminated")
                        current_wp_idx += 1                   
                
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

        # Save prediction data to .npy file
        if self.prediction_data:
            npy_filename = os.path.splitext(self.distances_file)[0] + '.npy'
            np.save(os.path.join(output_dir, npy_filename), self.prediction_data)
            print(f"Saved prediction data to {os.path.join(output_dir, npy_filename)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run CARLA self-driving simulation.')
    parser.add_argument('--config', type=str, default='config.json', 
                        help='Path to the configuration JSON file (default: config.json)')
    parser.add_argument('--model', type=str, default='model.pth', 
                        help='Path to the trained model file (default: model.pth)')
    parser.add_argument('--distance_filename', type=str, default='self_driving_distances_05.txt', 
                        help='Path to the distances file (default: self_driving_distances_05.txt)')      
    parser.add_argument('--bins', type=int, choices=[3, 5, 15], default=15,
                        help='Number of steering bins (3, 5, or 15) (default: 15)')
    # Add noise-related arguments
    parser.add_argument('--noise_type', type=str, choices=['pepper_noise'], default=None,
                        help='Type of noise to apply (default: None)')
    parser.add_argument('--intensity', type=int, choices=range(0, 100), default=None,
                        help='Intensity level of noise (1-10) (default: None)')
    parser.add_argument('--network', type=str, choices=['cnn', 'vit'], default='cnn',
                        help='Network type (cnn or vit) (default: cnn)')
    parser.add_argument('--dataset_balance', type=str, choices=['balanced', 'unbalanced'], default='balanced',
                        help='Dataset balance (balanced or unbalanced) (default: balanced)')    
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        controller = CarlaSteering(config, model_path=args.model, distances_file=args.distance_filename, 
                                 bins=args.bins, network=args.network, dataset_balance=args.dataset_balance, noise_type=args.noise_type, intensity=args.intensity)
        controller.run()
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage:
# ClsViT5binBalanced pepper_noise 1%
# python 08-vit-self-driving-from-config-noise.py \
# --config /home/daniel/git/neurips-2025/scripts/config_640x480_segmented_06.json \
# --model /home/daniel/git/neurips-2025/scripts/best_quantized_steering_model_5_bins_balanced_20250529-211142.pth \
# --bins 5 \
# --noise_type pepper_noise \
# --intensity 1 \
# --network vit \
# --dataset_balance balanced

# For 15 bins
# python 08-self-driving-from-config_05_15_bins.py \
# --config /home/daniel/git/neurips-2025/scripts/config_640x480_segmented_05.json \
# --model /home/daniel/git/neurips-2025/scripts/best_quantized_steering_model_15_bins_balanced_20250529-211117.pth \
# --distance_filename ClsCNN15binBalanced_self_driving_distances.txt \
# --bins 15

# Example usage:
# For 5 bins with pepper noise at intensity 2
# python 08-self-driving-from-config-noise.py \
# --config /home/daniel/git/neurips-2025/scripts/config_640x480_segmented_06.json \
# --model /home/daniel/git/neurips-2025/scripts/best_quantized_steering_model_5_bins_balanced_20250529-211142.pth \
# --bins 5 \
# --noise_type pepper_noise \
# --intensity 2

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