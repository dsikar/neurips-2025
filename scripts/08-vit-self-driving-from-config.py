import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import carla_helpers as helpers
import carla
import numpy as np
import cv2
import queue
import threading
import time
from config_utils import load_config
import argparse
from pathlib import Path
import math

#############
# SPECTATOR #
#############

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
    for i in range(len(waypoints) - 1):
        wp1 = waypoints[i]
        wp2 = waypoints[i + 1]
        world.debug.draw_line(
            wp1.transform.location,
            wp2.transform.location,
            thickness=thickness,
            color=color,
            life_time=life_time
        )

##############
# SELF-DRIVE #
##############

class CarlaSteering:
    def __init__(self, config, distance_filename='self_driving_distances.txt'):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.distance_filename = distance_filename

        # Add paths for image and prediction files
        self.image_dir = Path(config['output'].get('image_dir', '/home/daniel/dev/claude-dr/transformer-regressor/input_dir'))
        self.prediction_file = Path(config['output'].get('prediction_file', '/home/daniel/dev/claude-dr/transformer-regressor/output_dir/prediction.txt'))
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.awaiting_prediction = False
        self.frame_count = 0        

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
        
    def setup_vehicle(self, waypoint_distance=1):
        """Spawn and setup the ego vehicle with sensors"""
        sim_config = self.config['simulation']
        self.client.load_world(sim_config['town'])
        self.world = self.client.get_world()
        
        vehicle_config = self.config['vehicle']
        print("Setting waypoint distance to", waypoint_distance)
        self.waypoints = helpers.get_town04_figure8_waypoints(self.world, lane_id=-2, waypoint_distance=waypoint_distance)
        print(f"Loaded {len(self.waypoints)} waypoints with distance {waypoint_distance} meters.")
        
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
        img = np.array(image.raw_data).reshape(self.image_height, self.image_width, 4)
        img = img[:, :, :3]  # Remove alpha channel
        img = img[:, :, [2, 1, 0]]  # Convert BGR to RGB
        self.current_image = img
        
    def preprocess_image(self, img):
        """Preprocess image for external inference"""
        self.original_img = img.copy()
        
        cropped = img[self.crop_top:self.crop_bottom, :]
        resized = cv2.resize(cropped, (self.resize_width, self.resize_height))
        yuv = cv2.cvtColor(resized, cv2.COLOR_RGB2YUV)
        
        return yuv
        
    def display_images(self):
        """Display only the original camera feed"""
        if hasattr(self, 'original_img'):
            display_height = 264
            aspect_ratio = self.original_img.shape[1] / self.original_img.shape[0]
            display_width = int(display_height * aspect_ratio)
            original_resized = cv2.resize(self.original_img, (display_width, display_height))
            
            cv2.putText(original_resized, 'Original Camera Feed', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow('Camera View', original_resized)
            cv2.waitKey(1)
        
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
            
    def get_latest_prediction(self, image_filename):
        if not self.prediction_file.exists():
            print(f"{self.prediction_file} does not exist yet")
            return None
        
        with open(self.prediction_file, "r") as f:
            lines = f.readlines()
            print(f"Read {self.prediction_file}: {len(lines)} predictions found")
            if lines:
                try:
                    steering = float(lines[-1].strip())
                    print(f"Found prediction for {image_filename}: steering = {steering:.4f}")
                    self.prediction_file.unlink()
                    print(f"Cleared {self.prediction_file} after reading prediction")
                    image_path = self.image_dir / image_filename
                    if image_path.exists():
                        image_path.unlink()
                        print(f"Deleted processed image {image_filename}")
                    return steering
                except ValueError:
                    print(f"Invalid prediction format in {self.prediction_file}: {lines[-1].strip()}")
            else:
                print(f"Empty prediction file {self.prediction_file}")
        return None
        
    def run(self):
        """Main control loop with waypoint distance computation"""
        try:
            output_dir = self.config['output']['directory']
            os.makedirs(output_dir, exist_ok=True)
        except (KeyError, OSError) as e:
            print(f"Error setting up output directory: {e}")
            output_dir = None

        self_driving_distances = [0.0]
        current_wp_idx = 1

        try:
            self.setup_vehicle(waypoint_distance=self.waypoint_distance)
            print("Vehicle and sensors initialized. Starting control loop...")
            
            while current_wp_idx < len(self.waypoints):
                self.world.tick()
                
                # Process new image if available
                if self.current_image is not None:
                    img = self.current_image
                    self.current_image = None
                    
                    # Preprocess image to set self.original_img for display
                    self.preprocess_image(img)
                    
                    # Save image for external processing
                    self.frame_count += 1
                    image_filename = f"frame_{self.frame_count:06d}.jpg"
                    image_path = self.image_dir / image_filename
                    cv2.imwrite(str(image_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    print(f"Generated {image_filename} for prediction")
                    
                    if not self.awaiting_prediction:
                        self.image_queue.put((image_filename, img))
                        self.awaiting_prediction = True
                        print(f"State: Sent {image_filename} for prediction")
                
                if not self.image_queue.empty():
                    try:
                        image_filename, _ = self.image_queue.get_nowait()
                        print(f"Checking prediction for {image_filename}")
                        
                        steering = None
                        start_time = time.time()
                        while time.time() - start_time < 10.0:
                            steering = self.get_latest_prediction(image_filename)
                            if steering is not None:
                                break
                            time.sleep(0.1)
                        
                        if steering is not None:
                            self.awaiting_prediction = False
                            steering = np.clip(steering, -self.max_steering_angle, self.max_steering_angle)
                            self.last_steering = steering
                            print(f"Confirmed steering prediction: {steering:.4f}")
                            print(f"State: Prediction received, ready to send next image")
                        else:
                            steering = self.last_steering
                            print(f"No prediction received after timeout for {image_filename}, using last steering: {steering:.4f}")
                        
                        self.apply_control(steering)
                        set_spectator_camera_following_car(self.world, self.vehicle)
                        self.display_images()
                        
                        vehicle_location = self.vehicle.get_transform().location
                        distance_to_waypoint = vehicle_location.distance(self.waypoints[current_wp_idx].transform.location)
                        if distance_to_waypoint < 0.5:
                            path_distance = self.get_perpendicular_distance(vehicle_location, self.waypoints[current_wp_idx - 1], self.waypoints[current_wp_idx])
                            self_driving_distances.append(path_distance)
                            print(f"Reached waypoint {current_wp_idx + 1}/{len(self.waypoints)}, path distance: {path_distance:.4f}")
                            current_wp_idx += 1
                    
                    except queue.Empty:
                        pass
                
                else:
                    print("Warning: No new image available, applying last steering")
                    self.apply_control(self.last_steering)
                    self.display_images()  # Ensure display is called even if no new prediction
                
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
                with open(os.path.join(output_dir, self.distance_filename), 'w') as f:
                    for dist in self_driving_distances:
                        f.write(f"{dist:.4f}\n")
                print(f"Self-driving ended. Recorded {len(self_driving_distances)} distances.")
            else:
                print("Self-driving ended. No distances saved due to invalid output directory or no distances recorded.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run CARLA self-driving simulation.')
    parser.add_argument('--config', type=str, default='config_640x480_segmented_04.json', 
                        help='Path to the configuration JSON file (default: config.json)')
    parser.add_argument('--distance_filename', type=str, default='self_driving_distances.txt',
                        help='Filename for saving driving distances (default: self_driving_distances.txt)')
    parser.add_argument('--waypoint_distance', type=int, default=1,
                        help='Distance between waypoints in meters (default: 1)')
    args = parser.parse_args()

    # Display parsed arguments
    print("Parsed command-line arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    try:
        config = load_config(args.config)
        controller = CarlaSteering(config, distance_filename=args.distance_filename)
        controller.waypoint_distance = args.waypoint_distance  # Store for use in setup_vehicle
        controller.run()
    except Exception as e:
        print(f"An error occurred: {e}")

"""
Example usage:
# ClsViT3binBalanced
python 08-vit-self-driving-from-config.py \
    --config config_640x480_segmented_04.json \
    --distance_filename ClsViT3binBalanced_self_driving_distances.txt \
    --waypoint_distance 3.0

"""        