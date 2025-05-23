# Notes for tomorrow, tuesday, 2025-05-13
# use a sliver and two fiducial markers to mark the left and right edges of the image
# use waypoints two metres apart (more jagged steering, more information content)
# Delete the current dataset - no good to us, maybe save one for reference in the report - save commit as well for reproducibility
# carla_dataset_640x480_segmented_with_fiducials/20250512_205810_792747_steering_0.0000.jpg
# -*- coding: utf-8 -*-
"""carla_figure8_selfdrive_recorder.py

Modified to use configuration file, draw permanent waypoint lines for lane segmentation, and add fiducial markers.
"""

import carla
import math
import random
import time
import os
import cv2
import numpy as np
from datetime import datetime
from config_utils import load_config

def get_figure8_waypoints(world):
    road_sequence = [42, 267, 43, 35, 861, 36, 760, 37, 1602, 38, 1091, 39, 1184, 40, 1401, 
                     41, 6, 45, 145, 46, 1072, 47, 774, 48, 901, 49, 1173, 50]
    carla_map = world.get_map()
    waypoints = carla_map.generate_waypoints(1.0)

    all_waypoints = []
    for road_id in road_sequence:
        road_waypoints = [wp for wp in waypoints if wp.road_id == road_id and wp.lane_id == -2]
        road_waypoints.sort(key=lambda x: x.s)
        all_waypoints.extend(road_waypoints)

    return all_waypoints

def set_spectator_camera_following_car(world, vehicle):
    spectator = world.get_spectator()
    vehicle_transform = vehicle.get_transform()
    forward_vector = vehicle_transform.get_forward_vector()
    offset_location = vehicle_transform.location + carla.Location(
        x=-5 * forward_vector.x,
        y=-5 * forward_vector.y,
        z=3
    )
    spectator.set_transform(carla.Transform(
        offset_location,
        carla.Rotation(pitch=-15, yaw=vehicle_transform.rotation.yaw, roll=0)
    ))
    return spectator

def compute_control(vehicle, target_wp, target_speed):
    control = carla.VehicleControl()
    current_transform = vehicle.get_transform()
    current_velocity = vehicle.get_velocity()
    speed = 3.6 * math.sqrt(current_velocity.x**2 + current_velocity.y**2)

    forward = current_transform.get_forward_vector()
    target_vector = target_wp.transform.location - current_transform.location
    forward_dot = forward.x * target_vector.x + forward.y * target_vector.y
    right_dot = -forward.y * target_vector.x + forward.x * target_vector.y
    steering = math.atan2(right_dot, forward_dot) / math.pi

    # Define the discrete steering values
    steering_values = [-0.065, -0.055, -0.045, -0.035, -0.025, -0.015, -0.005, 0.0, 0.005, 0.015, 0.025, 0.035, 0.045, 0.055, 0.065]

    # Adjust steering to nearest value in the list
    if steering < steering_values[0]:
        steering = steering_values[0]  # Cap at minimum
    elif steering > steering_values[-1]:
        steering = steering_values[-1]  # Cap at maximum
    else:
        # Find the nearest value in the list
        steering = min(steering_values, key=lambda x: abs(x - steering))

    # Clamp to CARLA's expected range [-1.0, 1.0]
    control.steer = max(-1.0, min(1.0, steering))

    speed_error = target_speed - speed
    if speed_error > 0:
        control.throttle = min(0.3, speed_error / target_speed)
        control.brake = 0.0
    else:
        control.throttle = 0.0
        control.brake = min(0.3, -speed_error / target_speed)

    return control

def setup_cameras(world, vehicle, config):
    """Setup RGB camera attached to the vehicle with parameters from config."""
    blueprint_library = world.get_blueprint_library()
    
    # Get camera parameters from config
    image_width = config['camera']['image_width']
    image_height = config['camera']['image_height']
    fov = config['camera']['fov']
    cam_pos = config['camera']['position']
    cam_rot = config['camera']['rotation']
    
    camera_transform = carla.Transform(
        carla.Location(x=cam_pos['x'], y=cam_pos['y'], z=cam_pos['z']),
        carla.Rotation(pitch=cam_rot['pitch'], yaw=cam_rot['yaw'], roll=cam_rot['roll'])
    )

    # RGB Camera
    rgb_bp = blueprint_library.find('sensor.camera.rgb')
    rgb_bp.set_attribute('image_size_x', str(image_width))
    rgb_bp.set_attribute('image_size_y', str(image_height))
    rgb_bp.set_attribute('fov', str(fov))
    rgb_camera = world.spawn_actor(rgb_bp, camera_transform, attach_to=vehicle)

    return rgb_camera

def process_images(rgb_image, config, steering_angle, image_counter=[0]):
    """Process and save the RGB image with fiducial markers, including steering data."""
    output_dir = config['output']['directory']
    os.makedirs(output_dir, exist_ok=True)

    # Process RGB image
    rgb_array = np.frombuffer(rgb_image.raw_data, dtype=np.uint8)
    rgb_array = np.reshape(rgb_array, (rgb_image.height, rgb_image.width, 4))[:, :, :3]
    rgb_array = rgb_array[:, :, [2, 1, 0]]  # Convert BGR to RGB

    # # Add fiducial markers
    # marker_size = 20
    # marker_color = [255, 0, 0]  # Red in RGB
    # offset = 90  # Pixels from left/right edges
    # bottom_y = rgb_image.height - marker_size  # Start at bottom of image

    # # Left marker
    # rgb_array[bottom_y:bottom_y + marker_size, offset:offset + marker_size, :] = marker_color

    # # Right marker
    # right_x = rgb_image.width - offset - marker_size
    # rgb_array[bottom_y:bottom_y + marker_size, right_x:right_x + marker_size, :] = marker_color

    # Save the image with steering angle
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    image_filename = f"{timestamp}_steering_{steering_angle:.4f}.jpg"
    cv2.imwrite(os.path.join(output_dir, image_filename), cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR))

    image_counter[0] += 1
    if image_counter[0] % 100 == 0:
        print(f"Recorded {image_counter[0]} images at {datetime.now().strftime('%H:%M:%S')}")

    return image_counter[0]

def get_perpendicular_distance(vehicle_location, wp1, wp2):
    """Calculate perpendicular distance from vehicle_location to line segment wp1-wp2."""
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

def drive_figure_eight(world, vehicle, waypoints, rgb_camera, config):
    output_dir = config['output']['directory']
    target_speed = config['simulation']['target_speed']
    os.makedirs(output_dir, exist_ok=True)
    image_counter = [0]
    latest_rgb = [None]
    rgb_camera.listen(lambda image: latest_rgb.__setitem__(0, image))
    ground_truth_distances = []

    start_time = datetime.now()
    try:
        # Start from waypoint 2 (index 1)
        for i in range(1, len(waypoints)):
            wp = waypoints[i]  # Target W_{i+1} (e.g., W_2 for i=1)
            print(f"Current target waypoint {i + 1}/{len(waypoints)}: {wp.transform.location}")
            world.debug.draw_point(wp.transform.location, size=0.2, color=carla.Color(255, 0, 0), life_time=5.0)
            min_path_distance = float('inf')
            while True:
                control = compute_control(vehicle, wp, target_speed)
                vehicle.apply_control(control)
                set_spectator_camera_following_car(world, vehicle)
                if latest_rgb[0] is not None:
                    process_images(latest_rgb[0], config, control.steer, image_counter)
                current_location = vehicle.get_transform().location
                # Compute perpendicular distance to W_i-W_{i+1} (e.g., W_1-W_2 for W_2)
                #path_distance = get_perpendicular_distance(current_location, waypoints[i-1], wp)
                # min_path_distance = min(min_path_distance, path_distance)
                distance_to_waypoint = current_location.distance(wp.transform.location)
                if distance_to_waypoint < 0.5:
                    path_distance = get_perpendicular_distance(current_location, waypoints[i-1], wp)
                    ground_truth_distances.append(path_distance)
                    print(f"Distance to waypoint {i + 1}: {path_distance:.4f} m")
                    break
                if world.get_settings().synchronous_mode:
                    world.tick()
                else:
                    time.sleep(0.1)
    except KeyboardInterrupt:
        print("Simulation interrupted by user.")
        control = carla.VehicleControl(throttle=0, brake=1)
        vehicle.apply_control(control)
    finally:
        end_time = datetime.now()
        rgb_camera.stop()
        with open(os.path.join(output_dir, 'ground_truth_distances_02.txt'), 'w') as f:
            for dist in ground_truth_distances:
                f.write(f"{dist:.4f}\n")
        with open(os.path.join(output_dir, 'log.txt'), 'w') as f:
            f.write(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Images Recorded: {image_counter[0]}\n")
            f.write(f"Ground Truth Distances Saved: {len(ground_truth_distances)}\n")
        print(f"Simulation ended. Recorded {image_counter[0]} images, {len(ground_truth_distances)} distances.")
    return ground_truth_distances

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

def main(config_path='config_640x480_segmented_05.json'):
    # Load configuration
    config = load_config(config_path)
    
    # Connect to CARLA server
    client = carla.Client(config['simulation']['server_host'], config['simulation']['server_port'])
    client.set_timeout(10.0)
    
    # Load specified town
    world = client.load_world(config['simulation']['town'])

    # Set simulation settings
    settings = world.get_settings()
    settings.synchronous_mode = config['simulation']['synchronous_mode']
    settings.fixed_delta_seconds = config['simulation']['fixed_delta_seconds']
    world.apply_settings(settings)

    try:
        waypoints = get_figure8_waypoints(world)
        print(f"Number of waypoints retrieved: {len(waypoints)}")

        # Draw permanent waypoint lines
        draw_permanent_waypoint_lines(world, waypoints, color=carla.Color(35, 35, 0), thickness=1.5, life_time=0)
        print("Permanent waypoint lines drawn.")

        # Spawn vehicle
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter(config['vehicle']['model'])[0]
        spawn_point = waypoints[0].transform
        spawn_point.location.z += config['vehicle']['spawn_height_offset']
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        print(f"Vehicle spawned at {spawn_point.location}")

        # Setup camera and start driving
        rgb_camera = setup_cameras(world, vehicle, config)
        drive_figure_eight(world, vehicle, waypoints, rgb_camera, config)

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'rgb_camera' in locals():
            rgb_camera.destroy()
            print("RGB Camera destroyed.")
        if 'vehicle' in locals():
            vehicle.destroy()
            print("Vehicle destroyed.")
        settings.synchronous_mode = False
        world.apply_settings(settings)

if __name__ == "__main__":
    main('config_640x480_segmented_05.json')