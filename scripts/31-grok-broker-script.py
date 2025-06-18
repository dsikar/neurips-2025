import os
import time
import subprocess
import logging
import tempfile
import shutil
import numpy as np

# --- Configuration ---
LOCAL_INPUT_DIR = os.path.expanduser("~/git/neurips-2025/qwen/input_dir")
LOCAL_OUTPUT_DIR = os.path.expanduser("~/git/neurips-2025/qwen/output_dir")
REMOTE_SCP_USER = "aczd097"
REMOTE_SCP_HOST = "localhost"
REMOTE_SCP_PORT = "2001"
REMOTE_INPUT_ARCHIVE_PATH = "~/archive/git/neurips-2025/qwen/input_dir/"
REMOTE_OUTPUT_ARCHIVE_PATH = "~/archive/git/neurips-2025/qwen/output_dir/"
SSH_KEY_PATH = os.path.expanduser("~/.ssh/id_alcescluster")
POLLING_INTERVAL = 0.1

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def transfer_and_delete_file(file_path):
    """
    Constructs and executes the scp command to transfer a file from local to remote,
    then deletes the local file if the transfer is successful.
    """
    local_filename = os.path.basename(file_path)
    remote_destination = f"{REMOTE_SCP_USER}@{REMOTE_SCP_HOST}:{REMOTE_INPUT_ARCHIVE_PATH}{local_filename}"
    scp_upload_command = [
        "scp", "-P", REMOTE_SCP_PORT, "-i", SSH_KEY_PATH, file_path, remote_destination
    ]
    logging.info(f"Attempting to transfer: {file_path} to {remote_destination}")
    try:
        result = subprocess.run(scp_upload_command, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            logging.info(f"Successfully transferred: {local_filename}")
            os.remove(file_path)
            logging.info(f"Deleted local file: {local_filename}")
        else:
            logging.error(f"SCP upload failed for {local_filename}. Return code: {result.returncode}")
            logging.error(f"STDOUT: {result.stdout.strip()}")
            logging.error(f"STDERR: {result.stderr.strip()}")
    except FileNotFoundError:
        logging.error(f"Error: 'scp' command not found.")
    except Exception as e:
        logging.error(f"An unexpected error occurred during transfer of {local_filename}: {e}")

def download_file_from_remote(remote_filename, max_retries=5, initial_delay=0.5):
    """
    Constructs and executes the scp command to download a file from remote to local.
    Retries the download if np.load fails with a pickle error.
    """
    remote_source = f"{REMOTE_SCP_USER}@{REMOTE_SCP_HOST}:{REMOTE_OUTPUT_ARCHIVE_PATH}{remote_filename}"
    # Use a temporary file to avoid partial reads
    with tempfile.NamedTemporaryFile(delete=False, dir=LOCAL_OUTPUT_DIR, prefix="temp_", suffix=".npy") as temp_file:
        temp_destination = temp_file.name
    local_destination = os.path.join(LOCAL_OUTPUT_DIR, remote_filename)

    for attempt in range(max_retries):
        # Build the scp command for downloading
        scp_download_command = [
            "scp", "-P", REMOTE_SCP_PORT, "-i", SSH_KEY_PATH, remote_source, temp_destination
        ]
        logging.info(f"Attempt {attempt + 1}/{max_retries} to download: {remote_source} to {temp_destination}")
        
        try:
            # Execute SCP download
            result = subprocess.run(scp_download_command, capture_output=True, text=True, check=False)
            if result.returncode != 0:
                logging.error(f"SCP download failed for {remote_filename}. Return code: {result.returncode}")
                logging.error(f"STDOUT: {result.stdout.strip()}")
                logging.error(f"STDERR: {result.stderr.strip()}")
                if os.path.exists(temp_destination):
                    os.remove(temp_destination)
                if attempt < max_retries - 1:
                    delay = initial_delay * (2 ** attempt)
                    logging.info(f"Retrying download after {delay:.2f} seconds...")
                    time.sleep(delay)
                continue

            # Try loading the file to verify it's valid
            try:
                np.load(temp_destination, allow_pickle=True)
                # File is valid, move it to the final destination
                shutil.move(temp_destination, local_destination)
                logging.info(f"Successfully downloaded and verified: {remote_filename}")
                return  # Success, exit the function
            except Exception as e:
                logging.warning(f"Failed to load {temp_destination}: {e}")
                if os.path.exists(temp_destination):
                    os.remove(temp_destination)
                if attempt < max_retries - 1:
                    delay = initial_delay * (2 ** attempt)
                    logging.info(f"Retrying download after {delay:.2f} seconds...")
                    time.sleep(delay)
                continue

        except FileNotFoundError:
            logging.error(f"Error: 'scp' command not found.")
            if os.path.exists(temp_destination):
                os.remove(temp_destination)
            break
        except Exception as e:
            logging.error(f"Unexpected error during download of {remote_filename}: {e}")
            if os.path.exists(temp_destination):
                os.remove(temp_destination)
            break

    logging.error(f"Failed to download and verify {remote_filename} after {max_retries} attempts.")

def ensure_local_directory_exists(path):
    """Ensures the local directory exists."""
    os.makedirs(path, exist_ok=True)
    logging.info(f"Ensured local directory exists: {path}")

def ensure_remote_directory_exists(remote_path):
    """
    Ensures a specific remote directory exists by attempting to create it via SSH.
    """
    ssh_command = [
        "ssh", "-p", REMOTE_SCP_PORT, "-i", SSH_KEY_PATH,
        f"{REMOTE_SCP_USER}@{REMOTE_SCP_HOST}", f"mkdir -p {remote_path}"
    ]
    logging.info(f"Attempting to ensure remote directory exists: {remote_path}")
    try:
        result = subprocess.run(ssh_command, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            logging.info(f"Remote directory {remote_path} is confirmed to exist.")
            return True
        else:
            logging.error(f"Failed to confirm/create remote directory {remote_path}. Return code: {result.returncode}")
            logging.error(f"STDOUT: {result.stdout.strip()}")
            logging.error(f"STDERR: {result.stderr.strip()}")
            return False
    except FileNotFoundError:
        logging.error(f"Error: 'ssh' command not found.")
        exit(1)
    except Exception as e:
        logging.error(f"Unexpected error during remote directory check for {remote_path}: {e}")
        exit(1)

def main():
    """
    Main loop to monitor the local input directory and process new files.
    """
    ensure_local_directory_exists(LOCAL_INPUT_DIR)
    ensure_local_directory_exists(LOCAL_OUTPUT_DIR)
    if not ensure_remote_directory_exists(REMOTE_INPUT_ARCHIVE_PATH):
        logging.critical(f"Aborting: Cannot ensure remote input directory {REMOTE_INPUT_ARCHIVE_PATH} exists.")
        exit(1)
    if not ensure_remote_directory_exists(REMOTE_OUTPUT_ARCHIVE_PATH):
        logging.critical(f"Aborting: Cannot ensure remote output directory {REMOTE_OUTPUT_ARCHIVE_PATH} exists.")
        exit(1)

    processed_upload_files = set()
    processed_download_files = set()
    logging.info(f"Starting to monitor: {LOCAL_INPUT_DIR} (for uploads) and {REMOTE_OUTPUT_ARCHIVE_PATH} (for downloads)")
    try:
        while True:
            # Upload local files
            current_local_input_files = set()
            for filename in os.listdir(LOCAL_INPUT_DIR):
                file_path = os.path.join(LOCAL_INPUT_DIR, filename)
                if os.path.isfile(file_path):
                    current_local_input_files.add(file_path)
            new_files_to_upload = current_local_input_files - processed_upload_files
            for file_path in new_files_to_upload:
                time.sleep(POLLING_INTERVAL)
                transfer_and_delete_file(file_path)
                processed_upload_files.add(file_path)
            processed_upload_files = {f for f in processed_upload_files if os.path.exists(f)}
            # Download remote files
            target_remote_download_file = "prediction.npy"
            try:
                ssh_list_command = [
                    "ssh", "-p", REMOTE_SCP_PORT, "-i", SSH_KEY_PATH,
                    f"{REMOTE_SCP_USER}@{REMOTE_SCP_HOST}", f"ls -1 {REMOTE_OUTPUT_ARCHIVE_PATH}"
                ]
                list_result = subprocess.run(ssh_list_command, capture_output=True, text=True, check=True)
                remote_files = set(list_result.stdout.strip().splitlines())
                if target_remote_download_file in remote_files and \
                   (target_remote_download_file not in processed_download_files or \
                    not os.path.exists(os.path.join(LOCAL_OUTPUT_DIR, target_remote_download_file))):
                    logging.info(f"Detected new or missing remote file for download: {target_remote_download_file}")
                    download_file_from_remote(target_remote_download_file)
                    processed_download_files.add(target_remote_download_file)
            except subprocess.CalledProcessError as e:
                logging.error(f"Failed to list remote output directory: {e}")
            except Exception as e:
                logging.error(f"Unexpected error during remote file listing: {e}")
            time.sleep(POLLING_INTERVAL)
    except KeyboardInterrupt:
        logging.info("Script stopped by user (Ctrl+C).")
    except Exception as e:
        logging.critical(f"A critical error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    main()
