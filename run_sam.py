import argparse
import concurrent.futures
import glob
import os
import random
import shutil
import time

import numpy as np
from watchdog import events
from watchdog import observers


class SAMEmulatorFileCopier(events.FileSystemEventHandler):
  _DELAY_MIN = 60 # seconds
  _DELAY_MAX = 120 # seconds
  _NUM_THREADS = 4 # Expected number of GPUS for the real thing.

  def __init__(self, preprocessed_directory, postprocessed_directory):
    self.preprocessed_directory = preprocessed_directory
    self.postprocessed_directory = postprocessed_directory
    self.executor = concurrent.futures.ThreadPoolExecutor(self._NUM_THREADS)

  def _get_all_files_in_dir(self, dir):
    return [
      os.path.basename(path) for path in glob.glob(os.path.join(dir, '*.npz'))]

  def _copy_file_with_delay(self, src, dest):
    time.sleep(random.uniform(self._DELAY_MIN, self._DELAY_MAX))
    shutil.copy(src, dest + '.tmp')
    shutil.move(dest + '.tmp', dest)

  def initial_load(self):
    preprocessed_files = set(self._get_all_files_in_dir(
      self.preprocessed_directory))
    postprocessed_files = set(self._get_all_files_in_dir(
      self.postprocessed_directory))
    unprocessed_files = preprocessed_files - postprocessed_files
    tasks = []
    for file in unprocessed_files:
      src_path = os.path.join(self.preprocessed_directory, file)
      dst_path = os.path.join(self.postprocessed_directory, file)
      print(f'enqueuing {file}')
      tasks.append(self.executor.submit(
        self._copy_file_with_delay, src_path, dst_path))

    for task in tasks:
      task.result()

  def on_created(self, event):
    assert isinstance(event, events.FileCreatedEvent)
    file_name = os.path.basename(event.src_path)
    print(f'enqueuing {file_name}')
    self.executor.submit(
      self._copy_file_with_delay, event.src_path,
      os.path.join(self.postprocessed_directory, file_name))


def main():
  parser = argparse.ArgumentParser(
    description='Use SAM to create postprocessed version of saved chunks '
                'from DreamerV3 with additional segmentation masks for '
                'the images.')
  parser.add_argument(
    '--logdir', required=True, help='Directory for log files')

  args = parser.parse_args()

  # Check if logdir exists
  if not os.path.exists(args.logdir):
    print(f"The specified log directory '{args.logdir}' does not exist.")
    return

  # Create paths for preprocessed and postprocessed replay directories
  preprocessed_replay_path = os.path.join(args.logdir, 'preprocessed_replay')
  postprocessed_replay_path = os.path.join(args.logdir,
                                           'postprocessed_replay')

  if not os.path.exists(preprocessed_replay_path):
    print(f'Specified preprocessed directory does not exist.')
    return

  if not os.path.exists(postprocessed_replay_path):
    print(f'Specified postprocessed directory does not exist.')
    return

  sam_handler = SAMEmulatorFileCopier(
      preprocessed_replay_path, postprocessed_replay_path)
  sam_handler.initial_load()
  observer = observers.Observer()
  observer.schedule(
      sam_handler, path=preprocessed_replay_path, recursive=False)
  observer.start()
  try:
    while True:
      time.sleep(1)
  finally:
    observer.stop()
    observer.join()


if __name__ == '__main__':
  main()
