import os
import shutil

cache_dir = os.path.expanduser('~/.cache/emnist')

if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
    print("Deleted corrupted files.")
else:
    print("Cannot find folder. Ready to try again.")