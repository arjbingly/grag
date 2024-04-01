import os
import shutil
from pathlib import Path

jenkins_home = os.getenv('JENKINS_HOME')

lock_path = Path(jenkins_home) / 'ci_test_data/data/vectordb/ci_test/dataset_lock.lock'

if os.path.exists(lock_path):
    shutil.rmtree(lock_path)
    print('Deleting lock file: {}'.format(lock_path))
