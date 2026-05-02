# Drive Run Layout

Recommended Drive root:
- /content/drive/MyDrive/mitosis_detection

Data:
- data/raw_tif/
- data/mitosis_events.csv

Runs:
- runs/<project_name>/<YYYYMMDD>/<run_name>/
  - checkpoints/   (created by train.py)
  - logs/          (created by train.py)
  - meta/          (created by scripts/colab_run.py)
    - git_commit.txt
    - git_branch.txt
    - git_remote.txt
    - config_used.yaml
    - pip_freeze.txt
    - command.txt
  - outputs/       (reserved for inference results)