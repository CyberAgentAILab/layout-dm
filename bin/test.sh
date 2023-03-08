poetry run python3 -m src.trainer.trainer.test \
    cond=<COND> \  #  choices: (unconditional, c, cwh, partial, refinement, relation)
    dataset_dir=./download/datasets \
    job_dir=<JOB_DIR> \
    result_dir=<RESULT_DIR>
