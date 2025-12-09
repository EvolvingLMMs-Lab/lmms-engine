EditReward Service
==================

Files:
- editreward_scorer.py  : Core model and scorer class
- editreward.py         : Model loader for server
- app_editreward.py     : Flask server
- gunicorn_editreward.conf.py : Gunicorn config
- test_editreward.py    : Test client

Environment Variables (Required):
- EDITREWARD_MODEL_PATH      : Path to base model (e.g., MiMo-VL-7B-SFT-2508)
- EDITREWARD_CHECKPOINT_PATH : Path to checkpoint (e.g., EditReward-MiMo-VL-7B-SFT-2508)

Environment Variables (Optional):
- EDITREWARD_PORT        : Server port (default: 18087)
- EDITREWARD_NUM_WORKERS : Gunicorn workers (default: 8)
- EDITREWARD_TEST_IMAGES : Test images directory for testing

Usage:

1. Start server:
   export EDITREWARD_MODEL_PATH=/path/to/base_model
   export EDITREWARD_CHECKPOINT_PATH=/path/to/checkpoint
   cd .../editreward_service
   python app_editreward.py  # Debug mode
   # or
   gunicorn -c gunicorn_editreward.conf.py 'app_editreward:create_app()'

2. Test server:
   export EDITREWARD_TEST_IMAGES=/path/to/test/images
   python test_editreward.py

3. Direct Python usage:
   from lmms_engine.utils.rewards.editreward_service import EditRewardScorer
   scorer = EditRewardScorer(
       checkpoint_path="/path/to/checkpoint",
       model_name_or_path="/path/to/base_model",
   )
   scores = scorer.score(prompts, source_images, edited_images)
