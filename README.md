# Flappy Bird Reinforcement Learning

Reinforcement learning assignment for `3MD3220`, comparing:

- Monte Carlo control
- Sarsa(lambda)

The project is built around the compact `TextFlappyBird-v0` environment and includes:

- a submission notebook: `flappy_bird.ipynb`
- a reusable Python module: `flappy_bird.py`
- a local copy of the `text_flappy_bird_gym` environment package for reproducibility
- a compact 3-page report in `report/report.pdf`

## Repository structure

- `flappy_bird.ipynb`: main notebook used for the assignment
- `flappy_bird.py`: training, evaluation, plotting, and helper functions
- `text_flappy_bird_gym/`: local environment package used by the notebook and scripts
- `build_submission_assets.py`: regenerates report figures and metrics
- `report/report.tex`: LaTeX source of the report
- `report/report.pdf`: compiled report

## Quick start

```bash
py -3 -m pip install -r requirements.txt
```

Then open and run:

- `flappy_bird.ipynb`
