from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.events.possession import NO_POSSESSION, possession_share
from src.utils.video import count_sampled_frames, sampled_frames
from src.viz.radar import COLOUR_BALL, COLOUR_REFEREE, COLOUR_TEAM, Radar

