from __future__ import annotations
import cv2 
import numpy as np

PITCH_LENGTH=12000.0
PITCH_WIDTH=7000.0

PENALTY_BOX_LENGTH =1650.0
PENALTY_BOX_WIDTH = 4030.0
GOAL_BOX_LENGTH= 550.0
GOAL_BOX_WIDTH = 1830.0
CENTRE_CIRCLE_RADIUS =915.0
PENALTY_SPOT = 1100.0

COLOUR_GRASS = (58, 92, 48)
COLOUR_LINES = (180, 200, 180)
COLOUR_TEAM = [(220, 120, 40), (40, 90, 230)]   # team 0, team 1
COLOUR_REFEREE = (60, 200, 240)
COLOUR_BALL = (255, 255, 255)
COLOUR_HIGHLIGHT = (70, 255, 120)

class Radar:

    def __init__(self,scale:float = 0.06,padding:int=30):
        self.scale=scale
        self.padding = padding
        self.width=int(PITCH_LENGTH*scale) + 2 * padding
        self.height = int(PITCH_WIDTH*scale) +2 *padding
        self._background = self._draw_pitch()

     def _px(self, x_cm: float, y_cm: float) -> tuple[int, int]:
        return (
            int(x_cm * self.scale) + self.padding,
            int(y_cm * self.scale) + self.padding,
        )
 
    def _draw_pitch(self) -> np.ndarray:
        canvas = np.full((self.height, self.width, 3), COLOUR_GRASS, dtype=np.uint8)
        line = 2
 
        def rect(x1, y1, x2, y2):
            cv2.rectangle(canvas, self._px(x1, y1), self._px(x2, y2),
                          COLOUR_LINES, line)
 
        # Touchlines and halfway.
        rect(0, 0, PITCH_LENGTH, PITCH_WIDTH)
        cv2.line(canvas, self._px(PITCH_LENGTH / 2, 0),
                 self._px(PITCH_LENGTH / 2, PITCH_WIDTH), COLOUR_LINES, line)
 
        # Centre circle and spot.
        cv2.circle(canvas, self._px(PITCH_LENGTH / 2, PITCH_WIDTH / 2),
                   int(CENTRE_CIRCLE_RADIUS * self.scale), COLOUR_LINES, line)
        cv2.circle(canvas, self._px(PITCH_LENGTH / 2, PITCH_WIDTH / 2),
                   3, COLOUR_LINES, -1)
 
        # Boxes at both ends.
        for near in (True, False):
            pb_x1 = 0.0 if near else PITCH_LENGTH - PENALTY_BOX_LENGTH
            pb_x2 = PENALTY_BOX_LENGTH if near else PITCH_LENGTH
            rect(pb_x1, (PITCH_WIDTH - PENALTY_BOX_WIDTH) / 2,
                 pb_x2, (PITCH_WIDTH + PENALTY_BOX_WIDTH) / 2)
 
            gb_x1 = 0.0 if near else PITCH_LENGTH - GOAL_BOX_LENGTH
            gb_x2 = GOAL_BOX_LENGTH if near else PITCH_LENGTH
            rect(gb_x1, (PITCH_WIDTH - GOAL_BOX_WIDTH) / 2,
                 gb_x2, (PITCH_WIDTH + GOAL_BOX_WIDTH) / 2)
 
            spot_x = PENALTY_SPOT if near else PITCH_LENGTH - PENALTY_SPOT
            cv2.circle(canvas, self._px(spot_x, PITCH_WIDTH / 2), 3,
                       COLOUR_LINES, -1)
 
        return canvas
 
    # -- drawing ------------------------------------------------------------
 
    def render(
        self,
        people: list[dict],
        ball_xy: tuple[float, float] | None = None,
        possessor_track: int | None = None,
    ) -> np.ndarray:
        """
        people: dicts with keys pitch_x, pitch_y, team, class_name, track_id.
                Rows whose coordinates are NaN are skipped — a failed
                projection should vanish, not appear at the origin.
        """
        canvas = self._background.copy()
 
        for person in people:
            x, y = person.get("pitch_x"), person.get("pitch_y")
            if x is None or y is None or np.isnan(x) or np.isnan(y):
                continue
 
            team = int(person.get("team", -1))
            if person.get("class_name") == "referee" or team < 0:
                colour = COLOUR_REFEREE
            else:
                colour = COLOUR_TEAM[team % len(COLOUR_TEAM)]
 
            centre = self._px(x, y)
            cv2.circle(canvas, centre, 7, (20, 20, 20), -1)      # shadow
            cv2.circle(canvas, centre, 6, colour, -1)
 
            if possessor_track is not None and person.get("track_id") == possessor_track:
                cv2.circle(canvas, centre, 11, COLOUR_HIGHLIGHT, 2)
 
        if ball_xy is not None and not any(np.isnan(v) for v in ball_xy):
            centre = self._px(*ball_xy)
            cv2.circle(canvas, centre, 5, (20, 20, 20), -1)
            cv2.circle(canvas, centre, 4, COLOUR_BALL, -1)
 
        return canvas
 
    def with_scoreboard(
        self, radar: np.ndarray, shares: dict[int, float], timestamp: float
    ) -> np.ndarray:
        """Strip along the top with possession percentages and a clock."""
        bar_h = 34
        out = np.full((radar.shape[0] + bar_h, radar.shape[1], 3), (28, 28, 28),
                      dtype=np.uint8)
        out[bar_h:] = radar
 
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(out, f"{timestamp:6.1f}s", (10, 23), font, 0.55,
                    (230, 230, 230), 1, cv2.LINE_AA)
 
        x = 120
        for team in sorted(shares):
            cv2.circle(out, (x, 17), 7, COLOUR_TEAM[team % len(COLOUR_TEAM)], -1)
            cv2.putText(out, f"{shares[team]:.0%}", (x + 14, 23), font, 0.55,
                        (230, 230, 230), 1, cv2.LINE_AA)
            x += 90
 
        return out
    