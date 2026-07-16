import cv2 
import numpy as np
import random
import math
from datetime import datetime, timedelta

WIDTH = 640
HEIGHT = 480

FPS = 30
DURATION = 5

TOTAL_FRAMES = FPS * DURATION

OUTPUT_VIDEO = "data/raw_video/cctv_video.mp4"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    FPS,
    (WIDTH, HEIGHT)
)

""" 
This FUnction will create Wall

-----------------

Floor
"""

def draw_background(frame):
    frame[:] = (205, 205, 205)

    cv2.rectangle(
        frame,
        (0, 350),
        (640, 480),
        (170, 170, 170),
        -1
    )

    cv2.line(
        frame, 
        (0, 350),
        (640, 480),
        (120, 120, 120),
        2
    )

    return frame


def draw_cctv_overlay(frame, frame_number):

    timestamp = (
        datetime(2026, 7, 15, 12, 0, 0) + timedelta(
            seconds=frame_number/FPS
            )
    )

    cv2.putText(
        frame,
        timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        (15, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        2
    )

    cv2.putText(
        frame,
        "CAMERA 01",
        (500, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 255),
        2
    )


def draw_person(frame, x, y, activity, frame_number):

    # Shadow
    cv2.ellipse(
        frame,
        (x, y + 52),
        (18, 6),
        0,
        0,
        360,
        (120,120,120),
        -1
    )

    # Head
    cv2.circle(
        frame,
        (x, y),
        10,
        (30,30,30),
        -1
    )

    # Body
    cv2.line(
        frame,
        (x, y+10),
        (x, y+40),
        (30,30,30),
        3
    )

    arm = 10 * math.sin(frame_number * 0.3)
    leg = 12 * math.sin(frame_number * 0.3)

    if activity == "Walking":

        cv2.line(frame,(x,y+20),(x-12,int(y+25+arm)),(30,30,30),3)
        cv2.line(frame,(x,y+20),(x+12,int(y+25-arm)),(30,30,30),3)

        cv2.line(frame,(x,y+40),(x-10,int(y+60+leg)),(30,30,30),3)
        cv2.line(frame,(x,y+40),(x+10,int(y+60-leg)),(30,30,30),3)

    elif activity == "Running":

        arm = 16 * math.sin(frame_number*0.6)
        leg = 18 * math.sin(frame_number*0.6)

        cv2.line(frame,(x,y+20),(x-16,int(y+18+arm)),(30,30,30),3)
        cv2.line(frame,(x,y+20),(x+16,int(y+18-arm)),(30,30,30),3)

        cv2.line(frame,(x,y+40),(x-14,int(y+64+leg)),(30,30,30),3)
        cv2.line(frame,(x,y+40),(x+14,int(y+64-leg)),(30,30,30),3)

    elif activity == "Sitting":

        cv2.line(frame,(x,y+10),(x,y+32),(30,30,30),3)

        cv2.line(frame,(x,y+18),(x-10,y+25),(30,30,30),3)
        cv2.line(frame,(x,y+18),(x+10,y+25),(30,30,30),3)

        cv2.line(frame,(x,y+32),(x-12,y+42),(30,30,30),3)
        cv2.line(frame,(x-12,y+42),(x+5,y+42),(30,30,30),3)

        # Chair
        cv2.rectangle(frame,(x-15,y+32),(x+15,y+38),(80,80,80),-1)

    elif activity == "Fighting":

        punch = 18 * math.sin(frame_number*0.8)

        cv2.line(frame,(x,y+20),(x-18,int(y+18+punch)),(30,30,30),3)
        cv2.line(frame,(x,y+20),(x+18,int(y+18-punch)),(30,30,30),3)

        cv2.line(frame,(x,y+40),(x-8,y+60),(30,30,30),3)
        cv2.line(frame,(x,y+40),(x+8,y+60),(30,30,30),3)

# Main animation Loop:

def main():

    for frame_number in range(TOTAL_FRAMES):

        frame = np.full((HEIGHT, WIDTH, 3), 220, dtype=np.uint8)

        draw_background(frame)
        draw_cctv_overlay(frame, frame_number)

        # Select activity based on frame number
        if frame_number < 38:
            activity = "Walking"
            x = 50 + frame_number * 6
            y = 250

        elif frame_number < 75:
            activity = "Sitting"
            x = 300
            y = 250

        elif frame_number < 113:
            activity = "Fighting"
            x = 300
            y = 250

        else:
            activity = "Running"
            x = 50 + (frame_number - 113) * 12
            y = 250

        draw_person(frame, x, y, activity, frame_number)

        """
        # Display current activity
        cv2.putText(
            frame,
            f"Activity: {activity}",
            (15, 455),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )
        """

        # Add CCTV noise
        noise = np.random.normal(0, 5, frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Slight blur
        frame = cv2.GaussianBlur(frame, (3, 3), 0)

        writer.write(frame)

    writer.release()

    print("✅ CCTV Video Generated Successfully!")

if __name__ == "__main__":
    main()
