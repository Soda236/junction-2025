import argparse
import time
import math
import cv2
import numpy as np

from evio.core.pacer import Pacer
from evio.source.dat_file import BatchRange, DatFileSource
from ultralytics import YOLO

object_history = {}  # {label: [(timestamp, (cx, cy))]}

def get_window(event_words: np.ndarray, time_order: np.ndarray, win_start: int, win_stop: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    event_indexes = time_order[win_start:win_stop]
    words = event_words[event_indexes].astype(np.uint32, copy=False)
    x_coords = (words & 0x3FFF).astype(np.int32, copy=False)
    y_coords = ((words >> 14) & 0x3FFF).astype(np.int32, copy=False)
    pixel_polarity = ((words >> 28) & 0xF) > 0
    return x_coords, y_coords, pixel_polarity

def get_frame(window: tuple[np.ndarray, np.ndarray, np.ndarray], width: int = 1280, height: int = 720,
              *, base_color: tuple[int, int, int] = (127, 127, 127), on_color: tuple[int, int, int] = (255, 255, 255),
              off_color: tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    x_coords, y_coords, polarities_on = window
    frame = np.full((height, width, 3), base_color, np.uint8)
    frame[y_coords[polarities_on], x_coords[polarities_on]] = on_color
    frame[y_coords[~polarities_on], x_coords[~polarities_on]] = off_color
    return frame

def draw_hud(frame: np.ndarray, pacer: Pacer, batch_range: BatchRange, *, color: tuple[int, int, int] = (0, 0, 0)) -> None:
    if pacer._t_start is None or pacer._e_start is None:
        return
    wall_time_s = time.perf_counter() - pacer._t_start
    rec_time_s = max(0.0, (batch_range.end_ts_us - pacer._e_start) / 1e6)
    first_row_str = f"speed={pacer.speed:.2f}x" if pacer.force_speed else f"(target) speed={pacer.speed:.2f}x  force_speed=False"
    second_row_str = f"wall={wall_time_s:7.3f}s  rec={rec_time_s:7.3f}s"
    cv2.putText(frame, first_row_str, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    cv2.putText(frame, second_row_str, (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dat", help="Path to .dat file")
    parser.add_argument("--window", type=float, default=10, help="Window duration in ms")
    parser.add_argument("--speed", type=float, default=1, help="Playback speed (1 is real time)")
    parser.add_argument("--force-speed", action="store_true", help="Force playback speed by dropping windows")
    args = parser.parse_args()

    src = DatFileSource(args.dat, width=1280, height=720, window_length_us=args.window * 1000)
    pacer = Pacer(speed=args.speed, force_speed=args.force_speed)

    cv2.namedWindow("Evio Player", cv2.WINDOW_NORMAL)
    model = YOLO("models/best.pt")

    for batch_range in pacer.pace(src.ranges()):
        window = get_window(src.event_words, src.order, batch_range.start, batch_range.stop)
        frame = get_frame(window)
        color_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = model.predict(color_frame, imgsz=620)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = r.names[int(box.cls[0])]
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                timestamp = time.perf_counter()

                if label not in object_history:
                    object_history[label] = []
                object_history[label].append((timestamp, (cx, cy)))

                speed_text = ""
                if len(object_history[label]) >= 2:
                    t1, (x_prev, y_prev) = object_history[label][-2]
                    t2, (x_curr, y_curr) = object_history[label][-1]
                    dt = t2 - t1
                    dx = x_curr - x_prev
                    dy = y_curr - y_prev
                    speed = math.sqrt(dx**2 + dy**2) / dt
                    speed_text = f"{speed:.1f}px/s"

                color = (0, 0, 255) if label.lower() == "drone" else (0, 255, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                if speed_text:
                    cv2.putText(frame, speed_text, (x1, y2 + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        draw_hud(frame, pacer, batch_range)
        cv2.imshow("Evio Player", frame)

        if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
            break

if __name__ == "__main__":
    main()
