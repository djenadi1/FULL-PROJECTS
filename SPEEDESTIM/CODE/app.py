"""Upload a traffic video, get it back with every car tracked and its speed.

Run it with:  streamlit run app.py
"""
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import streamlit as st

st.title("Car speed from a traffic video")
st.write("YOLO finds the cars, ByteTrack follows them and a small transformer guesses their speed. "
         "Everything was trained on synthetic Unreal Engine renders only.")
st.info("With the right camera setup the speed model is 2.75 km/h off on average (measured on synthetic cameras): "
        "a camera 15 to 17 m above the road, pitched 11 to 24 degrees down, with a 35 to 72 degree field of view, "
        "and cars in the closer half of the picture. Lower cameras and far cars are less accurate.")

video = st.file_uploader("Your video", type=["mp4", "mov", "avi", "webm"])
height = st.number_input("Camera height above the road (m)", value=15.0)
pitch = st.number_input("Camera pitch (degrees, minus is looking down)", value=-16.0)
fov = st.number_input("Camera field of view (degrees)", value=50.0)
seconds = st.slider("How many seconds of the video to use", 5, 60, 10)

if video and st.button("Start"):
    # save the upload so the script can read it
    path = Path(tempfile.mkdtemp()) / video.name
    path.write_bytes(video.getvalue())

    bar = st.progress(0.0, "Starting...")
    start = time.time()
    script = Path(__file__).parent / "track_speed.py"
    run = subprocess.Popen([sys.executable, str(script), str(path), str(height), str(pitch), str(fov), str(seconds)],
                           stdout=subprocess.PIPE, text=True)

    # the script prints "progress 0.42" lines, turn them into the loading bar
    for line in run.stdout:
        if line.startswith("progress"):
            done = float(line.split()[1])
            if done > 0:
                left = (time.time() - start) / done * (1 - done)
                bar.progress(done, f"{done:.0%} done, about {left:.0f} seconds left")
    run.wait()

    bar.progress(1.0, "Done")
    st.session_state["result"] = str(path.with_name(path.stem + "_speed.mp4"))

# keep showing the result, the page reruns on every click
if "result" in st.session_state:
    st.video(st.session_state["result"])
    st.download_button("Download the video", Path(st.session_state["result"]).read_bytes(), file_name="speed.mp4")
