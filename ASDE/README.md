# Alpine Snow Depth Estimation

**Status:** research proposal only. Nothing has been rendered or trained yet, so there are no results here.

**Paper:** [snow_depth_research_paper.pdf](snow_depth_research_paper.pdf)

Depth models have very little to work with in snow and flat light, and real depth labels are hard to get because LiDAR picks up snowflakes. My plan is to render alpine snow scenes in Unreal Engine 5, where the depth is exact, and train a small model on them that corrects the depth from a frozen Depth Anything 3.

## The plan

| Part | What it is |
|---|---|
| UE5 generator | Alpine terrain, snow, whiteout lighting and a moving camera. Exports RGB, exact depth and pose for every frame. |
| Sensor simulators | Python scripts that turn exact depth into LiDAR with snow noise, and exact poses into noisy GPS/IMU. |
| Refiner | CNN encoder, transformer bottleneck and CNN decoder on 8 input channels. Takes DA3Metric-Large depth, LiDAR, a warped previous depth and RGB, then outputs corrected depth and a confidence map. |
| Evaluation | Five hypotheses written down before any training, tested on held-out renders and on real snow driving scenes from DENSE and Boreas. |

LiDAR and motion are randomly switched off during training, so one model can run with whatever sensors are available.

## Things I decided

- **Depth Anything 3 stays frozen.** The repo only documents inference, and at 0.35 billion parameters it probably won't fit fine-tuning on my RTX 3080. I use DA3Metric-Large because that checkpoint is Apache 2.0.
- **No real alpine test set.** Going to mountain sites with the right equipment costs more than I can cover as one person without funding, so I use public real data instead. This means the results won't prove anything about real alpine photos, and the paper says that.
