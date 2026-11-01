# Alpine snow depth estimation

**Status:** the dataset is rendered and Depth Anything 3 is scored on it, but the refiner isn't trained yet.

**Paper:** [snow_depth_research_paper.pdf](snow_depth_research_paper.pdf)

**Dataset:** [Ab1231/alpine-snow-depth](https://huggingface.co/datasets/Ab1231/alpine-snow-depth) on Hugging Face

Depth models have very little to work with in snow and flat light, and real depth labels are hard to get because LiDAR picks up snowflakes. I rendered alpine snow scenes in Unreal Engine 5, where the depth is exact, and my plan is to train a small model on them that corrects the depth from a frozen Depth Anything 3.

## The plan

| Part | What it is |
|---|---|
| Dataset (done) | 1,556 images in 778 pairs on real terrain around the Eiger, in sun, flat light and snowfall. Each has exact depth and pose. |
| Sensor simulators (done) | Python scripts that turn exact depth into LiDAR and exact poses into noisy GPS/IMU. The snow noise is measured from 30 real Boreas scans in heavy snow, so it's only a first calibration. |
| Refiner | CNN encoder, transformer bottleneck and CNN decoder on 8 input channels. Takes DA3Metric-Large depth, LiDAR, a warped previous depth and RGB, then outputs corrected depth and a confidence map. |
| Evaluation | Five hypotheses written down before any training, tested on held-out renders and on real snow driving scenes from DENSE and Boreas. |

LiDAR and motion are randomly switched off during training, so one model can run with whatever sensors are available.

## Depth Anything 3 on the renders

I ran frozen DA3Metric-Large on the 1,426 train and validation frames and left the test split unscored. These are plain averages over frames, with sky left out.

| Group | AbsRel | Order | Spread | Overlap |
|---|---:|---:|---:|---:|
| Overall | 0.962 | 0.951 | 0.402 | 0.518 |
| Sun | 0.961 | 0.979 | 0.487 | 0.587 |
| Flat light | 0.962 | 0.946 | 0.392 | 0.512 |
| Snowfall | 0.963 | 0.918 | 0.291 | 0.425 |
| High views | 0.999 | 0.933 | 0.401 | 0.550 |
| Ground views | 0.906 | 0.979 | 0.404 | 0.471 |
| Perfect score | 0 | 1 | 1 | 1 |

AbsRel scores real distance. Order is the Spearman correlation with the exact depth, so it shows whether DA3 knows what's near and far. Spread is the ratio of the two standard deviations of log depth, and overlap compares their histograms after removing the overall scale.

DA3 gets the order right but the metres wrong: the median pixel reads about 0.001 of its true distance, and even within 100 m it reads 0.177 of it. Spread also drops from 0.487 in sun to 0.291 in snowfall, so snow flattens its depth. A single scale correction won't fix that, which leaves the refiner real work to do.

## Things I decided

- **Depth Anything 3 stays frozen.** The repo only documents inference, and at 0.35 billion parameters it probably won't fit fine-tuning on my RTX 3080. I use DA3Metric-Large because that checkpoint is Apache 2.0.
- **No real alpine test set.** Going to mountain sites with the right equipment costs more than I can cover as one person without funding, so I use public real data instead. This means the results won't prove anything about real alpine photos, and the paper says that.
- **The generator stays private.** The dataset is public under MIT, with the swisstopo and Copernicus credits kept, but the generator code isn't in this repo.
