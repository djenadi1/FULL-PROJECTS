# TikTok Video Data — September 2026 slice

## About this data

The files in this folder are a **small part of a much bigger dataset**. They cover **only one month (September 2026)** of video metadata, taken from a public dataset that spans **years of TikTok activity** — from 2014 (archived early videos) up to 2026 — and holds about **5.6 billion videos** in total.

- **Source dataset:** [datasocial/tiktok-5.6B-videos on Hugging Face](https://huggingface.co/datasets/datasocial/tiktok-5.6B-videos)
- **Full dataset size:** ~5.6 billion rows (one row per video/photo post), Parquet format
- **This folder:** one monthly file, plus a 100k-row sample for quick work

Anything learned from this folder describes one month only. It does not stand for TikTok as a whole or for trends over time. To study long-term trends, pull more months from the source dataset.

## Files

| File | Size | Description |
|---|---|---|
| `2026-09.parquet` | ~10.9 GB | The September 2026 slice of the full dataset |
| `sample_100k.parquet` | ~17 MB | 100,000-row sample for quick exploration and testing code |
| `README.md` | — | This file |

## Columns

| Column | Type | Notes |
|---|---|---|
| `video_id` | UInt64 | `tiktok.com/@_/video/{video_id}` |
| `posted_at` | DateTime | UTC |
| `country` | String | region TikTok gives the video |
| `language` | String | `un` = undetermined |
| `type` | String | `video` or `photo` |
| `duration_ms` | UInt32 | 0 on photos |
| `image_count` | UInt8 | photo posts |
| `width`, `height` | UInt16 | pixels |
| `caption` | String | |
| `hashtags`, `hashtag_ids` | Array | lower-case, parallel arrays |
| `mentions` | Array(UInt64) | creator_ids |
| `on_screen_text` | Array(String) | text stickers |
| `sound_id` | UInt64 | 0 = none |
| `is_ad` | UInt8 | Spark Ads |
| `branded_content` | String | `paid_partnership`, `shop_affiliate` or empty |
| `product_id`, `seller_id` | UInt64 | TikTok Shop; 0 = none |
| `is_shop_video` | UInt8 | |
| `is_ai_generated` | UInt8, nullable | null = TikTok did not say |
| `not_recommended_for_fyp` | UInt8, nullable | TikTok keeps it off the For You page |
| `sound_muted` | UInt8 | audio removed for copyright |
| `views`, `likes`, `comments`, `shares`, `saves`, `downloads` | UInt64 | as of `stats_updated_at` |
| `engagement_rate` | Float32, nullable | (likes + comments + shares + saves) / views |
| `stats_updated_at` | DateTime | UTC |

## Notes

- Engagement numbers (`views`, `likes`, …) are snapshots taken at `stats_updated_at`, not final totals.
- In the full dataset, older videos come from an archive: some fields are 0 or null there (size, downloads, AI flag), and duration is rounded to the second.

## License

The source dataset is released under **CC BY-NC 4.0**: free for research and personal use, **not for commercial use**. See the [Hugging Face dataset page](https://huggingface.co/datasets/datasocial/tiktok-5.6B-videos) for full terms and contact details of the dataset authors.
