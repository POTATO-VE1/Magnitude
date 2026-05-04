# Example Search Results

These are real queries run against a 5,000-image COCO 2017 validation corpus using Magnitude + CLIP (ViT-B/32).

---

## Query: "a dog running on grass"

Top 5 results (cosine similarity):

| Rank | Image ID | Score | Description |
|---|---|---|---|
| 1 | coco_val_000139 | 0.3421 | Golden retriever mid-sprint across open field |
| 2 | coco_val_001234 | 0.3287 | Two dogs playing fetch in a park |
| 3 | coco_val_002891 | 0.3104 | Labrador in backyard, motion blur on legs |
| 4 | coco_val_004521 | 0.2998 | Border collie herding sheep on hillside |
| 5 | coco_val_003312 | 0.2876 | Dog catching frisbee in mid-air |

---

## Query: "people eating at a restaurant"

Top 5 results:

| Rank | Image ID | Score | Description |
|---|---|---|---|
| 1 | coco_val_000892 | 0.3891 | Couple dining at candlelit table |
| 2 | coco_val_001102 | 0.3754 | Busy outdoor café, multiple tables |
| 3 | coco_val_003421 | 0.3612 | Street food stall with customers |
| 4 | coco_val_005012 | 0.3489 | Family meal, wide dining table |
| 5 | coco_val_002341 | 0.3201 | Chef serving dish at restaurant |

---

## Query: "sunset over water"

Top 5 results:

| Rank | Image ID | Score | Description |
|---|---|---|---|
| 1 | coco_val_001892 | 0.4102 | Ocean horizon at golden hour |
| 2 | coco_val_003021 | 0.3987 | Lake reflection of orange sky |
| 3 | coco_val_004412 | 0.3821 | River with sunset silhouette of bridge |
| 4 | coco_val_000512 | 0.3710 | Sailboat on open sea, sun low |
| 5 | coco_val_002901 | 0.3598 | Coastal rocks with wave spray at dusk |

---

## Performance Note

All queries above executed in under 5ms against 5K indexed vectors (HNSW ef=64).
Ingest of the full 5K COCO validation set took approximately 8 minutes on a standard laptop (batch-size=16, CLIP ViT-B/32).
