SDXL + ControlNet (Canny) Baseline for Community Urban Layout

Overview
- Input (control): community-level road network rendered from OSM shapefiles (1024×1024, white background; road colors major=red, minor=blue, other=gray)
- Target: roads + filled building polygons (blue) (1024×1024)
- Model: stabilityai/stable-diffusion-xl-base-1.0 + ControlNet (canny). Full finetune via Diffusers+Accelerate on Ubuntu.

Dataset preparation
1) Ensure community JSONs and OSM shapefiles exist:
   - Communities: evaluation_communities_v3/*.json
   - Roads: Dataset/osm/<city>_roads.shp (CRS auto-set per city)

2) Generate pairs:
   ```bash
   conda create -n sdxl_cn python=3.10 -y
   conda activate sdxl_cn
   pip install -r requirements.txt

   python baseline/sdxl_controlnet_canny/prepare_dataset.py \
     --data_dir evaluation_communities_v3 \
     --osm_dir Dataset/osm \
     --output_dir baseline/sdxl_controlnet_canny/data \
     --cities Atlanta Chicago Dallas "Los Angeles" "New York" Philadelphia Phoenix "San Antonio" "San Diego" \
     --radius 500 --save_canny
   ```

This writes per-sample folders:
```
baseline/sdxl_controlnet_canny/data/<City>/<Community_ID>/
  control.png        # roads only (RGB)
  control_canny.png  # optional canny edge (if --save_canny)
  target.png         # roads + buildings (RGB)
  meta.json
```

Training (Ubuntu, Diffusers)
- Preferred approach: full finetune on 1× H100. Use HF Diffusers ControlNet training examples adapted to SDXL. Recommended flags: bf16 if supported, gradient checkpointing, gradient accumulation to fit batch size.

Sketch:
```bash
accelerate launch train_sdxl_controlnet.py \
  --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
  --controlnet_conditioning canny \
  --train_data_dir baseline/sdxl_controlnet_canny/data \
  --resolution 1024 --random_flip --mixed_precision bf16 \
  --train_batch_size 2 --gradient_accumulation_steps 8 \
  --learning_rate 1e-5 --max_train_steps 100000 \
  --output_dir baseline/sdxl_controlnet_canny/ckpts
```

Notes
- Aspect is strictly equal; rendering avoids any matplotlib scaling distortions.
- Use only the cities available in Dataset/osm.
- No Stage 4 integration; this baseline is for paper comparison only.

