import argparse
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
import io
from PIL import Image

def extract_parquet(source_dir: str, out_dir: str, dataset_name: str, max_images: int = 1500):
    source = Path(source_dir)
    out_dir = Path(out_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    
    parquet_files = list(source.rglob("*.parquet"))
    if not parquet_files:
        print(f"No parquet files found in {source_dir}")
        return

    records = []
    count = 0
    for pf in parquet_files:
        if count >= max_images:
            break
        print(f"Reading {pf.name}...")
        try:
            table = pq.read_table(pf)
            df = table.to_pandas()
        except Exception as e:
            print(f"Failed to read {pf}: {e}")
            continue
            
        for idx, row in df.iterrows():
            if count >= max_images:
                break
            
            # Find the image column (usually named 'image' or 'img')
            img_col = None
            if 'image' in row:
                img_col = 'image'
            elif 'img' in row:
                img_col = 'img'
            elif 'main_image' in row:
                img_col = 'main_image'
            
            if not img_col:
                continue
                
            img_data = row[img_col]
            if isinstance(img_data, dict) and 'bytes' in img_data:
                img_bytes = img_data['bytes']
            elif isinstance(img_data, bytes):
                img_bytes = img_data
            else:
                continue
            img_id = f"{dataset_name}_{count:05d}.jpg"
            img_path = img_dir / img_id
            
            try:
                img = Image.open(io.BytesIO(img_bytes))
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img.save(img_path)
            except Exception as e:
                print(f"Error saving image {img_id}: {e}")
                continue
                
            record = {"image_id": img_id, "image_path": str(img_path.absolute())}
            
            # Preserve all other textual metadata (like reports, labels)
            for col in df.columns:
                if col != img_col:
                    record[col] = row[col]
                    
            records.append(record)
            count += 1
            if count % 100 == 0:
                print(f"  Extracted {count} images...")

    if records:
        out_csv = out_dir / f"{dataset_name}_manifest.csv"
        pd.DataFrame(records).to_csv(out_csv, index=False)
        print(f"Finished {dataset_name}. Wrote {len(records)} records to {out_csv}")
    else:
        print(f"No valid image records found for {dataset_name}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--max", type=int, default=1500)
    args = parser.parse_args()
    
    extract_parquet(args.source, args.out, args.name, args.max)
