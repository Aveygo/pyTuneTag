import logging

from pytunetag.pytunetag import pyTuneTag
from pathlib import Path
import argparse

logging.basicConfig(level=logging.INFO)
logging.info("START")


def main(mp3_paths: list[Path], model_files: Path, genres: list[str]):
    if not model_files.exists():
        model_files.mkdir(parents=True, exist_ok=True)
        
    tagger = pyTuneTag(list(set(genres)), model_files)
    
    for mp3_path in mp3_paths:
        scores = tagger(mp3_path)
        
        if scores is None:
            logging.warning(f"Failed to get scores for '{mp3_path}'.")
            continue
        
        genres_out = []
        for idx, r in enumerate(scores):
            name, score = r
            if score > 0.5 or idx == 0:
                if name.startswith("This song is in the music genre of "):
                    name = name.replace("This song is in the music genre of ", "")
                genres_out.append(name)
            
        name = tagger.set_meta(mp3_path, genres_out)
        logging.info(f"Set '{name}' to {genres_out}")

def cli():
    parser = argparse.ArgumentParser(description="Tag MP3 files with genres using pyTuneTag. Supports single files or entire directories.")
    
    parser.add_argument("src", type=str, help="Path to an MP3 file or a directory containing MP3 files.")
    parser.add_argument("--models", type=str, default="./models/", help="Path to store the model files (default: ./models/).")
    parser.add_argument("--genres", type=str, default="default", help="List of genres to detect from (will use default list otherwise)")
    parser.add_argument("--recursive", "-r", action="store_true", help="If src is a directory, recurse into subdirectories to find MP3 files.")
    
    args = parser.parse_args()
    
    src_path = Path(args.src)
    model_files = Path(args.models)
    
    if args.genres == "default":
        genres = ["Pop", "Rock", "Hip-Hop/Rap", "R&B/Soul", "Electronic/Dance/EDM", "Country", "Latin", "Jazz", "Classical", "Blues", "Folk", "Reggae", "Metal", "Punk", "Funk", "Alternative/Indie", "Gospel/Christian", "World/International"]
    
    mp3_paths = []
    
    if src_path.is_file():
        if src_path.suffix.lower() == '.mp3':
            mp3_paths.append(src_path)
        else:
            logging.error(f"'{src_path}' is not an MP3 file.")
            return
    elif src_path.is_dir():
        glob_method = src_path.rglob if args.recursive else src_path.glob
        for p in glob_method('*.mp3'):
            mp3_paths.append(p)
        if not mp3_paths:
            logging.info(f"No MP3 files found in '{src_path}'{' or subdirectories' if args.recursive else ''}.")
            return
    else:
        logging.error(f"'{src_path}' does not exist or is not a file/directory.")
        return
    
    main(mp3_paths, model_files, genres)

if __name__ == "__main__":
    cli()