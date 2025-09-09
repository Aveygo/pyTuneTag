
<p align="center">
    <img src="media/logo.svg" width=120>
</p>

<h3 align="center">pyTagTune</h3>
<p align="center" style="font-weight:300;">
    Automatic music genre detection
</p>

## TLDR
Uses embedding data from [mtrpp](https://github.com/seungheondoh/music-text-representation-pp) to automatically detect which genre an mp3 file is in, eg:

> Stairway to Heaven -> "Progressive Rock"

then, uses [mutagen](https://github.com/quodlibet/mutagen) to save the detected genre to the mp3's metadata.

## Running

1. Install via ```pip install git+https://github.com/Aveygo/pyTuneTag.git```
2. Then run with ```pytunetag <path to mp3 file>```

See ```pytunetag -h``` for more details

## Caveats

This program requires several *gigabytes* of storage for dependencies / models, and ideally a GPU. 

From what I could find, this project is actually the first real useable automatic genre-tagging script so the trade-off is unfortunately necessary.