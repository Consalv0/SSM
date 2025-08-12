# Startopia SSM Importer for Blender

This is a Blender importer for the 3D model format used in the game **Startopia** by Mucky Foot Productions.  
It can read `.SSM` (Space Station Model) files and bring them into Blender with animations, materials, textures, and skeletons.  
The format was reverse engineered from the original game files. Maybe I'll do a exporter.

## Features

- Imports most, if not all, SSM models from Startopia  
- Supports animations, skeletons, materials, and textures  
- Option to display bone shapes  
- Adjustable import scale  
- Includes an ImHex pattern file describing the `.SSM` format

## Requirements

- Blender (tested with recent versions)  
- `.tga.ddt` textures need to be renamed or converted to `.tga` before use (I did a copy of the assets folder and renamed all .ddt textures)

## Usage

1. Clone or download this repository.  
2. In Blender, go to **Edit → Preferences → Add-ons → Install...**  
3. Select the importer `.py` file from this repository.  
4. Enable the add-on.

5. Choose **File → Import → Space Station Model (.ssm)**.  
6. Pick a `.SSM` file.
7. Import the model.

## ImHex pattern file

The `SSM.hexpattern` file in the repository is an [ImHex](https://imhex.werwolv.net/) pattern language description of the `.SSM` format.  
Copy the text in ImHex with an `.SSM` file loaded to see the parsed structure.

## License

MIT License — do what you want with it, but credit is appreciated.