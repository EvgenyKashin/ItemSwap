# Item swap
This is source code for picsart.ai hackaton. It consist of two part: swapping face and swapping 
background.

## Examples
Source image:

<img src="images/a.jpg" width="256">

Face swap:

<img src="images/b.jpg" width="256">

Background swap:

<img src="images/c.jpg" width="256">

All together:

<img src="images/d.jpg" width="256">

## Using
### For initialization:
```
- git clone https://github.com/EvgenyKashin/ItemSwap.git
- make docker-build
- make load-weights
```

### For training:
```
- make train-face
- TODO: write instruction
```

### For face and background swap:
```
- make convert-face-video
- make convert-face-image
- make convert-background-image
```
For parameters see Makefile

***Important:*** there is ```loadSize``` parameter in ```make convert-background-image``` for changing image size.
