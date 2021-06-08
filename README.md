# Detectron2 Object Detection with TTA
## 1. Build image and run docker container
`docker-compose up --build -d && docker exec -it detectron2_header_1.0v bash`.

## 2. Set TTA parameters in `run.py`
In `run.py` file, you can change hyperparameter of TTA using argparse.
### 2.1. no TTA
`args.aug` must be `False`
### 2.2. flip parametermulti scale parameter
**Flip the image horizontally.**<br>
`args.aug` must be `True`. <br>
`args.flip` must be `True`. <br>

### 2.3. multi scale parameter
**Resize the given size.<br>
Number of input image for TTA are 10 => (9 images are resized, 1 image is original size)**<br>
`args.aug` must be `True`. <br>
`args.multi_scale` is `[size1, size2 ... size 9]`

### 2.4. contrast parameter
**Contrast intensity is uniformly sampled in (intensity_min, intensity_max).<br>
    - intensity < 1 will reduce contrast <br>
    - intensity = 1 will preserve the input image <br>
    - intensity > 1 will increase contrast**<br>

`args.aug` must be `True`. <br>
`args.contrast` is `[intensity_min, intensity_max]` Randomly transforms image contrast.

## 3. Just run code
After setting TTA parameters. <br>
Run `python3 run.py`.

## 4. check the test AP in your terminal.
Or check a log which located in `./src/model/log.txt`.