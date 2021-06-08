# Detectron2 Object Detection with TTA
## 1. Build image and run docker container
`docker-compose up --build -d`.

## 2. Go into docker container using bash shell
`docker exec -it detectron2_header_1.0v bash`.

## 3. Set TTA parameters in `run.py`
In `run.py` file, you can change hyperparameter of TTA using argparse.
### 3.1. no TTA
`args.aug` must be `False`
### 3.2. flip parametermulti scale parameter
`args.aug` must be `True`
`args.flip` must be `True`
### 3.3. multi scale parameter
`args.aug` must be `True`
`args.multi_scale` is `[400, 500, 600, 700, 800, 900, 1000, 1100, 1200]`
### 3.4. color parameter
`args.aug` must be `True`
`args.color_trans` is `[0.9, 1.2]`

## 4. Just run code
After setting TTA parameters. <br>
Run `python3 run.py`.

## 5. check the test AP in your terminal.
Or check a log which located in `./src/model/log.txt`.