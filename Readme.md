## padock\_photo\_classifier

### What's this?

Predict horse's distance aptitude (sprint / mile / middle / long) from their padock photo images.

###### distance type score

- 0.0 - 0.2: sprint
- 0.2 - 0.4: mile
- 0.4 - 0.6: middle
- 0.6 - 1.0: long

#### Requirements

- Python3
- chainer
- numpy
- PIL
- tqdm

#### Training

1. Prepare padock photo images and store them into `data/original/` directory.

	or you can download my sample dataset.
	
	```
	$ cd data
	$ ./download_sample.sh
	$ unzip keibado_padock_images.zip
	$ find keibado_padock_images -type f | xargs -J% cp % original/
	```
1. Preprocess images

	```
	$ ./preprocess_all.sh
	```
	
1. Generate a mean image

	```
	$ cd ../
	$ ./compute_mean.sh 
	```
1. Train dataset

	```
	$ train.sh
	```
		 	
#### Evaluation

Output loss for test images.

```
$ eval.sh result/your_model_output
```

