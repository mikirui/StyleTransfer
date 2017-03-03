import os, json, argparse
from threading import Thread 
from Queue import Queue

import numpy as np 
from scipy.misc import imread, imresize
import h5py

parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', default='/dev_data/ylh/Dataset/MSCOCO/train2014')
parser.add_argument('--val_dir', default='/dev_data/ylh/Dataset/MSCOCO/val2014')
parser.add_argument('--output_file', default='/dev_data/wrz/mscoco_256.h5')
parser.add_argument('--height', type=int, default=256)
parser.add_argument('--width', type=int, default=256)
parser.add_argument('--max_images', type=int, default=-1)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--include_val', type=int, default=1)
parser.add_argument('--max_resize', default=16, type=int)
args = parser.parse_args()

def add_data(h5_file, image_dir, prefix, args):
	image_list = []
	image_exts = {'.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG'}
	for filename in os.listdir(image_dir):
		ext = os.path.splitext(filename)[1]
		if ext in image_exts:
			image_list.append(os.path.join(image_dir, filename))
	num_images = len(image_list)
	print("there are %d images"%(num_images))

	dset_name = os.path.join(prefix, 'images')
	dset_size = (num_images, 3, args.height, args.width)
	imgs_dset = h5_file.create_dataset(dset_name, dset_size, np.uint8)

	input_queue = Queue()
	output_queue = Queue()

	#Read images from disk and resize them
	def read():
		while True:
			idx, filename = input_queue.get()
			img = imread(filename)
			try:
				H, W = img.shape[0], img.shape[1]
				H_crop = H - H % args.max_resize
				W_crop = W - W % args.max_resize
				img = img[:H_crop, :W_crop]
				img = imresize(img, (args.height, args.width))
			except (ValueError, IndexError) as e:
				print filename
				print img.shape, img.dtype
				print e
			input_queue.task_done()
			output_queue.put((idx, img))

	def write():
        		num_written = 0
        		while True:
        			idx, img = output_queue.get()
        			if img.ndim == 3:
        				imgs_dset[idx] = img.transpose(2, 0, 1)   #[H,W,C] -> [C, H, W]
        			elif img.ndim == 2:
        				imgs_dset[idx] = img
        			output_queue.task_done()
        			num_written = num_written + 1
        			if num_written % 100 == 0:
        				print "Copied %d / %d images" % (num_written, num_images)

	for i in xrange(args.num_workers):
        		t = Thread(target = read)
        		t.daemon = True
        		t.start()

	t = Thread(target = write)
	t.daemon = True
	t.start()

	for idx, filename in enumerate(image_list):
        		if args.max_images > 0 and idx >= args.max_images: break
        		input_queue.put((idx, filename))

	input_queue.join()
	output_queue.join()

if __name__ == '__main__':
	with h5py.File(args.output_file, 'w') as f:
		add_data(f, args.train_dir, 'train2014', args)
		if args.include_val !=0:
			add_data(f, args.val_dir, 'val2014', args)
