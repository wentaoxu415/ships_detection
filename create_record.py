import os 
import io 
import PIL
import tensorflow as tf 
from object_detection.utils import dataset_util


def dict_to_tf_example(file_name, 
	                   masks,
	                   class_names, 
					   label_map_dict, 
					   image_directory,
					   image_size=(768, 768)):

	height = image_size[0]
	width = image_size[1]
	xmins = []
	ymins = []
	xmaxs = []
	ymaxs = []
	encoded_masks = []
 
 	# Read image
 	img_path = os.path.join(image_directory, file_name)
	
	with tf.gfile.GFile(img_path, 'rb') as fid:
		encoded_jpg = fid.read()
	
	encoded_jpg_io = io.BytesIO(encoded_jpg)
	image = PIL.Image.open(encoded_jpg_io)
	
	if image.format != 'JPEG':
	    raise ValueError('Image format not JPEG')
	
	key = hashlib.sha256(encoded_jpg).hexdigest()

	# Look up class id 
	classes = [label_map_dict[class_text] for class_text in classes_text]

	# Encode mask into png 
	for mask in masks:
		encoded_mask = convert_img_array_to_png_str(mask)
		encoded_masks.append(encoded_mask)

	feature_dict = {
		'image/height': dataset_util.int64_feature(height),
		'image/width': dataset_util.int64_feature(width),
		'image/filename': dataset_util.bytes_feature(file_name.encode('utf8')),
		'image/source_id': dataset_util.bytes_feature(file_name.encode('utf8')),
		'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
		'image/encoded': dataset_util.bytes_feature(encoded_jpg),
		'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
		'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      	'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      	'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      	'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      	'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      	'image/object/class/label': dataset_util.int64_list_feature(classes),
      	'image/object/mask': dataset_util.bytes_list_feature(encoded_masks),

	}
	example = tf.train.Example(features=tf.train.Features(feature=feature_dict))	
	
	return example 


def convert_rle_to_img_array(rle, shape=(768, 768)):
    '''
    rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    
    return img.reshape(shape).T  # Needed to align to RLE direction


def convert_img_array_to_png_str(img_array):
	img = PIL.Image.fromarray(img_array)
	output = io.BytesIO()
	img.save(output, format='PNG')
	
	return output.getvalue()