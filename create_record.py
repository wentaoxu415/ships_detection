from __future__ import division
import os 
import io 
import logging 
import contextlib2
import PIL
import hashlib
import numpy as np 
import pandas as pd 
import tensorflow as tf 
from skimage.measure import label, regionprops
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
from object_detection.dataset_tools import tf_record_creation_util
from sklearn.model_selection import train_test_split


flags = tf.app.flags
flags.DEFINE_string('masks_csv', '', 'Path to the Mask CSV')
flags.DEFINE_float('train_proportion', 0.8, 'Proportion of dataset dedicated for training')
flags.DEFINE_string('image_directory', '', 'Input directory for raw images')
flags.DEFINE_string('output_dir', '', 'Output directory for TFRecords')
flags.DEFINE_string('label_map_path', '', 'Path to label map proto')
flags.DEFINE_integer('num_shards', 1, 'Number of shards to split the dataset into')
FLAGS = flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)


def main(_):
    masks_csv = FLAGS.masks_csv
    train_proportion = FLAGS.train_proportion
    image_directory = FLAGS.image_directory
    train_output_path = os.path.join(FLAGS.output_dir, 'train/train.record')
    val_output_path = os.path.join(FLAGS.output_dir, 'eval/eval.record')
    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)
    num_shards = FLAGS.num_shards

    masks_dict = create_masks_dict(masks_csv)
    image_ids = [key for key in masks_dict.keys()]
    train_ids, val_ids = train_test_split(image_ids, train_size=train_proportion, random_state=0)
    train_masks_dict = {key: masks_dict[key] for key in train_ids}
    val_masks_dict = {key: masks_dict[key] for key in val_ids}

    create_tf_record(train_output_path, 
                     label_map_dict,
                     image_directory,
                     train_masks_dict,
                     num_shards) 

    create_tf_record(val_output_path, 
                     label_map_dict,
                     image_directory,
                     val_masks_dict,
                     num_shards)


def create_tf_record(output_path, 
                     label_map_dict,
                     image_directory, 
                     masks_dict,
                     num_shards):

    with contextlib2.ExitStack() as tf_record_close_stack:
        output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
            tf_record_close_stack, output_path, num_shards)

        for index, (image_id, masks) in enumerate(masks_dict.items()):
            if index % 1000 == 0:
                logging.info('On image {0} of {1}'.format(index, len(masks_dict)))

            try:
                class_names = ['ship'] * len(masks)
                tf_example = create_tf_example(image_id, masks, class_names, label_map_dict, 
                    image_directory)
                output_shard_index = index % num_shards
                output_tfrecords[output_shard_index].write(tf_example.SerializeToString())

            except ValueError:
                logging.error("Error while attempting to create a record for {}".format(image_id))


def create_tf_example(file_name, 
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
    class_ids = [label_map_dict[class_name] for class_name in class_names]

    # Encode class names into bytes
    class_names = [name.encode('utf8') for name in class_names]

    # Encode mask into png and get bounding box coordinates  
    for mask in masks:
        mask_array = convert_mask_rle_to_img_array(mask)
        encoded_mask = convert_img_array_to_png_str(mask_array)
        encoded_masks.append(encoded_mask)
        
        xmin, xmax, ymin, ymax = get_bbox_coordinates(mask_array)

        xmins.append(xmin / width)
        xmaxs.append(xmax / width)
        ymins.append(ymin / height)
        ymaxs.append(ymax / height)

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
        'image/object/class/text': dataset_util.bytes_list_feature(class_names),
        'image/object/class/label': dataset_util.int64_list_feature(class_ids),
        'image/object/mask': dataset_util.bytes_list_feature(encoded_masks),

    }
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))    
    
    return example 


def create_masks_dict(csv_file):
    df = pd.read_csv(csv_file)
    df = df[df.EncodedPixels.notnull()]
    masks_dict = {}

    for row in df.itertuples():
        if row.ImageId not in masks_dict:
            masks_dict[row.ImageId] = [row.EncodedPixels]
        else:
            masks_dict[row.ImageId].append(row.EncodedPixels)

    return masks_dict



def convert_mask_rle_to_img_array(mask_rle, shape=(768, 768)):
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


def get_bbox_coordinates(mask):
    lbl = label(mask)
    props = regionprops(lbl)

    # Only keep masks that have bounding box area of greater than 1
    props = [prop for prop in props if prop.bbox_area > 1]
    if len(props) != 1:
        raise ValueError("The mask had {} regions".format(len(props)))
    else:
        prop = props[0]
        xmin = prop.bbox[1]
        xmax = prop.bbox[3]
        ymin = prop.bbox[0]
        ymax = prop.bbox[2]

        return xmin, xmax, ymin, ymax 


if __name__ == '__main__':
    tf.app.run()
    