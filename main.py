import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torchvision.datasets as datasets
import tensorflow as tf
from utils import preprocess_for_eval
from pnasnet import build_pnasnet_large, pnasnet_large_arg_scope

slim = tf.contrib.slim

parser = argparse.ArgumentParser()
parser.add_argument('--valdir', type=str, default='data/val',
                    help='path to ImageNet val folder')
parser.add_argument('--image_size', type=int, default=331,
                    help='image size')


def main():
  args = parser.parse_args()

  image_ph = tf.placeholder(tf.uint8, (None, None, 3))
  image_proc = preprocess_for_eval(image_ph, args.image_size, args.image_size)
  images = tf.expand_dims(image_proc, 0)
  with slim.arg_scope(pnasnet_large_arg_scope()):
    logits, _ = build_pnasnet_large(images, num_classes=1001, is_training=False)

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)
  ckpt_restorer = tf.train.Saver()
  ckpt_restorer.restore(sess, 'data/model.ckpt')

  c1, c5 = 0, 0
  val_dataset = datasets.ImageFolder(args.valdir)
  for i, (image, label) in enumerate(val_dataset):
    logits_val = sess.run(logits, feed_dict={image_ph: image})
    top5 = logits_val.squeeze().argsort()[::-1][:5]
    top1 = top5[0]
    if label + 1 == top1:
      c1 += 1
    if label + 1 in top5:
      c5 += 1
    print('Test: [{0}/{1}]\t'
          'Prec@1 {2:.3f}\t'
          'Prec@5 {3:.3f}\t'.format(
          i + 1, len(val_dataset), c1 / (i + 1.), c5 / (i + 1.)))


if __name__ == '__main__':
  main()
