import os
import sys
import argparse

import tensorflow as tf
from converter.model import build_dlib_model, ScaleLayer, ReshapeLayer
from converter.weights import load_weights
from converter.tensorflow import convert_to_tf_saved_model
from converter.tensorflow import convert_to_tf_frozen_model

def main(args):
    """ Main entry point """

    # Build the model (just the graph)
    keras_model = build_dlib_model(use_bn=False)
    keras_model.summary()   

    # parse xml and load weights
    load_weights(keras_model, args.xml_weights)

    # save it as h5
    keras_model.save("dlib_fr.h5", custom_objects={
        "ScaleLayer": ScaleLayer,
        "ReshapeLayer": ReshapeLayer
    })

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    tflite_model = converter.convert()

    with open("dlib_fr.tflite", "wb") as file_out:
        file_out.write(tflite_model)

    """ # save it as saved_model
    convert_to_tf_saved_model(keras_model, os.curdir)

    # save it as a frozen graph
    convert_to_tf_frozen_model(keras_model, os.curdir) """


def parse_arg(argv):
    """ Parse the arguments """
    arg_paser = argparse.ArgumentParser()

    arg_paser.add_argument(
        '--xml-weights',
        type=str,
        required=True,
        help='Path to the dlib recognition xml file')    

    return arg_paser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arg(sys.argv[1:]))