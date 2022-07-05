# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import sys
from typing import List
import numpy as np
import tritonclient.grpc
import argparse


def load_image(img_path: str) -> np.ndarray:
    """
    Loads an encoded image as an array of bytes.
    """
    return np.fromfile(img_path, dtype='uint8')


def load_labels(labels_path: str) -> List[str]:
    """
    Loads the labels contained in the label file into a list of strings
    """
    with open(labels_path) as file:
        lines = file.readlines()
        labels = [line.rstrip() for line in lines]
    return labels


def generate_inputs(
    labels_array: np.ndarray, 
    image_array: np.ndarray,
) -> List[tritonclient.grpc.InferInput]:
    """
    Generates inputs from image and label arrays
    """
    
    inputs = []

    inputs.append(
        tritonclient.grpc.InferInput("INPUT_0", image_array.shape, "UINT8"))
    inputs.append(
        tritonclient.grpc.InferInput("INPUT_1", labels_array.shape, "BYTES"))
    
    inputs[0].set_data_from_numpy(image_array)
    inputs[1].set_data_from_numpy(labels_array)

    return inputs


def print_results(output_data: np.ndarray, labels: List[str]) -> None:
    """
    Pretty prints the class-wise probabilities and best match
    """
    print()
    for idx, label in enumerate(labels):
        print(f"{label}: {output_data[idx]:.2f}")

    best_label_index = np.argmax(output_data)
    print("Best match: {}".format(labels[best_label_index]) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",
                        type=str,
                        required=False,
                        default="ensemble",
                        help="Model name")
    parser.add_argument("--image",
                        type=str,
                        required=True,
                        help="Path to the image")
    parser.add_argument("--url",
                        type=str,
                        required=False,
                        default="localhost:8001",
                        help="Inference server URL. Default is localhost:8001.")
    parser.add_argument('-v',
                        "--verbose",
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
    parser.add_argument("--label_file",
                        type=str,
                        default="./clip_repository/preprocess/labels.txt",
                        help="Path to the file with text representation of available labels")
        		 
    args = parser.parse_args()

    try:
        triton_client = tritonclient.grpc.InferenceServerClient(
            url=args.url, verbose=args.verbose)
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit(1)

    # Load text labels
    labels = load_labels(args.label_file)
    labels_array = np.array(list(labels), dtype=np.object_)
    
    # Load image data
    image_data = load_image(args.image)
    image_data = np.expand_dims(image_data, axis=0)

    # Create input and output inference objects
    inputs = generate_inputs(labels_array, image_data)
    outputs = [tritonclient.grpc.InferRequestedOutput("OUTPUT")]
    
    results = triton_client.infer(model_name=args.model_name,
                                  inputs=inputs,
                                  outputs=outputs)

    output0_data = results.as_numpy("OUTPUT")
    print_results(output0_data, labels)
