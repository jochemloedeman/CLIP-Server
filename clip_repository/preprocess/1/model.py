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

from typing import List, Dict
import numpy as np
import json
import io
import torch
import triton_python_backend_utils as pb_utils
import torchvision.transforms as transforms
import clip

from PIL import Image

class TritonPythonModel:
    """
    This model applies a sequence of torchvision transforms on incoming
    images, and tokenizes the incoming text.
    """

    def initialize(self, args: Dict) -> None:

        self.model_config = model_config = json.loads(args['model_config'])

        output0_config = pb_utils.get_output_config_by_name(
            model_config, "OUTPUT_0"
        )
        output1_config = pb_utils.get_output_config_by_name(
            model_config, "OUTPUT_1"
        )

        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type'])
        self.output1_dtype = pb_utils.triton_string_to_numpy(
            output1_config['data_type'])

        self.image_transforms = transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711]
                )
            ])

    def execute(
        self, 
        requests: List[pb_utils.InferenceRequest]
    ) -> List[pb_utils.InferenceResponse]:
        """
        This function is called when an inference is requested
        for this model
        """

        output0_dtype = self.output0_dtype
        output1_dtype = self.output1_dtype

        responses = []

        for request in requests:
            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT_0")
            in_1 = pb_utils.get_input_tensor_by_name(request, "INPUT_1")

            # decode and transform images
            image = in_0.as_numpy()
            image = Image.open(io.BytesIO(image.tobytes()))
            image_out = self._transform_image(image).numpy()

            # decode and tokenized the captions
            captions = in_1.as_numpy().astype(np.object_).tolist()
            captions = [string.decode('UTF-8') for string in captions]
            tokenized_captions = np.array(clip.tokenize(captions))

            # build outputs
            out_tensor_0 = pb_utils.Tensor(
                "OUTPUT_0",
                image_out.astype(output0_dtype)
            )
            out_tensor_1 = pb_utils.Tensor(
                "OUTPUT_1",
                tokenized_captions.astype(output1_dtype)
            )

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0, out_tensor_1])
            responses.append(inference_response)

        return responses

    def finalize(self) -> None:
        print('Cleaning up...')

    def _transform_image(self, image: Image) -> torch.Tensor:
        image = self.image_transforms(image)
        image = image.unsqueeze(0)
        return image
