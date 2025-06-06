#!/usr/bin/env python
#
# Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file
# except in compliance with the License. A copy of the License is located at
#
# http://aws.amazon.com/apache2.0/
#
# or in the "LICENSE.txt" file accompanying this file. This file is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or implied. See the License for
# the specific language governing permissions and limitations under the License.
import sys

from optimum.commands import optimum_cli


def main():
    model_id = "openai/whisper-tiny"
    sys.argv = ["model_zoo_importer.py", "export", "onnx", "-m", model_id, "--optimize", "O3", "--monolith", "onnx"]

    optimum_cli.main()


if __name__ == "__main__":
    main()
