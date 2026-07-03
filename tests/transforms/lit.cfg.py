# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Lit config for CoralNPUTransforms."""

# pylint: disable=undefined-variable

import os
import tempfile

import lit.formats

config.name = "CoralNPUTransforms"
config.suffixes = [".mlir", ".txt"]
config.test_format = lit.formats.ShTest(execute_external=True)

# Forward all IREE and CoralNPU environment variables, as well as some passthroughs.
passthrough_env_vars = [
    "VK_ICD_FILENAMES",
    "VCTOOLSINSTALLDIR",
    "UNIVERSALCRTSDKDIR",
    "UCRTVERSION",
]
config.environment.update({
    k: v for k, v in os.environ.items() if k.startswith("IREE_") or
    k.startswith("CORALNPU_") or k in passthrough_env_vars
})

# Use the most preferred temp directory.
config.test_exec_root = (os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR") or
                         os.environ.get("TEST_TMPDIR") or
                         os.path.join(tempfile.gettempdir(), "lit"))

config.substitutions.append(
    ('%coralnpu_compile', 'coralnpu-compile '
     '--iree-hal-target-device=local '
     '--iree-hal-local-target-device-backends=vmvx '
     '--iree-hal-target-device=coralnpu '
     '--coralnpu-target-abi=ilp32 '
     '--coralnpu-target-cpu-features=+m,+f,+zvl128b,+zve32f '))
