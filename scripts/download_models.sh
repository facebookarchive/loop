# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

mkdir -p models
pushd models

wget https://dl.fbaipublicfiles.com/loop/vctk_model.zip
unzip vctk_model.zip
rm vctk_model.zip

wget https://dl.fbaipublicfiles.com/loop/vctk_alt_model.zip
unzip vctk_alt_model.zip
rm vctk_alt_model.zip

wget https://dl.fbaipublicfiles.com/loop/blizzard_model.zip
unzip blizzard_model.zip
rm blizzard_model.zip

popd
