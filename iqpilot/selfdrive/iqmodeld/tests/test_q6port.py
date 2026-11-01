# Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
import base64
import pickle

import numpy as np
import onnx
from onnx import TensorProto, helper

from iqpilot.selfdrive.iqmodeld.metadata import build_metadata_record
from iqpilot.selfdrive.iqmodeld.temporal_state import spec_from_meta
from iqpilot.selfdrive.iqmodeld.daemon import ReplayLedger


def test_metadata_preserves_onnx_types_and_style(tmp_path):
  inputs = [helper.make_tensor_value_info(name, dtype, shape) for name, dtype, shape in (
    ('map_tile', TensorProto.UINT8, [1, 3, 256, 256]), ('drive_profile', TensorProto.FLOAT, [1, 5]),
    ('features_buffer', TensorProto.FLOAT16, [1, 24, 512]))]
  graph = helper.make_model(helper.make_graph([], 'types', inputs, []))
  helper.set_model_props(graph, {'output_slices': base64.b64encode(pickle.dumps({})).decode(),
                                 'map_tile_style': 'iqmodel-map-raster-v3'})
  path = tmp_path / 'types.onnx'
  onnx.save(graph, path)
  meta = build_metadata_record(path)
  assert meta['input_dtypes'] == {'map_tile': 'uint8', 'drive_profile': 'float32', 'features_buffer': 'float16'}
  assert spec_from_meta(meta)['map_tile'] == ((1, 3, 256, 256), 'uint8')
  assert meta['map_tile_style'] == 'iqmodel-map-raster-v3'
  ledger = ReplayLedger(meta['input_shapes'], [], meta['input_dtypes'])
  assert ledger.inputs['map_tile'].dtype == np.uint8
  assert ledger.inputs['features_buffer'].dtype == np.float32

