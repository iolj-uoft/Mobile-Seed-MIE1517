from mmseg.datasets.builder import PIPELINES
import os
import mmcv
import numpy as np

@PIPELINES.register_module()
class LoadEdgeFromFile:
    def __init__(self, edge_dir, edge_suffix='_edge.png'):
        self.edge_dir = edge_dir
        self.edge_suffix = edge_suffix

    def __call__(self, results):
        img_filename = results['filename']
        img_basename = os.path.basename(img_filename)
        edge_filename = os.path.splitext(img_basename)[0] + self.edge_suffix
        full_edge_path = os.path.join(self.edge_dir, edge_filename)

        edge = mmcv.imread(full_edge_path, flag='grayscale')
        if edge is None:
            raise FileNotFoundError(f"Edge map not found: {full_edge_path}")

        # Handle edge shape issues
        if edge.ndim == 3:
            edge = edge.squeeze()  # e.g. shape (H, W, 1) → (H, W)
        elif edge.ndim == 1:
            raise ValueError(f"Invalid edge shape {edge.shape} for file: {full_edge_path}")
        elif edge.ndim != 2:
            raise ValueError(f"Unexpected edge dimensions: {edge.shape}")

        results['gt_semantic_sebound'] = edge.astype(np.uint8)
        results['seg_fields'].append('gt_semantic_sebound')

        return results
