from __future__ import annotations
import numpy as np

def densecrf_refine(image_rgb: np.ndarray, prob_fg: np.ndarray, iters: int = 5) -> np.ndarray:
    """
    image_rgb: HxWx3 uint8
    prob_fg:   HxW float in [0,1]
    returns:   HxW uint8 {0,1}
    """
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian

    H, W = prob_fg.shape
    prob_fg = np.clip(prob_fg, 1e-6, 1 - 1e-6)
    prob_bg = 1.0 - prob_fg
    probs = np.stack([prob_bg, prob_fg], axis=0)  # [2,H,W]

    d = dcrf.DenseCRF2D(W, H, 2)
    U = unary_from_softmax(probs)
    d.setUnaryEnergy(U)

    feats_gauss = create_pairwise_gaussian(sdims=(3, 3), shape=(H, W))
    d.addPairwiseEnergy(feats_gauss, compat=3)

    feats_bilat = create_pairwise_bilateral(
        sdims=(40, 40),
        schan=(13, 13, 13),
        img=image_rgb,
        chdim=2,
    )
    d.addPairwiseEnergy(feats_bilat, compat=10)

    Q = d.inference(iters)
    refined = np.array(Q).reshape((2, H, W))
    out = (refined[1] > refined[0]).astype(np.uint8)
    return out
