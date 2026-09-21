from typing import Dict, Any
from . import KnownPoseLoss, GradientPoseLoss, ESLPoseLoss, MLPPoseLoss

def build_pose_loss(
    cfg: Dict[str, Any], 
    load_rots: bool, 
    load_shifts: bool,
    num_mrcs: int, 
    renderer
):
    """Construct the pose loss named by pose_loss.method."""
    if load_shifts:
        raise NotImplementedError("Known shifts not implemented yet.")

    method = cfg['pose_loss']['method']
    if method == "known":
        if not load_rots:
            raise ValueError("pose_loss 'known' requires load_rots=True.")
        return KnownPoseLoss(renderer=renderer)

    if method == "gradient":
        return GradientPoseLoss(
            num_mrcs=num_mrcs,
            rot_grid_path=cfg['pose_loss']['rot_grid_path'],
            renderer=renderer,
        )

    if method == "esl":
        esl = cfg['pose_loss']['esl']
        return ESLPoseLoss(
            eta=esl['eta'],
            num_pts_base=esl['num_pts_base'],
            num_iters=esl['num_iters'],
            rot_grid_path=cfg['pose_loss']['rot_grid_path'],
            renderer=renderer,
        )

    if method == "mlp_est":
        mlp = cfg['pose_loss']['mlp_est']
        return MLPPoseLoss(
            num_layers_mlppose=mlp['num_layers_mlppose'],
            num_hidden_mlppose=mlp['num_hidden_mlppose'],
            latent_dim=cfg['network']['latent_dim'],
            act_name=mlp['act_name'],
            renderer=renderer,
        )

    raise ValueError(f"Unknown pose loss: {method!r}")
