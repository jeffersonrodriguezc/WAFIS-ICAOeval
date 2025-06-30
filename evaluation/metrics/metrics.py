import torch
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

def get_message_accuracy(
    msg: torch.Tensor,
    deco_msg: torch.Tensor,
    msg_num: int
) -> float:
    """
    Calculate the bit accuracy of decoded messages.

    Args:
        msg (torch.Tensor): The original message tensor.
        deco_msg (torch.Tensor): The decoded message tensor.
        msg_num (int): The number of the message segments.

    Returns:
        float: The bit accuracy of the decoded messages.

    Raises:
        TypeError: If the inputs are not torch.Tensors or msg_num is not an int.
        ValueError: If the msg_num is less than 1.
    """
    if not isinstance(msg, torch.Tensor) or not isinstance(deco_msg, torch.Tensor):
        raise TypeError("Inputs msg and deco_msg must be torch.Tensors.")
    
    if not isinstance(msg_num, int):
        raise TypeError("Input msg_num must be an int.")
    
    if msg_num < 1:
        raise ValueError("msg_num must be at least 1.")
    
    if 'cuda' in str(deco_msg.device):
        deco_msg = deco_msg.cpu()
        msg = msg.cpu()

    if msg_num == 1:
        deco_msg = torch.round(deco_msg)
        correct_pred = torch.sum((deco_msg - msg) == 0, dim=1)
        bit_acc = torch.sum(correct_pred).numpy() / deco_msg.numel()
    else:
        bit_acc = 0.0
        for i in range(msg_num):
            cur_deco_msg = torch.round(deco_msg[:, i, :])
            correct_pred = torch.sum((cur_deco_msg - msg[:, i, :]) == 0, dim=1)
            bit_acc += torch.sum(correct_pred).numpy() / cur_deco_msg.numel()
        bit_acc /= msg_num

    return bit_acc

def ssim(img1, img2, cal_ssim):
    s = cal_ssim(img1, img2)
    return s.cpu().detach().numpy()

def psnr(cover, generated, cal_psnr):
    psnr = cal_psnr(generated, cover)
    return psnr.cpu().detach().numpy()

def compute_image_score(cover, generated):
    
    p = psnr(cover, generated)
    s = ssim(cover, generated)
    
    return p, s