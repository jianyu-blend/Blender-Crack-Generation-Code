"""Save a probability map from the new checkpoint and a binary brick layout."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from models import UNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--brick-mask', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if ckpt['mask_encoding'] != 'adaptive_brick_and_complement':
        raise ValueError('Unexpected checkpoint mask encoding.')
    model = UNet(in_ch=2, base=ckpt['base'], p_drop=ckpt['p_drop']).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    image = Image.open(args.brick_mask).convert('L')
    if image.size != (512, 512):
        raise ValueError('Expected a 512x512 binary brick mask.')
    a = np.array(image)
    if not np.all((a == 0) | (a == 255)):
        raise ValueError('Input must be binary brick occupancy, not RGB or a crack mask.')
    brick = torch.from_numpy(a.copy()).float()/255
    x = torch.stack([brick, 1-brick])[None].to(device)
    with torch.no_grad():
        prob = model(x).sigmoid()[0, 0].cpu().numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output.with_suffix('.npy'), prob)
    Image.fromarray(np.rint(prob*255).astype(np.uint8)).save(args.output.with_suffix('.png'))
    print(f'Saved float probability array and display PNG: {args.output.with_suffix("")}')


if __name__ == '__main__':
    main()
