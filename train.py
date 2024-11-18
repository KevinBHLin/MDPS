
import os
import argparse
from omegaconf import OmegaConf
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import transforms
from src.dataset import MVTec, BTAD
from src.diffusion import diffusion_loss
from src.models.unet import UNetModel

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def trainer(args):
    config = OmegaConf.load(args.config)
    print(config.data.category)

    model = UNetModel(config.data.image_size, 64, dropout=0.0, n_heads=4 ,in_channels=config.data.imput_channel)
    print("Num params: ", sum(p.numel() for p in model.parameters()))
    model = model.to(config.model.device)
    model.train()
    model = torch.nn.DataParallel(model)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.model.learning_rate, weight_decay=config.model.weight_decay
    )
    if config.data.name == 'MVTec':
        train_dataset = MVTec(
            root= config.data.data_dir,
            category=config.data.category,
            config = config,
            is_train=True,
        )
    if config.data.name == 'BTAD':
        train_dataset = BTAD(
            root= config.data.data_dir,
            category=config.data.category,
            config = config,
            is_train=True,
        )
    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.model.num_workers,
        drop_last=True,
    )
    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')
    if not os.path.exists(config.model.checkpoint_dir):
        os.mkdir(config.model.checkpoint_dir)

    for epoch in range(config.model.epochs):
        for step, batch in enumerate(trainloader):
            t = torch.randint(1, config.model.diffusion_steps, (batch[0].shape[0],), device=config.model.device).long()
            optimizer.zero_grad()
        
            loss = diffusion_loss(model, batch[0], t, config) 
            loss.backward()
            optimizer.step()
            if epoch % 1 == 0 and step == 0:
                print(f"Epoch {epoch} | Loss: {loss.item()}")
            if epoch % config.model.epochs_checkpoint == 0 and step ==0:
                model_save_dir = os.path.join(os.getcwd(), config.model.checkpoint_dir, config.data.category)
                if not os.path.exists(model_save_dir):
                    os.mkdir(model_save_dir)
                print('saving model')
                torch.save(model.state_dict(), os.path.join(model_save_dir, str(epoch)))
                    
def parse_args():
    parser = argparse.ArgumentParser('MDPS')    
    parser.add_argument('-cfg', '--config', help='config file')
    args, unknowns = parser.parse_known_args()
    return args


if __name__ == "__main__":
    seed = 42
    torch.cuda.empty_cache()
    args = parse_args()
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    trainer(args)