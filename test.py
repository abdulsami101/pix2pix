# import os
# import config
# import torch
# from torch.utils.data import DataLoader
# from dataset import anime

# from generator_model import Generator
# import torchvision
# import matplotlib.pyplot as plt 
# import matplotlib.animation as animation
# from IPython.display import HTML

# # Set the device
# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# # Load the trained generator model
# model = Generator().to(device)

# # Load the model for inference
# loaded_model = Generator()  # Create an instance of the Generator class
# FILE = './gen.pth.tar'
# loaded_model.load_state_dict(torch.load(FILE))
# loaded_model.to(device)
# loaded_model.eval()  # Set the model for evaluation

# val_dataset = anime(root_dir=config.VAL_DIR)
# val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# # Set the output directory
# output_directory = './output/'
# os.makedirs(output_directory, exist_ok=True)

# # Initialize the animation
# fig = plt.figure(figsize=(8, 8))
# plt.axis("off")
# ims = []

# # Perform testing
# with torch.no_grad():
#     for i, (x, _) in enumerate(val_loader):
#         x = x.to(device)
        
#         # Generate fake images using the trained generator
#         fake_images = loaded_model(x)

#         # Convert torch tensor to NumPy array
#         fake_np = torchvision.utils.make_grid(fake_images, normalize=True).cpu().numpy().transpose(1, 2, 0)

#         # Display the fake image (you can save it to the output directory as well)
#         plt.imshow(fake_np)
#         plt.title(f"Fake Image {i + 1}")
#         plt.pause(0.1)  # Pause for a short duration to create the animation effect
#         ims.append([plt.imshow(fake_np, animated=True)])

# # Create the animation
# ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

# # Display the animation
# plt.rcParams['animation.embed_limit'] = 50  # Set to 50 MB, for example
# HTML(ani.to_jshtml())

# print('Testing complete!')

########################################################################################################################################################################################
# import torch
# from torch.utils.data import DataLoader
# from dataset import anime
# from generator_model import Generator
# from utils import load_checkpoint
# import config
# from torchvision.utils import save_image
# import os

# def test():
#     # Initialize generator model
#     gen = Generator(in_channels=3, features=64).to(config.DEVICE)

#     # Create a dummy optimizer (as it is not needed during testing)
#     dummy_optimizer = torch.optim.Adam(gen.parameters(), lr=0.001)

#     # Load the pre-trained generator checkpoint
#     load_checkpoint(config.CHECKPOINT_GEN, gen, dummy_optimizer, config.LEARNING_RATE)

#     # Switch to evaluation mode
#     gen.eval()

#     # Create a dataset and data loader for testing
#     test_dataset = anime(root_dir=config.TEST_DIR)  # Change to your test dataset directory
#     test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

#     # Create a folder to save the generated images
#     output_folder = "test_results"
#     os.makedirs(output_folder, exist_ok=True)

#     # Test the generator and save the generated images
#     for idx, (x, y) in enumerate(test_loader):
#         x = x.to(config.DEVICE)
#         y = y.to(config.DEVICE)

#         # Generate fake images
#         with torch.no_grad():
#             y_fake = gen(x)

#         # Save the generated images
#         save_image(y_fake, os.path.join(output_folder, f"generated_{idx}.png"))
#         save_image(x, os.path.join(output_folder, f"input_{idx}.png"))
#         save_image(y, os.path.join(output_folder, f"target_{idx}.png"))

#     print("Testing completed. Generated images saved in 'test_results' folder.")

# if __name__ == "__main__":
#     test()

####################################################################################################################################################################################

import torch
from torch.utils.data import DataLoader
from dataset import anime
from generator_model import Generator
from utils import load_checkpoint
import config
from torchvision.utils import save_image, make_grid
import os

def save_combined_images(input, target, generated, index, folder):
    combined_images = torch.cat([input, target, generated], dim=2)
    save_image(combined_images, os.path.join(folder, f"combined_{index}.png"))

def test():
    # Initialize generator model
    gen = Generator(in_channels=3, features=64).to(config.DEVICE)

    # Create a dummy optimizer (as it is not needed during testing)
    dummy_optimizer = torch.optim.Adam(gen.parameters(), lr=0.001)

    # Load the pre-trained generator checkpoint
    load_checkpoint(config.CHECKPOINT_GEN, gen, dummy_optimizer, config.LEARNING_RATE)

    # Switch to evaluation mode
    gen.eval()

    # Create a dataset and data loader for testing
    test_dataset = anime(root_dir=config.TEST_DIR)  # Change to your test dataset directory
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Create a folder to save the generated images
    output_folder = "test_results"
    os.makedirs(output_folder, exist_ok=True)

    # Test the generator and save the generated images in a single grid
    for idx, (x, y) in enumerate(test_loader):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)

        # Generate fake images
        with torch.no_grad():
            y_fake = gen(x)

        # Save the input, target, and generated images separately
        save_image(x, os.path.join(output_folder, f"input_{idx}.png"))
        save_image(y, os.path.join(output_folder, f"target_{idx}.png"))
        save_image(y_fake, os.path.join(output_folder, f"generated_{idx}.png"))

        # Save the combined images in a single grid
        save_combined_images(x, y, y_fake, idx, output_folder)

    print("Testing completed. Images saved in 'test_results' folder.")

if __name__ == "__main__":
    test()
