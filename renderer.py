import pickle
import numpy as np
import jax
import jax.numpy as jnp
import chex
import matplotlib.pyplot as plt

from functools import partial

def point_in_rect(xmin, xmax, ymin, ymax):
    def fn(x, y):
        return x >= xmin and x <= xmax and y >= ymin and y <= ymax
    return fn

def fill_coords(img, fn, color):
    new_img = img.copy()
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            yf = (y + 0.5) / img.shape[0]
            xf = (x + 0.5) / img.shape[1]
            if fn(xf, yf):
                new_img[y, x] = color
    return new_img

def point_in_triangle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    def fn(x, y):
        v0 = c - a
        v1 = b - a
        v2 = np.array((x, y)) - a

        # Compute dot products
        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot02 = np.dot(v0, v2)
        dot11 = np.dot(v1, v1)
        dot12 = np.dot(v1, v2)

        # Compute barycentric coordinates
        inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom

        # Check if point is in triangle
        return (u >= 0) and (v >= 0) and (u + v) < 1

    return fn


def add_border(tile):
    new_tile = fill_coords(tile, point_in_rect(
        0, 0.031, 0, 1), (100, 100, 100))
    return fill_coords(new_tile, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))


def replace_color(image, old_color, new_color):
    # Convert the image and colors to JAX arrays if they aren't already
    image = jnp.array(image)
    old_color = jnp.array(old_color)
    new_color = jnp.array(new_color)

    # Create a mask where all pixels match the old_color
    mask = jnp.all(image == old_color, axis=-1)

    # Replace the color
    new_image = image.at[mask].set(new_color)

    return new_image

def load_image_dict(file: str):

    with open(file, 'rb') as f:
        image_dict = pickle.load(f)

    tile_size = image_dict['images'].shape[-2]

    images = image_dict['images']

    image_dict['images'] = np.array(
        [replace_color(images[i], (255, 255, 255), (0,0,0)) for i in range(len(images))]
    )

    extra_keys = [
        ('wall', np.tile([100, 100, 100], (tile_size, tile_size, 1))),
        ('empty', add_border(np.tile([0, 0, 0], (tile_size, tile_size, 1)))),
    ]

    for key, img in extra_keys:
        assert not key in image_dict['keys']
        image_dict['keys'] = [key] + image_dict['keys']
        image_dict['images'] = jnp.concatenate((img[None], image_dict['images']))

    return image_dict

def create_image_grid_from_image_tensor(images, max_cols: int = 10):
    num_images = images.shape[0]
    img_height, img_width, channels = images[0].shape

    # Calculate the number of rows and columns
    num_cols = min(max_cols, num_images)
    num_rows = (num_images + num_cols - 1) // num_cols  # Ceiling division

    # Create a blank canvas
    grid_height = num_rows * img_height
    grid_width = num_cols * img_width
    canvas = jnp.zeros((grid_height, grid_width, channels), dtype=images.dtype)

    # Place each image in the canvas
    for idx, image in enumerate(images):
        row = idx // num_cols
        col = idx % num_cols
        row_start = row * img_height
        row_end = row_start + img_height
        col_start = col * img_width
        col_end = col_start + img_width

        canvas = canvas.at[row_start:row_end, col_start:col_end, :].set(image)

    return canvas


def make_agent_tile(direction: int, tile_size: int):
    TRI_COORDS = np.array([
        [0.12, 0.19],
        [0.87, 0.50],
        [0.12, 0.81],
    ])

    agent_tile = np.tile([0, 0, 0], (tile_size, tile_size, 1))
    agent_tile = fill_coords(
        agent_tile, point_in_triangle(*TRI_COORDS), [255, 0, 0])
    add_border = lambda x:x
    if direction == 0:
        return add_border(agent_tile)  # right
    elif direction == 1:
        return add_border(np.rot90(agent_tile, k=3))  # down
    elif direction == 2:
        return add_border(np.rot90(agent_tile, k=2))  # left
    elif direction == 3:
        return add_border(np.rot90(agent_tile, k=1))  # up

def create_image_from_grid(grid, agent_pos, agent_dir, image_dict):
    # Assumes wall_index is the index for the wall image in image_dict['images']
    wall_index = image_dict['keys'].index('wall')

    # Expand grid size by 2 in each direction (top, bottom, left, right)
    H, W = grid.shape[:2]
    new_H, new_W = H + 2, W + 2

    # Create a new grid with wall index
    new_grid = jnp.full((new_H, new_W, 1), wall_index, dtype=grid.dtype)

    # Place the original grid in the center of the new grid
    new_grid = new_grid.at[1:H+1, 1:W+1, :].set(grid)

    # Flatten the grid for easier indexing
    new_grid_flat = new_grid.squeeze()  # Removes the last dimension assuming it's 1

    # Retrieve the images tensor
    images = image_dict['images']

    # Use advanced indexing to map the grid indices to actual images
    mapped_images = jax.vmap(lambda x: images[x])(new_grid_flat)

    # Create the agent tile with the specified direction
    tile_size = images.shape[-2]
    agent_tile = make_agent_tile(agent_dir, tile_size)

    # Adjust agent position to account for the expanded grid
    agent_x, agent_y = agent_pos
    agent_x += 1
    agent_y += 1

    # Dimensions of the new grid
    # Assuming all images are of the same shape and channel number
    img_H, img_W, C = images.shape[1:]

    # Reshape and transpose to form the single image
    # First, reshape mapped_images to (new_H, new_W, img_H, img_W, C)
    reshaped_images = mapped_images.reshape(new_H, new_W, img_H, img_W, C)
    reshaped_images = reshaped_images.at[agent_y, agent_x].set(agent_tile)

    # Then, transpose to (new_H, img_H, new_W, img_W, C)
    transposed_images = reshaped_images.transpose(0, 2, 1, 3, 4)
    # Finally, reshape to (new_H * img_H, new_W * img_W, C)
    final_image = transposed_images.reshape(new_H * img_H, new_W * img_W, C)

    return final_image
