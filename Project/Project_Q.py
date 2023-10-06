
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym
import cv2 as cv
import numpy as np
from collections import deque
import random
import string

###################################Tensorflow Imports#############################################
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model

# Constants and hyperparameters
NUM_ACTIONS = 6  # Number of possible actions in Super Mario
NUM_OBSERVATIONS = 4  # Number of frames to stack as input
GAMMA = 0.99  # Discount factor
MEMORY_SIZE = 30000  # Replay memory size
BATCH_SIZE = 256  # Mini-batch size for experience replay
EPSILON_START = 1  # Initial exploration rate
EPSILON_END = 0.02  # Final exploration rate
EPSILON_DECAY_STEPS = 10000  # Exploration rate decay steps
LEARNING_RATE = 0.00025  # Learning rate for Q-network
MAX_EPISODES = 10000000  # Maximum number of episodes
MAX_STEPS_PER_EPISODE = 10000000  # Maximum number of steps per episode

################################################################################

# change these values if you want more/less printing
PRINT_GRID      = True
PRINT_LOCATIONS = False

# If printing the grid doesn't display in an understandable way, change the
# settings of your terminal (or anaconda prompt) to have a smaller font size,
# so that everything fits on the screen. Also, use a large terminal window /
# whole screen.

# other constants (don't change these)
SCREEN_HEIGHT   = 240
SCREEN_WIDTH    = 256
MATCH_THRESHOLD = 0.9

################################################################################
# TEMPLATES FOR LOCATING OBJECTS

# ignore sky blue colour when matching templates
MASK_COLOUR = np.array([252, 136, 104])
# (these numbers are [BLUE, GREEN, RED] because opencv uses BGR colour format by default)

# You can add more images to improve the object locator, so that it can locate
# more things. For best results, paint around the object with the exact shade of
# blue as the sky colour. (see the given images as examples)
#
# Put your image filenames in image_files below, following the same format, and
# it should work fine.

# filenames for object templates
image_files = {
    "mario": {
        "small": ["marioA.png", "marioB.png", "marioC.png", "marioD.png",
                  "marioE.png", "marioF.png", "marioG.png"],
        "tall": ["tall_marioA.png", "tall_marioB.png", "tall_marioC.png"],
        # Note: Many images are missing from tall mario, and I don't have any
        # images for fireball mario.
    },
    "enemy": {
        "goomba": ["goomba.png"],
        "koopa": ["koopaA.png", "koopaB.png"],
    },
    "block": {
        "block": ["block1.png", "block2.png", "block3.png", "block4.png"],
        "question_block": ["questionA.png", "questionB.png", "questionC.png"],
        "pipe": ["pipe_upper_section.png", "pipe_lower_section.png"],
    },
    "item": {
        # Note: The template matcher is colourblind (it's using greyscale),
        # so it can't tell the difference between red and green mushrooms.
        "mushroom": ["mushroom_red.png"],
        # There are also other items in the game that I haven't included,
        # such as star.

        # There's probably a way to change the matching to work with colour,
        # but that would slow things down considerably. Also, given that the
        # red and green mushroom sprites are so similar, it might think they're
        # the same even if there is colour.
    }
}

def _get_template(filename):
    image = cv.imread(filename)
    assert image is not None, f"File {filename} does not exist."
    template = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    mask = np.uint8(np.where(np.all(image == MASK_COLOUR, axis=2), 0, 1))
    num_pixels = image.shape[0]*image.shape[1]
    if num_pixels - np.sum(mask) < 10:
        mask = None # this is important for avoiding a problem where some things match everything
    dimensions = tuple(template.shape[::-1])
    return template, mask, dimensions

def get_template(filenames):
    results = []
    for filename in filenames:
        results.append(_get_template(filename))
    return results

def get_template_and_flipped(filenames):
    results = []
    for filename in filenames:
        template, mask, dimensions = _get_template(filename)
        results.append((template, mask, dimensions))
        results.append((cv.flip(template, 1), cv.flip(mask, 1), dimensions))
    return results

# Mario and enemies can face both right and left, so I'll also include
# horizontally flipped versions of those templates.
include_flipped = {"mario", "enemy"}

# generate all templatees
templates = {}
for category in image_files:
    category_items = image_files[category]
    category_templates = {}
    for object_name in category_items:
        filenames = category_items[object_name]
        if category in include_flipped or object_name in include_flipped:
            category_templates[object_name] = get_template_and_flipped(filenames)
        else:
            category_templates[object_name] = get_template(filenames)
    templates[category] = category_templates

################################################################################
# LOCATING OBJECTS

def _locate_object(screen, templates, stop_early=False, threshold=MATCH_THRESHOLD):
    locations = {}
    for template, mask, dimensions in templates:
        results = cv.matchTemplate(screen, template, cv.TM_CCOEFF_NORMED, mask=mask)
        locs = np.where(results >= threshold)
        for y, x in zip(*locs):
            locations[(x, y)] = dimensions

        # stop early if you found mario (don't need to look for other animation frames of mario)
        if stop_early and locations:
            break
    
    #      [((x,y), (width,height))]
    return [( loc,  locations[loc]) for loc in locations]

def _locate_pipe(screen, threshold=MATCH_THRESHOLD):
    upper_template, upper_mask, upper_dimensions = templates["block"]["pipe"][0]
    lower_template, lower_mask, lower_dimensions = templates["block"]["pipe"][1]

    # find the upper part of the pipe
    upper_results = cv.matchTemplate(screen, upper_template, cv.TM_CCOEFF_NORMED, mask=upper_mask)
    upper_locs = list(zip(*np.where(upper_results >= threshold)))
    
    # stop early if there are no pipes
    if not upper_locs:
        return []
    
    # find the lower part of the pipe
    lower_results = cv.matchTemplate(screen, lower_template, cv.TM_CCOEFF_NORMED, mask=lower_mask)
    lower_locs = set(zip(*np.where(lower_results >= threshold)))

    # put the pieces together
    upper_width, upper_height = upper_dimensions
    lower_width, lower_height = lower_dimensions
    locations = []
    for y, x in upper_locs:
        for h in range(upper_height, SCREEN_HEIGHT, lower_height):
            if (y+h, x+2) not in lower_locs:
                locations.append(((x, y), (upper_width, h), "pipe"))
                break
    return locations

def locate_objects(screen, mario_status):
    # convert to greyscale
    screen = cv.cvtColor(screen, cv.COLOR_BGR2GRAY)

    # iterate through our templates data structure
    object_locations = {}
    for category in templates:
        category_templates = templates[category]
        category_items = []
        stop_early = False
        for object_name in category_templates:
            # use mario_status to determine which type of mario to look for
            if category == "mario":
                if object_name != mario_status:
                    continue
                else:
                    stop_early = True
            # pipe has special logic, so skip it for now
            if object_name == "pipe":
                continue
            
            # find locations of objects
            results = _locate_object(screen, category_templates[object_name], stop_early)
            for location, dimensions in results:
                category_items.append((location, dimensions, object_name))

        object_locations[category] = category_items

    # locate pipes
    object_locations["block"] += _locate_pipe(screen)

    return object_locations

################################################################################

def reward_function(info, prev_info):                                           #return reward for a given movement, ####weight needs tuning####
    mario_life = info["life"]
    mario_flag = info["flag_get"]
    mario_world_x = info["x_pos"]
    mario_world_y = info["y_pos"]

    mario_life_prev = 0
    mario_flag_prev = 0
    mario_world_x_prev = 0
    mario_world_y_prev = 0

    if prev_info != None:
        mario_life_prev = prev_info["life"]
        mario_flag_prev = prev_info["flag_get"]
        mario_world_x_prev = prev_info["x_pos"]
        mario_world_y_prev = prev_info["y_pos"]
    else:
        return 0

    moving_reward = 0
    if mario_world_x_prev < mario_world_x:                                        #The reward for jumping, might need to implemet checks for justified jumps
        moving_reward = 1
    else:
        moving_reward = -1

    jumping_reward = 0
    if mario_world_y_prev > mario_world_y:                                        #The reward for jumping, might need to implemet checks for justified jumps
        jumping_reward = 0.2
    else:
        jumping_reward = 0

    living_reward = 0
    if mario_life < mario_life_prev:                                              #The reward for staying alive, dying is not an option
        living_reward = -15
    else:
        living_reward = 0

    reward = moving_reward + jumping_reward + living_reward

    return reward

#
#def frame_to_input(frame):                                                          #Convert a preprocessed frame into a TensorFlow Keras Input object
    # Create an Input object with the shape of the frame
#    input_layer = Input(shape=frame.shape, dtype=tf.uint8, name="frame_input")

    # Convert the Input object to a float32 tensor
#    input_tensor = tf.cast(input_layer, tf.float32)

    # Normalize the input tensor to the range [0, 1] (if needed)
    # input_tensor /= 255.0

#    return input_tensor
#

def preprocess_frame(frame):                                                       #Preprocess a frame by resizing it to 84x84 pixels and converting it to grayscale
    # Resize the frame to 84x84 pixels using cubic interpolation
    resized_frame = cv.resize(frame, (84, 84), interpolation=cv.INTER_CUBIC)

    # Convert the frame to grayscale
    grayscale_frame = cv.cvtColor(resized_frame, cv.COLOR_RGB2GRAY)

    # Normalize pixel values to the range [0, 255]
    normalized_frame = np.uint8(grayscale_frame)

    return normalized_frame


def scale_coordinates(x, y, original_frame_size, target_frame_size):
    original_width, original_height = original_frame_size
    target_width, target_height = target_frame_size

    scale_x = target_width / original_width
    scale_y = target_height / original_height

    scaled_x = int(int(x) * scale_x)
    scaled_y = int(int(y) * scale_y)

    return (scaled_x, scaled_y)


def preprocess_cv_coordinates(x, y):
    return scale_coordinates(x, y, (256, 240), (84, 84))


def preprocess_extra_coordinates(x, y, block_locations):
    for block in block_locations:
        if block[0][0] - x >= 0 and block[0][0] - x < 30 and block[0][1] - y >=0 and block[0][1] - y < 5:
            return scale_coordinates(block[0][0], block[0][1], (256, 240), (84, 84))

    return (0, 0)


def state_processing(state, info):
    mario_status = info["status"]
    object_locations = locate_objects(state, mario_status)
    block_locations = object_locations['block']
    mario_locations = object_locations['mario']
    x = 0
    y = 0
    if len(mario_locations) > 0:
        location, dimensions, object_name = mario_locations[0]
        x, y = location
    return x, y, block_locations


def conv_model():                                                                         #Using a hybird approach of both downsized frames and open cv data. Implements the model initilisation, training loop and model update loop, implementes Q update function within the model training loop.
    frame_input = Input(shape=(84, 84, NUM_OBSERVATIONS))
    mario_coords_input = Input(shape=(2,), name="cv_coords")
    objects_coords_input = Input(shape=(2,), name="extra_coords")

    # Image processing branch
    conv1 = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(frame_input)
    conv2 = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(conv1)
    conv3 = Conv2D(64, (3, 3), activation='relu')(conv2)
    flat1 = Flatten()(conv3)

    # Coordinate processing branch
    coord_processing = Concatenate()([mario_coords_input, objects_coords_input])

    # Merge branches
    merged = Concatenate()([flat1, coord_processing])
    
    dense1 = Dense(512, activation='relu')(merged)
    output_layer = Dense(NUM_ACTIONS)(dense1)
    return tf.keras.Model(inputs=[frame_input, mario_coords_input, objects_coords_input], outputs=output_layer)


# Define the experience replay buffer
def experience_buffer(max_size):
    return deque(maxlen=max_size)


# Epsilon-greedy action selection
def epsilon_greedy_action(q_network, state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(NUM_ACTIONS)
    q_values = q_network.predict(state)
    return np.argmax(q_values)


def update_double_q_values(q_network1, q_network2, target_network1, target_network2, replay_buffer):
    if len(replay_buffer) < BATCH_SIZE:
        return
    
    # Sample a mini-batch from the replay buffer
    mini_batch = random.sample(replay_buffer, BATCH_SIZE)
    
    # Prepare input data
    frame_inputs = np.array([sample[0] for sample in mini_batch])
    cv_coords_inputs = np.array([sample[1] for sample in mini_batch])
    extra_coords_inputs = np.array([sample[2] for sample in mini_batch])
    
    # Compute Q-values for current and next states using both Q-networks
    current_q_values1 = q_network1.predict([frame_inputs, cv_coords_inputs, extra_coords_inputs])
    current_q_values2 = q_network2.predict([frame_inputs, cv_coords_inputs, extra_coords_inputs])
    
    next_frame_inputs = np.array([sample[5] for sample in mini_batch])
    next_cv_coords_inputs = np.array([sample[6] for sample in mini_batch])
    next_extra_coords_inputs = np.array([sample[7] for sample in mini_batch])
    
    next_q_values1 = target_network1.predict([next_frame_inputs, next_cv_coords_inputs, next_extra_coords_inputs])
    next_q_values2 = target_network2.predict([next_frame_inputs, next_cv_coords_inputs, next_extra_coords_inputs])
    
    # Compute target Q-values by combining Q-values from both networks
    targets = np.zeros((BATCH_SIZE, NUM_ACTIONS))
    
    for i in range(BATCH_SIZE):
        _, _, _, action, reward, _, _, _, done = mini_batch[i]
        if done:
            targets[i, action] = reward
        else:
            if random.random() < 0.5:
                targets[i, action] = reward + GAMMA * next_q_values1[i, np.argmax(current_q_values2[i])]
            else:
                targets[i, action] = reward + GAMMA * next_q_values2[i, np.argmax(current_q_values1[i])]
    
    # Update Q-networks
    q_network1.fit([frame_inputs, cv_coords_inputs, extra_coords_inputs], targets, verbose=0)
    q_network2.fit([frame_inputs, cv_coords_inputs, extra_coords_inputs], targets, verbose=0)


def step_action(env, action, skip, info, prev_info):
    total_reward = 0.0
    frames = []
    for i in range(skip):
        # Accumulate reward and repeat the same action
        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(preprocess_frame(obs))
        done = terminated or truncated
        if prev_info != None:
            total_reward += reward_function(info, prev_info) * 2 + reward * 1                                       #Linear weighted sum of rewards, def needs tuning
        else:
            total_reward += reward

        if done:
            break

    stacked_frames = np.stack(frames, axis=-1)

    return obs, total_reward, done, info, stacked_frames


# Main training loop
def train_agent():
    # Initialize the environment
    env = gym.make("SuperMarioBros-v0", apply_api_compatibility=True, render_mode="human")
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    
    next_state = None
    done = True
    env.reset()

    # Create two pairs of Q-networks: one for action selection (online) and one for estimating target Q-values
    q_network1 = conv_model()
    q_network2 = conv_model()
    target_network1 = conv_model()
    target_network2 = conv_model()
    target_network1.set_weights(q_network1.get_weights())
    target_network2.set_weights(q_network2.get_weights())

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    loss_fn = tf.keras.losses.MeanSquaredError()

    q_network1.compile(optimizer=optimizer, loss=loss_fn)
    q_network2.compile(optimizer=optimizer, loss=loss_fn)

    # Create the experience replay buffer
    replay_buffer = experience_buffer(MEMORY_SIZE)

    epsilon = EPSILON_START

    for episode in range(MAX_EPISODES):
        state = env.reset()
        action = 1
        state, reward, done, info, stacked_frames = step_action(env, action, NUM_OBSERVATIONS, None, None)
        prev_info = None
        episode_reward = 0

        for step in range(MAX_STEPS_PER_EPISODE):
            x, y, block_locations = state_processing(state, info)
            # Choose an action using epsilon-greedy strategy
            frame = stacked_frames
            cv_coords = preprocess_cv_coordinates(x, y)
            extra_coords = preprocess_extra_coordinates(x, y, block_locations)  # Additional coordinates
            state_input = [frame, cv_coords, extra_coords]
            #print(state_input)
            action = epsilon_greedy_action(q_network1, state_input, epsilon)

            # Take the chosen action and observe the next state and reward
            next_state, reward, done, info, stacked_frames = step_action(env, action, NUM_OBSERVATIONS, info, prev_info)

            x1, y1, block_locations1 = state_processing(next_state, info)

            next_frame = stacked_frames
            next_cv_coords = preprocess_cv_coordinates(x1, y1)
            next_extra_coords = preprocess_extra_coordinates(x1, y1, block_locations1)
            episode_reward += reward

            # Store the experience in the replay buffer
            replay_buffer.append((frame, cv_coords, extra_coords, action, reward, next_frame, next_cv_coords, next_extra_coords, done))

            # Update the Q-network
            update_double_q_values(q_network1, q_network2, target_network1, target_network2, replay_buffer)

            # Update the target network periodically
            if step % 100 == 0:
                target_network1.set_weights(q_network1.get_weights())
                target_network2.set_weights(q_network2.get_weights())

            state = next_state

            prev_info = info

            if done:
                break

        # Decay epsilon to reduce exploration over time
        if epsilon > EPSILON_END:
            epsilon -= (EPSILON_START - EPSILON_END) / EPSILON_DECAY_STEPS

        print(f"Episode {episode + 1}: Reward = {episode_reward}")

    # Save the trained Q-network
    q_network1.save("mario_q_network1.h5")
    q_network2.save("mario_q_network2.h5")


# Start training the agent
train_agent()
