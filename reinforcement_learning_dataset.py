import os
import glob
import gym
import textworld.gym


def get_training_game_env_1level100game(data_dir, difficulty_level, requested_infos, max_episode_steps, batch_size):
    assert difficulty_level in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    training_size = 100
    game_file_names = []
    game_path = data_dir + "/train_" + str(training_size) + "/difficulty_level_" + str(difficulty_level)
    if os.path.isdir(game_path):
        game_file_names += glob.glob(os.path.join(game_path, "*.z8"))
    else:
        game_file_names.append(game_path)
    game_file_names_touse = sorted(game_file_names)
    env_id = textworld.gym.register_games(game_file_names_touse, request_infos=requested_infos,
                                          max_episode_steps=max_episode_steps, batch_size=batch_size,
                                          name="training", asynchronous=True, auto_reset=False)
    env = gym.make(env_id)
    num_game = len(game_file_names_touse)
    print("Training: level {}, {} games".format(difficulty_level, num_game))
    return env, num_game

def get_training_game_env_1level25game(data_dir, difficulty_level, requested_infos, max_episode_steps, batch_size):
    assert difficulty_level in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    training_size = 100
    game_file_names = []
    game_path = data_dir + "/train_" + str(training_size) + "/difficulty_level_" + str(difficulty_level)
    if os.path.isdir(game_path):
        game_file_names += glob.glob(os.path.join(game_path, "*.z8"))
    else:
        game_file_names.append(game_path)
    game_file_names_touse = sorted(game_file_names)[:25]
    env_id = textworld.gym.register_games(game_file_names_touse, request_infos=requested_infos,
                                          max_episode_steps=max_episode_steps, batch_size=batch_size,
                                          name="training", asynchronous=True, auto_reset=False)
    env = gym.make(env_id)
    num_game = len(game_file_names_touse)
    print("Training: level {}, {} games".format(difficulty_level, num_game))
    return env, num_game

def get_training_game_env(data_dir, difficulty_level, training_size, requested_infos, max_episode_steps, batch_size):
    assert difficulty_level in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    assert training_size in [1, 20, 100]
    game_file_names = []
    game_path = data_dir + "/train_" + str(training_size) + "/difficulty_level_" + str(difficulty_level)
    if os.path.isdir(game_path):
        game_file_names += glob.glob(os.path.join(game_path, "*.z8"))
    else:
        game_file_names.append(game_path)
    env_id = textworld.gym.register_games(sorted(game_file_names), request_infos=requested_infos,
                                          max_episode_steps=max_episode_steps, batch_size=batch_size,
                                          name="training", asynchronous=True, auto_reset=False)
    env = gym.make(env_id)
    num_game = len(game_file_names)
    return env, num_game
    
def get_training_game_env_4level25game(data_dir, requested_infos, max_episode_steps, batch_size):
    """
    Choose 25 games per level
    """
    training_size = 100
    difficulty_level_list = [3,7,5,9]
    game_file_names_list = []
    print("===== Building training env on 4 levels, 25 games per level ======")
    for difficulty_level in difficulty_level_list:
        game_file_names = []
        game_path = data_dir + "/train_" + str(training_size) + "/difficulty_level_" + str(difficulty_level)
        if os.path.isdir(game_path):
            game_file_names += glob.glob(os.path.join(game_path, "*.z8"))
        else:
            game_file_names.append(game_path)
        game_file_names_touse = sorted(game_file_names)[:25]
        print("Level {}, {} games".format(difficulty_level, len(game_file_names_touse)))
        game_file_names_list.extend(game_file_names_touse)
    print("Totally {} games".format(len(game_file_names_list)))
    env_id = textworld.gym.register_games(sorted(game_file_names_list), request_infos=requested_infos,
                                          max_episode_steps=max_episode_steps, batch_size=batch_size,
                                          name="training", asynchronous=True, auto_reset=False)
    env = gym.make(env_id)
    num_game = len(game_file_names_list)
    return env, num_game
    
def get_training_game_env_4level100game(data_dir, requested_infos, max_episode_steps, batch_size):
    """
    Choose 100 games per level
    """
    training_size = 100
    difficulty_level_list = [3,7,5,9]
    game_file_names_list = []
    print("===== Building training env on 4 levels, 100 games per level ======")
    for difficulty_level in difficulty_level_list:
        game_file_names = []
        game_path = data_dir + "/train_" + str(training_size) + "/difficulty_level_" + str(difficulty_level)
        if os.path.isdir(game_path):
            game_file_names += glob.glob(os.path.join(game_path, "*.z8"))
        else:
            game_file_names.append(game_path)
        game_file_names_touse = sorted(game_file_names)
        print("Level {}, {} games".format(difficulty_level, len(game_file_names_touse)))
        game_file_names_list.extend(game_file_names_touse)
    print("Totally {} games".format(len(game_file_names_list)))
    env_id = textworld.gym.register_games(sorted(game_file_names_list), request_infos=requested_infos,
                                          max_episode_steps=max_episode_steps, batch_size=batch_size,
                                          name="training", asynchronous=True, auto_reset=False)
    env = gym.make(env_id)
    num_game = len(game_file_names_list)
    return env, num_game


def get_evaluation_game_env(data_dir, difficulty_level, requested_infos, max_episode_steps, batch_size, valid_or_test="valid"):
    assert valid_or_test in ["valid", "test"]
    assert difficulty_level in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    game_file_names = []
    game_path = data_dir + "/" + valid_or_test + "/difficulty_level_" + str(difficulty_level)
    if os.path.isdir(game_path):
        game_file_names += glob.glob(os.path.join(game_path, "*.z8"))
    else:
        game_file_names.append(game_path)

    env_id = textworld.gym.register_games(sorted(game_file_names), request_infos=requested_infos,
                                          max_episode_steps=max_episode_steps, batch_size=batch_size,
                                          name="eval", asynchronous=True, auto_reset=False)
    env = gym.make(env_id)
    num_game = len(game_file_names)
    return env, num_game
