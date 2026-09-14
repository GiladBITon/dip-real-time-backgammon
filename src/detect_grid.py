import numpy as np
from itertools import permutations
import cv2


# def is_available_moves(game):
#     player_checkers, available_tiles = get_available_tiles(game)
#     dice = game["curr_dice"]
#     steps_used = 0
#     step_options = []
#     options = []
#     bar_checkers = player_checkers[0]
#
#     if bar_checkers > 0:
#         if dice[0] == dice[1]:
#             num = dice[0]
#             remain_steps = 0
#             if available_tiles[num] == 1:
#                 steps_used = bar_checkers * num # Can't eat more than 5 - otherwise use np.clip
#                 remain_steps = 4 - bar_checkers
#
#             if remain_steps == 3:
#                 options.append([3 * num])
#                 options.append([num, 2 * num])
#                 options.append(3 * [num])
#             elif remain_steps == 2:
#                 options.append([2 * num])
#                 options.append([2 * [num]])
#             elif remain_steps == 1:
#                 options.append([num])
#
#             moves = []
#             for option in options:
#                 for k, perm in enumerate(set(permutations(option))):
#                     moves.append(0)
#                     for step in perm:
#                         for i in range(25):
#                             if player_checkers[i] > 0:
#                                 if i + step < 25 and available_tiles[i + step] == 1:
#                                     moves[k] += step
#                                     player_checkers[i] += 1
#                                     player_checkers[i + step] -= 1
#                                     break
#
#             total_steps = steps_used + int(max(np.sum(move) for move in moves))
#             return (total_steps // num) * [num]
#
#         else:   # Different dice
#             if bar_checkers == 2: # Can only eat up to 2 checkers
#                 if available_tiles[dice[0]] == 1 and available_tiles[dice[1]] == 1:
#                     return dice
#                 elif available_tiles[dice[0]] == 1:
#                     return [dice[0]]
#                 elif available_tiles[dice[1]] == 1:
#                     return [dice[1]]
#             else: # 1 in bar
#                 if available_tiles[dice[0]] == 1:
#                     steps_used = [dice[0]]
#                     step = dice[1]
#                     for i in range(25):
#                         if player_checkers[i] > 0:
#                             if i + step < 25 and available_tiles[i + step] == 1:
#                                 return dice
#                     step_options.append(steps_used)
#
#                 if available_tiles[dice[1]] == 1:
#                     steps_used = [dice[1]]
#                     step = dice[0]
#                     for i in range(25):
#                         if player_checkers[i] > 0:
#                             if i + step < 25 and available_tiles[i + step] == 1:
#                                 return dice
#                     step_options.append(steps_used)
#
#                 return step_options
#
#     if dice[0] == dice[1]:  # Double dice roll
#         num = dice[0]
#         options.append([4 * num])
#         options.append([num, 3 * num])
#         options.append([2 * num, 2 * num])
#         options.append([num, num, 2 * num])
#         options.append(4 * [num])
#     else:  # Regular dice roll
#         options.append([int(np.sum(dice))])
#         options.append(dice)
#
#     moves = []
#     for s, option in enumerate(options):
#         for k, perm in enumerate(set(permutations(option))):
#             moves.append([])
#             for step in perm:
#                 for i in range(25):
#                     if player_checkers[i] > 0:
#                         if i + step < 25 and available_tiles[i + step] == 1:
#                             moves[k] += step
#                             player_checkers[i] += 1
#                             player_checkers[i + step] -= 1
#                             break
#
#     steps_used = int(max(np.sum(move) for move in moves))
#     if dice[0] == dice[1]:
#         return (steps_used // dice[0]) * [dice[0]]
#     else:
#         if steps_used == int(np.sum(dice)):
#             return dice
#         else:
#             return [steps_used]
#
#
# def is_available_moves(game):
#     player_checkers, available_tiles = get_available_tiles(game)
#     dice = game["curr_dice"]
#     options = []
#     dice_used = []
#     bar_checkers = player_checkers[0]
#     if bar_checkers > 0:
#         if dice[0] == dice[1]:
#             remain_steps = 0
#             if available_tiles[dice[0]] == 1:
#                 dice_used = bar_checkers * [dice[0]] # Can't eat more than 5 - otherwise use np.clip
#                 remain_steps = 4 - bar_checkers
#
#             num = dice[0]
#             if remain_steps == 3:
#                 options.append([3 * num])
#                 options.append([num, 2 * num])
#                 options.append(3 * [num])
#             elif remain_steps == 2:
#                 options.append([2 * num])
#                 options.append([2 * [num]])
#             elif remain_steps == 1:
#                 options.append([num])
#
#         else:   # Different dice
#             if bar_checkers == 2: # Can only eat up to 2 checkers
#                 if available_tiles[dice[0]] == 1 and available_tiles[dice[1]] == 1:
#                     moves = [(0, dice[0]), (0, dice[1])]
#                 elif available_tiles[dice[0]] == 1:
#                     moves = [(0, dice[0])]
#                 elif available_tiles[dice[1]] == 1:
#                     moves = [(0, dice[1])]
#             else: # 1 in bar
#                 if available_tiles[dice[0]] == 1:
#                     moves = [(0, dice[0])]
#                     options.append([dice[1]])
#                 if available_tiles[dice[1]] == 1:
#                     moves = [(0, dice[1])]          ######################
#                     options.append([dice[0]])
#
#
#     for option in options:
#         for k, perm in enumerate(set(permutations(option))):
#             moves.append([])
#             for step in perm:
#                 for i in range(len(start_tiles)):
#                     if start_tiles[i] > 0:
#                         if i + die < len(change_state) - 1 and change_state[i + die] > 0:
#                             moves[k].append((i, i + die))
#                             change_state[i] += 1
#                             change_state[i + die] -= 1
#                             break
#
#
# def is_legal_move(game):
#     # print("Checking move...")
#     # Check if dice are visible
#     if len(game["curr_dice"]) != 2:
#         # print("[ERROR] Dice missing")
#         return False
#
#     dice = game["curr_dice"]
#     change_state = get_change_state(game)
#     # print(f"Curr state: {game["curr_game_state"]}")
#     # print(f"Prev state: {game["prev_game_state"]}")
#     print(f"Change state: {change_state}")
#     moved_checkers = np.sum(change_state[change_state > 0])
#
#     steps = is_available_moves(game)
#
#     options = []
#     if dice[0] == dice[1]:  # Double dice roll
#         num = dice[0]
#         if moved_checkers == 1:
#             options.append([4 * num])
#         elif moved_checkers == 2:
#             options.append([num, 3 * num])
#             options.append([2 * num, 2 * num])
#         elif moved_checkers == 3:
#             options.append([num, num, 2 * num])
#         elif moved_checkers == 4:
#             options.append(4 * [num])
#     else:  # Regular dice roll
#         if moved_checkers == 1:
#             options.append([np.sum(dice)])
#         elif moved_checkers == 2:
#             options.append(dice)
#     print(f"Dice options: {options}")
#
#     moves = []
#     for option in options:
#         for k, perm in enumerate(set(permutations(option))):
#             moves.append([])
#             for die in perm:
#                 for i in range(25):
#                     # while change_state[i] < 0:
#                     if change_state[i] < 0:
#                         if can_bear_off(game):
#                             if i + die < 25 and change_state[i + die] > 0:
#                                 moves[k].append((i, i + die))
#                                 change_state[i] += 1
#                                 change_state[i + die] -= 1
#                                 break
#                             elif
#                         elif i + die < 25 and change_state[i + die] > 0:
#                             moves[k].append((i, i + die))
#                             change_state[i] += 1
#                             change_state[i + die] -= 1
#                             break
#
#     chosen_move = None
#     for move in moves:
#         if len(move) == moved_checkers:
#             chosen_move = move
#             break
#
#     if chosen_move is not None:
#         print(f"{game["turn"]} move: {chosen_move}")
#         return True
#
#     # print("No legal moves")
#     return False


# Define fixed window size

fixed_width, fixed_height = 2000, 1000

# Get image dimensions

# Create a black canvas
canvas = np.zeros((fixed_height, fixed_width, 3), dtype=np.uint8)


# Place resized image on canvas

# Display the image
cv2.imshow("Fixed Size Window", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()