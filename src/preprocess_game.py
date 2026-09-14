import subprocess
from time import time

def run_gnubg_command(commands):
    """
    Runs GNU Backgammon commands inside the Docker container with minimal output.
    """
    gnubg_setup = """
    set display off
    set automatic game off
    set player 0 human
    set player 1 human
    new game
    """  # Disable AI playing and dice rolling

    full_command = f"{gnubg_setup}\n{commands}"

    docker_command = f"docker exec gnubg-container bash -c \"echo '{full_command}' | gnubg -t\""
    result = subprocess.run(docker_command, shell=True, capture_output=True, text=True)
    return result.stdout.strip()


def is_move_legal(position_id, dice, move):
    """
    Checks if a move is legal in GNU Backgammon without AI interference.
    """
    gnubg_commands = f"""
    set board {position_id}
    set dice {dice[0]} {dice[1]}
    move {move}
    """

    output = run_gnubg_command(gnubg_commands)

    # Print the output to verify it's minimal and accurate
    print("GNUBG Output:\n", output)

    # Check for common illegal move messages
    illegal_keywords = [
        "illegal move",
        "error",
        "you can't move that checker",
        "invalid",
        "cannot move",
        "(no game)"
    ]

    return not any(keyword in output.lower() for keyword in illegal_keywords)



# Example Usage
position_id = "XGID=-a-B-aE-C---bB---c-b-bB-:0:0:1:00:0:0:3:0:10"
dice_roll = (3, 4)

# Valid Move
valid_move = "24/21 13/9"
start = time()
print(f"Move {valid_move} is legal:", is_move_legal(position_id, dice_roll, valid_move))
end = time()
print(end-start)

# Invalid Move (Tile 50 does not exist)
# invalid_move = "24/21 13/50"
# print(f"Move {invalid_move} is legal:", is_move_legal(position_id, dice_roll, invalid_move))

"""
new game
set player 0 human
set player 1 human
set automatic game off
set cube use off
set dice 1 6 [dice 1 to 6]
set turn 0 [player '0'=gnubg, '1'=root]
move [bar/22, 19/22(2) - two checkers, 19/22 22/5]
Board position:
set board simple 0 2 4 0 -3 -5 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2
[i=0: white on bar | i=-1: black on bar | pos: white in tile | neg: black in tile]
"""