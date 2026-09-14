# Real-Time Backgammon Detection and Tracking 

## Overview
This project focuses on the real-time detection and tracking of a physical backgammon game using computer vision techniques. The system automatically monitors the gameplay, identifies the state of the board, and determines legal moves—without requiring any hardware interaction from the players. 

The vision-based setup captures the game from a top-down view using a single fixed camera, processing the video feed to detect the game board, segment its grid into playable areas, recognize the position of each checker and dice, and analyze player interactions.

## Features
* **Game Board Detection:** Automatically locates and segments the game board from a top-down camera feed.
* **Pieces & Dice Recognition:** Detects the position of all 30 checkers and reads the dice values.
* **Game State Validation:** Integrates logic validation based on backgammon rules to ensure consistency between visual input and expected game flow.
* **Real-Time Processing:** Analyzes frames sequentially to track turns and game progression.

## System Setup
1. Physical backgammon board with 15 checkers per player and 2 dice.
2. Overhead camera (Resolution: 1280x720 @ 7.5 fps).
3. Normal room lighting (avoiding direct light spots on the board).

## Core Algorithms
This project is heavily based on classical image processing pipelines:
* **Gaussian & Median Blur:** Used for noise reduction and smoothing while preserving object edges.
* **Canny Edge Detection:** Identifies sharp intensity changes to detect the boundaries of the board and the checkers.
* **Morphological Operations (Dilation, Erosion, Closing):** Used to connect disjointed features, close contours, and filter out small noise artifacts.
* **Contour Detection (FindContours):** Extracts the boundaries of objects (checkers, dice) for shape analysis and localization.

## Repository Structure
```text
├── src/                    # Source code (algorithms and logic)
├── data/                   # Sample images 
├── docs/                   # Final project reports and documentation
├── assets/                 # Images used in the Final Project report
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
