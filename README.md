# Deep Q learning play Tetris with interactive plots (WIP)

<p align="center">
    <img src="figures/record.gif" />
</p>


# Tried Combinations

| States      | Action Space | Tile Fall | Current Results |
| ----------- | -----------  | --------- | --------- |
| (cleared rows, holes, bumpiness, heights) | (up, down, right, left) | True | Fail |
| RGB frames   | (up, down, right, left) | True | Fail |
| RGB frames   | (up, right, left) | True | Fail |
| Binary grids   | (up, down, right, left) | True | Fail |
| Binary grids   | (up, right, left) | True | Fail |


# Train
```bash
    $ python main.py
```

# Play
```bash
    $ python play.py
```
Remember to change `fall_speed` in `tetris/constants.py` to higher number.
`n` get new tile<br>
`r` reset whole game<br>
`q`/`esc` quit game


# TODO

[ ] Weighted heights
[ ] Simplified action space
[ ] 1 x cols Convolution
[ ] DDQN
[ ] Skip Frames
[ ] More algorithm


# References
[https://github.com/uvipen/Tetris-deep-Q-learning-pytorch](https://github.com/uvipen/Tetris-deep-Q-learning-pytorch)
[https://codemyroad.wordpress.com/2013/04/14/tetris-ai-the-near-perfect-player/](https://codemyroad.wordpress.com/2013/04/14/tetris-ai-the-near-perfect-player/)
[https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html](https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html):w
