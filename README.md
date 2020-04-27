# Tetris-on-CNN

.drawio files are used to draw diagram

## DevLog

---

## 2020-04-27

1. 

## 2020-04-26

1. Store the values, then every time we want we can train the model.  

### TODO

1. Modify the whole loop.
    * Begin: Generate data or train the model and watch how it performs.  
        * If generate: After gameover, ask keep generating or training the model.
        * If train: Display the amount of actions had taken. Only collect the data that get score.
2. Create new algorithm diagram.

## 2020-04-23

1. Balance the number of each action, need to avoid some has too many.
2. Try another structure of cnn.
3. Change optimizer and loss function let tiles took multiple actions.  
    * optimizer: __adam &rArr; adagrad__  
    * loss: __categorical_crossentropy &rArr; mean_square_error__  
4. Change the frequency of taking actions in cnn_main.
5. Only one action taken still happened.
6. Getting one response per loop seems too many.

---

## 2020-04-22

1. The 'game &hArr; cnn' loop can run without error.
2. Set a stop point in human control when there are more than 1000 data.
3. Add games count text.
4. Train per gameover is not effcient. Train every 1000 data are collected.
5. Just collect grids and action seems not enough.
6. Maybe collect 'NOTHING' action ?
7. Change the frequence of collecting data.

---

## 2020-04-21

1. Keep debuging on the keras cnn model.
2. Transform the format of data to be consistent with the format in keras.
3. Maybe tomorrow will success, maybe.

Make sure training properly.

---

## 2020-04-20

Successfully collect data that are desired.

1. feed data into cnn
2. whether use a wrapper or just put all codes together (think the former is better)
3. don't be afraid, just try it

---

## 2020-04-18

Need to modify collect_clear & collect_increase

1. collect_clear
just collect if tiles are clreared or not is not usable.
store each data that before one clear is a better way.

2. collect_increase
the socres are compared between two games, not in a game loop,
but if we just train cnn know how to clear,
the score increases or not will not be important.

Collect data of each lock, then collect each 'lock data' of each clear.

One game has many clear times, one clear has many lock times.

---

## 2020-04-17

Problem with collect_clear & collect_action

1. collect_clear: keep empty numpt array
2. collect_action: will be all space control (i.e. 4)

Temporary solution: think carefully of each frame, to know where to update data.

---

## Earlier

Construct Tetris and connect to keras.  
Use draw.io to draw some diagram help develop.
