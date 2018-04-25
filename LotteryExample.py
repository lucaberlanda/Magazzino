import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

# imagine we have to play a lottery game, with 20 balls in an urn; each lottery game consists in 3 balls drawn from the
# urn; We have the possibility to place three different "bets" on each drawn sample. We want to understand if it is
# better to choose all different numbers in the three bets or if we should keep some numbers fixed in order to maximize
# the chances of winning

urn = pd.DataFrame(np.arange(20))
drawn_balls = pd.DataFrame(np.zeros(shape=(10000, 3)))

for i in range(len(drawn_balls.index)):
    urn_extraction = urn.sample(n=3, replace=False).values.flatten()  # the drawn ball is not replaced
    drawn_balls.iloc[i, :]= urn_extraction

# bilondi decides to leave the
first_bet_bilondi = [0, 1, 2]
second_bet_bilondi = [3, 4, 5]
third_bet_bilondi = [6, 7, 8]

# gigi decides to keep the first to number fixed (she has those two lucky numbers) and to change only the last number;
first_bet_gigi = [0, 1, 2]
second_bet_gigi = [0, 3, 4]
third_bet_gigi = [0, 5, 6]

# ______________________________________________________________________________________________________________________

guessed_balls_bilondi_1 = drawn_balls.isin(first_bet_bilondi)
guessed_balls_bilondi_2 = drawn_balls.isin(second_bet_bilondi)
guessed_balls_bilondi_3 = drawn_balls.isin(third_bet_bilondi)

guessed_balls_bilondi_1 = guessed_balls_bilondi_1.loc[guessed_balls_bilondi_1[0] == True]
print(guessed_balls_bilondi_1)
guessed_balls_bilondi_1 = guessed_balls_bilondi_1.loc[guessed_balls_bilondi_1[1] == True]
print(guessed_balls_bilondi_1)
guessed_balls_bilondi_1 = guessed_balls_bilondi_1.loc[guessed_balls_bilondi_1[2] == True]
print(guessed_balls_bilondi_1)
quit()
guessed_balls_bilondi_2 = guessed_balls_bilondi_2.loc[guessed_balls_bilondi_2[0] == True]
guessed_balls_bilondi_2 = guessed_balls_bilondi_2.loc[guessed_balls_bilondi_2[1] == True]
guessed_balls_bilondi_2 = guessed_balls_bilondi_2.loc[guessed_balls_bilondi_2[2] == True]

guessed_balls_bilondi_3 = guessed_balls_bilondi_3.loc[guessed_balls_bilondi_3[0] == True]
guessed_balls_bilondi_3 = guessed_balls_bilondi_3.loc[guessed_balls_bilondi_3[1] == True]
guessed_balls_bilondi_3 = guessed_balls_bilondi_3.loc[guessed_balls_bilondi_3[2] == True]

first_bet_wins_bilondi = len(guessed_balls_bilondi_1)
second_bet_wins_bilondi = len(guessed_balls_bilondi_2)
third_bet_wins_bilondi = len(guessed_balls_bilondi_3)
total_wins_bilondi = first_bet_wins_bilondi + second_bet_wins_bilondi + third_bet_wins_bilondi

# ______________________________________________________________________________________________________________________

guessed_balls_gigi_1 = drawn_balls.isin(first_bet_gigi)
guessed_balls_gigi_2 = drawn_balls.isin(second_bet_gigi)
guessed_balls_gigi_3 = drawn_balls.isin(third_bet_gigi)

guessed_balls_gigi_1 = guessed_balls_gigi_1.loc[guessed_balls_gigi_1[0] == True]
guessed_balls_gigi_1 = guessed_balls_gigi_1.loc[guessed_balls_gigi_1[1] == True]
guessed_balls_gigi_1 = guessed_balls_gigi_1.loc[guessed_balls_gigi_1[2] == True]

guessed_balls_gigi_2 = guessed_balls_gigi_2.loc[guessed_balls_gigi_2[0] == True]
guessed_balls_gigi_2 = guessed_balls_gigi_2.loc[guessed_balls_gigi_2[1] == True]
guessed_balls_gigi_2 = guessed_balls_gigi_2.loc[guessed_balls_gigi_2[2] == True]

guessed_balls_gigi_3 = guessed_balls_gigi_3.loc[guessed_balls_gigi_3[0] == True]
guessed_balls_gigi_3 = guessed_balls_gigi_3.loc[guessed_balls_gigi_3[1] == True]
guessed_balls_gigi_3 = guessed_balls_gigi_3.loc[guessed_balls_gigi_3[2] == True]

first_bet_wins_gigi = len(guessed_balls_gigi_1)
second_bet_wins_gigi = len(guessed_balls_gigi_2)
third_bet_wins_gigi = len(guessed_balls_gigi_3)
total_wins_gigi = first_bet_wins_gigi + second_bet_wins_gigi + third_bet_wins_gigi

# ______________________________________________________________________________________________________________________

print('bilondi wins %d' % total_wins_bilondi)
print('gigi wins %d' % total_wins_gigi)
